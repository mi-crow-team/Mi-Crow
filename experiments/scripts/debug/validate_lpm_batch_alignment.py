#!/usr/bin/env python3
# ruff: noqa
"""
Validate LPM batch alignment and diagnose IndexError.

This script replicates the failing LPM experiment settings and validates:
1. Do attention mask batch indices match dataset batches?
2. Is _seen_samples incrementing correctly in LPM?
3. Are sequence lengths consistent for the same samples?

Failing experiment settings:
- Model: speakleash/Bielik-1.5B-v3.0-Instruct
- Train dataset: plmix_train
- Test dataset: plmix_test
- Aggregation: mean
- Metric: euclidean
- Layer: 31
- Device: cpu

Error:
IndexError: index 498 is out of bounds for dimension 0 with size 181

Usage:
    uv run python -m experiments.scripts.debug.validate_lpm_batch_alignment
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from experiments.models.lpm.lpm import LPM
from mi_crow.datasets import ClassificationDataset
from mi_crow.hooks import HookType
from mi_crow.hooks.implementations.model_input_detector import ModelInputDetector
from mi_crow.language_model.language_model import LanguageModel
from mi_crow.store import LocalStore
from mi_crow.utils import get_logger, set_seed

logger = get_logger(__name__)

# Experiment configuration (from failing run)
MODEL_ID = "speakleash/Bielik-1.5B-v3.0-Instruct"
TRAIN_DATASET = "plmix_train"
TEST_DATASET = "plmix_test"
AGGREGATION = "mean"
METRIC = "euclidean"
LAYER = 31
DEVICE = "cpu"
BATCH_SIZE = 64
MAX_LENGTH = 512

# Dataset configurations
DATASET_CONFIGS = {
    "plmix_train": {
        "store_path": "datasets/plmix_train",
        "text_field": "text",
        "category_field": "text_harm_label",
        "positive_label": "harmful",
        "lang": "pl",
    },
    "plmix_test": {
        "store_path": "datasets/plmix_test",
        "text_field": "text",
        "category_field": "text_harm_label",
        "positive_label": "harmful",
        "lang": "pl",
    },
}

# Activation run ID
ACTIVATION_RUN_ID = "activations_bielik_1_5b_v3_0_instruct_plmix_train_layer31_20260117_123845"


class BatchAlignmentValidator:
    """Validate batch alignment between attention masks and dataset."""

    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.validation_results = {
            "attention_mask_batches": [],
            "dataset_batches": [],
            "lpm_seen_samples_log": [],
            "sequence_length_comparison": [],
            "errors": [],
        }

    def _get_layer_signature(self, lm: LanguageModel) -> str:
        """Get layer signature for the specified layer."""
        layer_signature = f"llamaforcausallm_model_layers_{LAYER}"
        layer_names = lm.layers.get_layer_names()
        if layer_signature not in layer_names:
            raise ValueError(f"Layer '{layer_signature}' not found in model")
        return layer_signature

    def load_datasets(self) -> tuple[ClassificationDataset, ClassificationDataset]:
        """Load train and test datasets."""
        logger.info("=" * 80)
        logger.info("STEP 1: Load Datasets")
        logger.info("=" * 80)

        train_config = DATASET_CONFIGS[TRAIN_DATASET]
        test_config = DATASET_CONFIGS[TEST_DATASET]

        train_store = LocalStore(base_path=str(self.store_dir / train_config["store_path"]))
        train_dataset = ClassificationDataset.from_disk(
            store=train_store,
            text_field=train_config["text_field"],
            category_field=train_config["category_field"],
        )
        logger.info("✅ Train: %d samples", len(train_dataset))

        test_store = LocalStore(base_path=str(self.store_dir / test_config["store_path"]))
        test_dataset = ClassificationDataset.from_disk(
            store=test_store,
            text_field=test_config["text_field"],
            category_field=test_config["category_field"],
        )
        logger.info("✅ Test: %d samples", len(test_dataset))

        return train_dataset, test_dataset

    def train_lpm(self, train_dataset: ClassificationDataset) -> LPM:
        """Train LPM on pre-saved activations."""
        logger.info("=" * 80)
        logger.info("STEP 2: Train LPM")
        logger.info("=" * 80)

        train_config = DATASET_CONFIGS[TRAIN_DATASET]

        lpm = LPM(
            layer_number=LAYER,
            distance_metric=METRIC,
            aggregation_method=AGGREGATION,
            device=DEVICE,
            positive_label=train_config["positive_label"],
        )

        activations_store = LocalStore(base_path=str(self.store_dir))
        lpm.fit(
            store=activations_store,
            run_id=ACTIVATION_RUN_ID,
            dataset=train_dataset,
            model_id=MODEL_ID,
            category_field=train_config["category_field"],
            max_samples=None,
        )

        logger.info("✅ LPM trained")
        logger.info("   Prototypes: %s", list(lpm.lpm_context.prototypes.keys()))

        return lpm

    def save_and_inspect_attention_masks(self, test_dataset: ClassificationDataset) -> tuple[str, dict[str, Any]]:
        """Save attention masks and inspect their structure."""
        logger.info("=" * 80)
        logger.info("STEP 3: Save and Inspect Attention Masks")
        logger.info("=" * 80)

        model_store = LocalStore(base_path=str(self.store_dir))
        lm = LanguageModel.from_huggingface(MODEL_ID, store=model_store, device=DEVICE)

        layer_0_signature = self._get_layer_signature(lm).replace(f"_{LAYER}", "_0")

        run_name = "validate_attention_masks"
        logger.info("Run name: %s", run_name)

        # Setup detector
        attention_mask_layer_sig = "attention_masks"
        if attention_mask_layer_sig not in lm.layers.name_to_layer:
            lm.layers.name_to_layer[attention_mask_layer_sig] = lm.context.model

        detector = ModelInputDetector(
            layer_signature=attention_mask_layer_sig,
            hook_id=f"attention_mask_detector_{run_name}",
            save_input_ids=False,
            save_attention_mask=True,
        )
        hook_id = lm.layers.register_hook(attention_mask_layer_sig, detector, HookType.PRE_FORWARD)

        test_texts = test_dataset.get_all_texts()
        num_batches = (len(test_texts) + BATCH_SIZE - 1) // BATCH_SIZE

        logger.info("Processing %d samples in %d batches...", len(test_texts), num_batches)

        batch_info = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(test_texts))
            batch_texts = test_texts[start_idx:end_idx]
            batch_size_actual = len(batch_texts)

            lm.activations._process_batch(
                batch_texts,
                run_name=run_name,
                batch_index=batch_idx,
                max_length=MAX_LENGTH,
                autocast=False,
                autocast_dtype=None,
                dtype=None,
                verbose=False,
                save_in_batches=True,
                stop_after_layer=layer_0_signature,
            )

            # Inspect saved attention mask using LocalStore (same way as LPM does)
            try:
                # Load using the same method as load_inference_attention_masks()
                attention_mask_tensor = model_store.get_detector_metadata_by_layer_by_key(
                    run_name, batch_idx, attention_mask_layer_sig, "attention_mask"
                )

                info = {
                    "batch_idx": batch_idx,
                    "expected_samples": batch_size_actual,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "mask_shape": list(attention_mask_tensor.shape),
                    "mask_dtype": str(attention_mask_tensor.dtype),
                    "sample_texts_preview": [t[:50] + "..." if len(t) > 50 else t for t in batch_texts[:3]],
                }
                batch_info.append(info)
                logger.info(
                    "  Batch %d: samples %d-%d, mask shape %s",
                    batch_idx,
                    start_idx,
                    end_idx - 1,
                    attention_mask_tensor.shape,
                )
            except FileNotFoundError:
                logger.warning("  Batch %d: attention mask not found!", batch_idx)
            except Exception as e:
                logger.warning("  Batch %d: error loading attention mask: %s", batch_idx, e)

        lm.layers.unregister_hook(hook_id)
        del lm

        self.validation_results["attention_mask_batches"] = batch_info

        logger.info("✅ Attention masks saved and inspected")
        return run_name, {"num_batches": num_batches, "batches": batch_info}

    def validate_inference_alignment(
        self,
        lpm: LPM,
        test_dataset: ClassificationDataset,
        attention_masks_run_id: str,
    ) -> dict[str, Any]:
        """Run inference with detailed validation logging."""
        logger.info("=" * 80)
        logger.info("STEP 4: Validate Inference Alignment")
        logger.info("=" * 80)

        # Load attention masks
        masks_store = LocalStore(base_path=str(self.store_dir))
        lpm.load_inference_attention_masks(masks_store, attention_masks_run_id)
        logger.info("✅ Loaded %d attention mask batches", len(lpm._inference_attention_masks))

        # Load model
        lm = LanguageModel.from_huggingface(MODEL_ID, store=masks_store, device=DEVICE)
        layer_signature = self._get_layer_signature(lm)

        # Instrument LPM to track _seen_samples
        original_process = lpm.process_activations
        seen_samples_log = []

        def instrumented_process(module, input, output):
            """Instrumented version that logs _seen_samples."""
            # Get current state BEFORE processing
            batch_size = output.shape[0]
            batch_idx_before = lpm._seen_samples // batch_size
            seen_before = lpm._seen_samples

            # Log attention mask info if available
            if batch_idx_before in lpm._inference_attention_masks:
                mask_shape = lpm._inference_attention_masks[batch_idx_before].shape
            else:
                mask_shape = "NOT_FOUND"

            log_entry = {
                "batch_idx_computed": batch_idx_before,
                "seen_samples_before": seen_before,
                "batch_size": batch_size,
                "activation_shape": list(output.shape),
                "attention_mask_shape": mask_shape if isinstance(mask_shape, str) else list(mask_shape),
                "has_attention_mask": batch_idx_before in lpm._inference_attention_masks,
            }

            try:
                # Call original
                result = original_process(module, input, output)

                log_entry["seen_samples_after"] = lpm._seen_samples
                log_entry["increment"] = lpm._seen_samples - seen_before
                log_entry["status"] = "success"

            except Exception as e:
                log_entry["seen_samples_after"] = lpm._seen_samples
                log_entry["status"] = "error"
                log_entry["error"] = str(e)

                # Log detailed error info
                if batch_idx_before in lpm._inference_attention_masks:
                    mask = lpm._inference_attention_masks[batch_idx_before]
                    log_entry["attention_mask_analysis"] = {
                        "shape": list(mask.shape),
                        "max_value": int(mask.max().item()),
                        "min_value": int(mask.min().item()),
                        "num_ones": int(mask.sum().item()),
                    }

                logger.error("❌ Error in batch %d: %s", batch_idx_before, e)
                logger.error("   Activation shape: %s", output.shape)
                logger.error("   Attention mask shape: %s", mask_shape)

                raise

            finally:
                seen_samples_log.append(log_entry)

            return result

        lpm.process_activations = instrumented_process

        # Register hook
        hook_id = lm.layers.register_hook(layer_signature, lpm, HookType.FORWARD)

        # Run inference
        test_texts = test_dataset.get_all_texts()
        num_batches = (len(test_texts) + BATCH_SIZE - 1) // BATCH_SIZE

        logger.info("Running inference on %d samples in %d batches...", len(test_texts), num_batches)

        inference_results = {"batches_processed": 0, "error_batch": None}

        try:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(test_texts))
                batch_texts = test_texts[start_idx:end_idx]

                logger.info(
                    "Processing batch %d/%d (samples %d-%d)...", batch_idx + 1, num_batches, start_idx, end_idx - 1
                )

                _ = lm.inference.execute_inference(
                    batch_texts,
                    tok_kwargs={"max_length": MAX_LENGTH},
                    autocast=False,
                    with_controllers=True,
                    stop_after_layer=layer_signature,
                )

                inference_results["batches_processed"] += 1

        except Exception as e:
            inference_results["error_batch"] = batch_idx
            inference_results["error"] = str(e)
            logger.error("❌ Inference failed at batch %d: %s", batch_idx, e)

        finally:
            lm.layers.unregister_hook(hook_id)
            self.validation_results["lpm_seen_samples_log"] = seen_samples_log

        logger.info("✅ Inference validation complete")
        return inference_results

    def compare_sequence_lengths(
        self, test_dataset: ClassificationDataset, attention_masks_run_id: str
    ) -> dict[str, Any]:
        """Compare sequence lengths between dataset and attention masks."""
        logger.info("=" * 80)
        logger.info("STEP 5: Compare Sequence Lengths")
        logger.info("=" * 80)

        masks_store = LocalStore(base_path=str(self.store_dir))
        test_texts = test_dataset.get_all_texts()

        # Load tokenizer
        model_store = LocalStore(base_path=str(self.store_dir))
        lm = LanguageModel.from_huggingface(MODEL_ID, store=model_store, device=DEVICE)
        tokenizer = lm.tokenizer
        del lm

        logger.info("Comparing %d test samples...", len(test_texts))

        # Load saved attention masks
        num_batches = (len(test_texts) + BATCH_SIZE - 1) // BATCH_SIZE
        comparisons = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(test_texts))
            batch_texts = test_texts[start_idx:end_idx]

            # Tokenize this batch independently (same way as during saving)
            tokenized = tokenizer(
                batch_texts, truncation=True, max_length=MAX_LENGTH, return_tensors="pt", padding=True
            )
            expected_mask = tokenized["attention_mask"]

            # Use LocalStore to load attention mask (same way as LPM does)
            try:
                # Load using the same method as load_inference_attention_masks()
                saved_mask_tensor = masks_store.get_detector_metadata_by_layer_by_key(
                    attention_masks_run_id, batch_idx, "attention_masks", "attention_mask"
                )
            except FileNotFoundError:
                logger.warning("Batch %d: attention mask not found", batch_idx)
                continue
            except Exception as e:
                logger.warning("Batch %d: error loading attention mask: %s", batch_idx, e)
                continue

            batch_comparison = {
                "batch_idx": batch_idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "saved_mask_shape": list(saved_mask_tensor.shape),
                "expected_mask_shape": list(expected_mask.shape),
                "shapes_match": saved_mask_tensor.shape == expected_mask.shape,
                "samples": [],
            }

            # Compare individual samples
            num_samples = min(saved_mask_tensor.shape[0], expected_mask.shape[0])
            for i in range(num_samples):
                sample_idx = start_idx + i
                saved_seq_len = int(saved_mask_tensor[i].sum().item())
                expected_seq_len = int(expected_mask[i].sum().item())

                sample_info = {
                    "sample_idx": sample_idx,
                    "saved_seq_len": saved_seq_len,
                    "expected_seq_len": expected_seq_len,
                    "match": saved_seq_len == expected_seq_len,
                    "text_preview": test_texts[sample_idx][:60] + "...",
                }
                batch_comparison["samples"].append(sample_info)

                if saved_seq_len != expected_seq_len:
                    logger.warning(
                        "  Sample %d: seq len mismatch (saved=%d, expected=%d)",
                        sample_idx,
                        saved_seq_len,
                        expected_seq_len,
                    )

            comparisons.append(batch_comparison)

        self.validation_results["sequence_length_comparison"] = comparisons

        logger.info("✅ Sequence length comparison complete")
        return {"num_batches": len(comparisons), "comparisons": comparisons}

    def generate_report(self, output_file: Path) -> None:
        """Generate comprehensive validation report."""
        logger.info("=" * 80)
        logger.info("Generating Report")
        logger.info("=" * 80)

        # Analysis
        report = {
            "experiment_config": {
                "model": MODEL_ID,
                "train_dataset": TRAIN_DATASET,
                "test_dataset": TEST_DATASET,
                "aggregation": AGGREGATION,
                "metric": METRIC,
                "layer": LAYER,
                "device": DEVICE,
                "batch_size": BATCH_SIZE,
                "max_length": MAX_LENGTH,
            },
            "validation_results": self.validation_results,
            "findings": {},
        }

        # Question 1: Do attention mask batch indices match dataset batches?
        attention_batches = self.validation_results["attention_mask_batches"]
        batch_alignment_ok = all(
            b["batch_idx"] == idx and b["expected_samples"] == b["mask_shape"][0]
            for idx, b in enumerate(attention_batches)
        )
        report["findings"]["q1_batch_indices_match"] = batch_alignment_ok

        # Question 2: Is _seen_samples incrementing correctly?
        seen_log = self.validation_results["lpm_seen_samples_log"]
        seen_increments_ok = all(entry.get("increment", 0) == entry.get("batch_size", 0) for entry in seen_log)
        report["findings"]["q2_seen_samples_increments_correctly"] = seen_increments_ok

        # Question 3: Are sequence lengths consistent?
        seq_comparisons = self.validation_results["sequence_length_comparison"]
        seq_lengths_consistent = all(
            all(s["match"] for s in comp["samples"]) for comp in seq_comparisons if "samples" in comp
        )
        report["findings"]["q3_sequence_lengths_consistent"] = seq_lengths_consistent

        # Identify errors
        errors = [entry for entry in seen_log if entry.get("status") == "error"]
        if errors:
            report["findings"]["errors_detected"] = True
            report["findings"]["error_details"] = errors
        else:
            report["findings"]["errors_detected"] = False

        # Save report
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("✅ Report saved: %s", output_file)

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info("Q1: Do attention mask batch indices match dataset batches?")
        logger.info("    → %s", "✅ YES" if batch_alignment_ok else "❌ NO")
        logger.info("Q2: Is _seen_samples incrementing correctly in LPM?")
        logger.info("    → %s", "✅ YES" if seen_increments_ok else "❌ NO")
        logger.info("Q3: Are sequence lengths consistent for the same samples?")
        logger.info("    → %s", "✅ YES" if seq_lengths_consistent else "❌ NO")
        logger.info("")
        logger.info("Errors detected: %s", "❌ YES" if errors else "✅ NO")

        if errors:
            logger.info("\nFirst error details:")
            first_error = errors[0]
            for key, value in first_error.items():
                logger.info("  %s: %s", key, value)

        logger.info("=" * 80)


def main() -> int:
    """Run validation."""
    parser = argparse.ArgumentParser(description="Validate LPM batch alignment")
    parser.add_argument("--store", type=str, default=None, help="Store directory")
    parser.add_argument("--output", type=str, default="validation_report.json", help="Output report file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Setup paths
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent.parent.parent
    store_dir = Path(args.store) if args.store else repo_dir / "store"
    output_file = Path(args.output)

    logger.info("=" * 80)
    logger.info("LPM BATCH ALIGNMENT VALIDATION")
    logger.info("=" * 80)
    logger.info("Store: %s", store_dir)
    logger.info("Output: %s", output_file)
    logger.info("=" * 80)

    validator = BatchAlignmentValidator(store_dir)

    try:
        # Execute validation steps
        train_dataset, test_dataset = validator.load_datasets()
        lpm = validator.train_lpm(train_dataset)
        attention_masks_run_id, _ = validator.save_and_inspect_attention_masks(test_dataset)
        validator.validate_inference_alignment(lpm, test_dataset, attention_masks_run_id)
        validator.compare_sequence_lengths(test_dataset, attention_masks_run_id)
        validator.generate_report(output_file)

        logger.info("\n✅ Validation complete!")
        return 0

    except Exception as e:
        logger.error("❌ Validation failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
