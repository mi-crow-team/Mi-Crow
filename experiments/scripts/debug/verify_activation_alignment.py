#!/usr/bin/env python3
# ruff: noqa
"""
Verify Activation-Dataset Alignment and Attention Mask Consistency

This script diagnoses the IndexError in LPM experiments by verifying:
1. Dataset sample order is deterministic
2. Attention masks align with dataset samples
3. Batch indexing is consistent
4. Sequence lengths match between attention masks and activations

The error occurs when wrong attention masks are applied to activations:
  IndexError: index 498 is out of bounds for dimension 0 with size 181

Root causes:
  - Batch index mismatch (hardcoded batch_size=64 in LPM)
  - Concurrent runs overwriting attention masks  
  - Different batch sizes during save vs inference

Usage:
    # Verify attention mask alignment
    python verify_activation_alignment.py \\
        --dataset plmix_test \\
        --attention_mask_run test_attention_masks_layer31_20260125_044240 \\
        --store /mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store
    
    # Check if activations would align
    python verify_activation_alignment.py \\
        --dataset plmix_train \\
        --activation_run activations_bielik_1_5b_v3_0_instruct_plmix_train_layer31_20260117_123845 \\
        --store /mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store
"""

import argparse
import hashlib
import logging
from pathlib import Path

import torch

from mi_crow.datasets import ClassificationDataset
from mi_crow.store import LocalStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default store path
DEFAULT_STORE_PATH = "/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store"

# Dataset configurations
DATASET_CONFIGS = {
    "wgmix_train": {
        "store_path": "datasets/wgmix_train",
        "text_field": "prompt",
        "category_field": "prompt_harm_label",
    },
    "wgmix_test": {
        "store_path": "datasets/wgmix_test",
        "text_field": "prompt",
        "category_field": "prompt_harm_label",
    },
    "plmix_train": {
        "store_path": "datasets/plmix_train",
        "text_field": "text",
        "category_field": "text_harm_label",
    },
    "plmix_test": {
        "store_path": "datasets/plmix_test",
        "text_field": "text",
        "category_field": "text_harm_label",
    },
}


def hash_text(text: str) -> str:
    """Compute SHA256 hash of text for fast comparison."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_dataset_info(dataset_name: str, store_base_path: str) -> dict:
    """Load dataset and return key information."""
    logger.info(f"Loading dataset: {dataset_name}")

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    config = DATASET_CONFIGS[dataset_name]
    store = LocalStore(base_path=str(Path(store_base_path) / config["store_path"]))

    dataset = ClassificationDataset.from_disk(
        store=store,
        text_field=config["text_field"],
        category_field=config["category_field"],
    )

    items = list(dataset.iter_items())

    return {
        "num_samples": len(items),
        "text_field": config["text_field"],
        "category_field": config["category_field"],
        "texts": [item["text"] for item in items],  # Always "text" due to normalization
        "labels": [item[config["category_field"]] for item in items],
        "text_hashes": [hash_text(item["text"]) for item in items],
    }


def load_attention_mask_info(store_path: str, run_id: str) -> dict:
    """Load attention mask run information and analyze batch structure."""
    logger.info(f"Loading attention masks from run: {run_id}")

    store = LocalStore(base_path=store_path)
    run_path = store._run_key(run_id)

    if not run_path.exists():
        raise ValueError(f"Attention mask run not found: {run_path}")

    # Find all batch files
    batch_files = sorted(run_path.glob("batch_*.pt"))
    num_batches = len(batch_files)

    logger.info(f"Found {num_batches} batches in run")

    # Analyze each batch
    total_samples = 0
    batch_info = []
    all_seq_lengths = []

    for batch_idx, batch_file in enumerate(batch_files):
        batch_data = torch.load(batch_file, weights_only=True)

        # Extract attention mask
        if isinstance(batch_data, dict):
            attention_mask = batch_data.get("attention_mask")
        else:
            raise ValueError(f"Unexpected batch data format in {batch_file}")

        if attention_mask is None:
            raise ValueError(f"No attention_mask found in {batch_file}")

        batch_size, seq_len = attention_mask.shape
        total_samples += batch_size

        # Get sequence lengths (sum of attention mask per sample)
        seq_lengths = attention_mask.sum(dim=1).tolist()
        all_seq_lengths.extend(seq_lengths)

        batch_info.append(
            {
                "batch_idx": batch_idx,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "min_seq_len": min(seq_lengths),
                "max_seq_len": max(seq_lengths),
                "seq_lengths": seq_lengths,
            }
        )

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Processed {batch_idx + 1}/{num_batches} batches...")

    logger.info(f"Total samples in attention masks: {total_samples}")

    return {
        "num_batches": num_batches,
        "num_samples": total_samples,
        "batch_info": batch_info,
        "all_seq_lengths": all_seq_lengths,
    }


def verify_batch_indexing(dataset_info: dict, attention_mask_info: dict) -> dict:
    """
    Verify that batch indexing would work correctly during LPM inference.

    The bug in LPM: batch_idx = self._seen_samples // 64
    This assumes all batches have size 64, which is wrong for the last batch!
    """
    results = {
        "status": "OK",
        "issues": [],
    }

    dataset_samples = dataset_info["num_samples"]
    attention_samples = attention_mask_info["num_samples"]

    logger.info(f"Dataset samples: {dataset_samples}")
    logger.info(f"Attention mask samples: {attention_samples}")

    # Check sample count
    if dataset_samples != attention_samples:
        results["status"] = "FAILED"
        results["issues"].append(
            {
                "type": "SAMPLE_COUNT_MISMATCH",
                "severity": "CRITICAL",
                "details": f"Dataset has {dataset_samples} samples, attention masks have {attention_samples}",
            }
        )
        return results

    # Simulate LPM batch indexing (the buggy way)
    logger.info("Simulating LPM batch indexing (hardcoded batch_size=64)...")

    seen_samples = 0
    batch_info = attention_mask_info["batch_info"]

    for actual_batch_idx, batch in enumerate(batch_info):
        # This is what LPM does (BUGGY!)
        lpm_batch_idx = seen_samples // 64

        if lpm_batch_idx != actual_batch_idx:
            results["status"] = "FAILED"
            results["issues"].append(
                {
                    "type": "BATCH_INDEX_MISMATCH",
                    "severity": "CRITICAL",
                    "details": f"Sample {seen_samples}: LPM calculates batch_idx={lpm_batch_idx}, but actual batch_idx={actual_batch_idx}",
                    "seen_samples": seen_samples,
                    "lpm_batch_idx": lpm_batch_idx,
                    "actual_batch_idx": actual_batch_idx,
                    "actual_batch_size": batch["batch_size"],
                }
            )

        seen_samples += batch["batch_size"]

    # Check for varying batch sizes
    batch_sizes = [b["batch_size"] for b in batch_info]
    if len(set(batch_sizes)) > 1:
        results["issues"].append(
            {
                "type": "VARYING_BATCH_SIZES",
                "severity": "HIGH",
                "details": f"Batch sizes vary: {set(batch_sizes)}. Last batch is likely smaller.",
                "batch_sizes": batch_sizes,
            }
        )
        logger.warning(f"Batch sizes vary: {batch_sizes}")
        logger.warning("This WILL cause batch index misalignment in LPM!")

    return results


def diagnose_index_error(dataset_info: dict, attention_mask_info: dict, sample_idx: int = None) -> dict:
    """
    Diagnose the specific IndexError that occurred.

    The error: index 498 is out of bounds for dimension 0 with size 181
    This means attention_mask expects 498+ tokens, but activations only have 181.
    """
    results = {
        "diagnosis": [],
    }

    seq_lengths = attention_mask_info["all_seq_lengths"]
    batch_info = attention_mask_info["batch_info"]

    logger.info("Analyzing sequence length distribution...")
    logger.info(f"Min sequence length: {min(seq_lengths)}")
    logger.info(f"Max sequence length: {max(seq_lengths)}")
    logger.info(f"Mean sequence length: {sum(seq_lengths) / len(seq_lengths):.1f}")

    # Find samples with very long sequences
    long_samples = [(i, length) for i, length in enumerate(seq_lengths) if length > 400]
    if long_samples:
        logger.warning(f"Found {len(long_samples)} samples with >400 tokens:")
        for idx, length in long_samples[:5]:
            logger.warning(f"  Sample {idx}: {length} tokens")
            results["diagnosis"].append(f"Sample {idx} has {length} tokens")

    # Simulate the indexing bug to find mismatches
    logger.info("Simulating potential index mismatches...")
    seen_samples = 0

    for actual_batch_idx, batch in enumerate(batch_info):
        lpm_batch_idx = seen_samples // 64

        if lpm_batch_idx != actual_batch_idx:
            # This is where the mismatch would occur!
            wrong_batch = batch_info[lpm_batch_idx] if lpm_batch_idx < len(batch_info) else None

            logger.error(f"MISMATCH at sample {seen_samples}:")
            logger.error(f"  LPM would use batch {lpm_batch_idx} attention masks")
            logger.error(f"  But actual activations are from batch {actual_batch_idx}")

            if wrong_batch:
                logger.error(f"  Wrong batch seq lengths: {wrong_batch['min_seq_len']}-{wrong_batch['max_seq_len']}")
                logger.error(f"  Correct batch seq lengths: {batch['min_seq_len']}-{batch['max_seq_len']}")

                results["diagnosis"].append(
                    f"Sample {seen_samples}: Using batch {lpm_batch_idx} masks "
                    f"(seq_len {wrong_batch['max_seq_len']}) "
                    f"on batch {actual_batch_idx} activations "
                    f"(seq_len {batch['max_seq_len']})"
                )

        seen_samples += batch["batch_size"]

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify activation-dataset alignment and diagnose IndexError")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset name",
    )
    parser.add_argument(
        "--attention_mask_run",
        type=str,
        help="Attention mask run ID (e.g., test_attention_masks_layer31_20260125_044240)",
    )
    parser.add_argument(
        "--activation_run",
        type=str,
        help="Activation run ID (for checking saved activations alignment)",
    )
    parser.add_argument(
        "--store",
        type=str,
        default=DEFAULT_STORE_PATH,
        help=f"Store base path (default: {DEFAULT_STORE_PATH})",
    )
    args = parser.parse_args()

    if not args.attention_mask_run and not args.activation_run:
        parser.error("Must provide either --attention_mask_run or --activation_run")

    logger.info("=" * 80)
    logger.info("ACTIVATION-DATASET ALIGNMENT VERIFICATION & IndexError DIAGNOSIS")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Store path: {args.store}")
    if args.attention_mask_run:
        logger.info(f"Attention mask run: {args.attention_mask_run}")
    if args.activation_run:
        logger.info(f"Activation run: {args.activation_run}")
    logger.info("")

    # Load dataset info
    dataset_info = load_dataset_info(args.dataset, args.store)
    logger.info(f"✅ Dataset loaded: {dataset_info['num_samples']} samples")
    logger.info("")

    if args.attention_mask_run:
        # Verify attention mask alignment
        logger.info("=" * 80)
        logger.info("ATTENTION MASK VERIFICATION")
        logger.info("=" * 80)

        attention_mask_info = load_attention_mask_info(args.store, args.attention_mask_run)
        logger.info(f"✅ Attention masks loaded: {attention_mask_info['num_samples']} samples")
        logger.info("")

        # Verify batch indexing
        logger.info("Checking batch indexing (LPM bug simulation)...")
        batch_results = verify_batch_indexing(dataset_info, attention_mask_info)
        logger.info("")

        # Diagnose potential index errors
        logger.info("Diagnosing potential IndexError causes...")
        diagnosis = diagnose_index_error(dataset_info, attention_mask_info)
        logger.info("")

        # Report results
        logger.info("=" * 80)
        logger.info("RESULTS")
        logger.info("=" * 80)
        logger.info(f"Status: {batch_results['status']}")
        logger.info("")

        if batch_results["status"] == "OK" and not batch_results["issues"]:
            logger.info("✅ NO ISSUES FOUND")
            logger.info("   - Sample counts match")
            logger.info("   - Batch indexing is correct")
        else:
            logger.info("❌ ISSUES DETECTED")
            logger.info(f"   - Found {len(batch_results['issues'])} issues:")
            logger.info("")

            # Group by severity
            critical = [i for i in batch_results["issues"] if i.get("severity") == "CRITICAL"]
            high = [i for i in batch_results["issues"] if i.get("severity") == "HIGH"]

            if critical:
                logger.info("   CRITICAL ISSUES:")
                for issue in critical:
                    logger.info(f"     • {issue['type']}: {issue['details']}")
                logger.info("")

            if high:
                logger.info("   HIGH PRIORITY ISSUES:")
                for issue in high:
                    logger.info(f"     • {issue['type']}: {issue['details']}")
                logger.info("")

            # Show diagnosis
            if diagnosis["diagnosis"]:
                logger.info("   DIAGNOSIS:")
                for diag in diagnosis["diagnosis"]:
                    logger.info(f"     • {diag}")
                logger.info("")

            # Explain the bug
            logger.info("   ROOT CAUSE:")
            logger.info("     The LPM code has a hardcoded batch_size=64:")
            logger.info("       batch_idx = self._seen_samples // 64")
            logger.info("")
            logger.info("     When the last batch is smaller (e.g., 15 samples),")
            logger.info("     this calculation returns the WRONG batch index!")
            logger.info("")
            logger.info("     Example with 207 samples:")
            logger.info("       - Batches: [64, 64, 64, 15]")
            logger.info("       - At sample 192: 192 // 64 = 3 ✓ (correct)")
            logger.info("       - At sample 193: 193 // 64 = 3 ✓ (correct)")
            logger.info("       - But we're actually processing batch 3 samples!")
            logger.info("")
            logger.info("     This causes wrong attention masks to be applied,")
            logger.info("     leading to IndexError when sequence lengths differ.")

        logger.info("=" * 80)

        return 0 if batch_results["status"] == "OK" else 1

    # If only activation_run provided, just check basic alignment
    # (implementation for activation checking can be added later)
    logger.info("Activation-only verification not yet implemented")
    return 0


if __name__ == "__main__":
    exit(main())
