"""
LPM Test Script for Cluster Execution

This script runs a complete LPM pipeline on the cluster:
1. Load pre-saved training activations
2. Train LPM with both Euclidean and Mahalanobis distances
3. Save test attention masks
4. Run inference on test set
5. Evaluate and analyze results
6. Analyze precision matrix (Mahalanobis)

Usage:
    cd /path/to/Mi-Crow
    uv run python -m experiments.scripts.debug.run_lpm_test
"""

from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

REPO_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_DIR))

from experiments.models.lpm.lpm import LPM  # noqa: E402
from mi_crow.datasets import ClassificationDataset  # noqa: E402
from mi_crow.hooks import HookType  # noqa: E402
from mi_crow.hooks.implementations.model_input_detector import ModelInputDetector  # noqa: E402
from mi_crow.language_model.language_model import LanguageModel  # noqa: E402
from mi_crow.store import LocalStore  # noqa: E402
from mi_crow.utils import get_logger, set_seed  # noqa: E402

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "speakleash/Bielik-1.5B-v3.0-Instruct"
LAYER_NUM = 31
AGGREGATION_METHOD = "last_token"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MAX_LENGTH = 512

# Dataset configuration
TRAIN_DATASET_CONFIG = {
    "store_path": "store/datasets/wgmix_train",
    "text_field": "prompt",
    "category_field": "prompt_harm_label",  # Primary label for training/evaluation
}

TEST_DATASET_CONFIG = {
    "store_path": "store/datasets/wgmix_test",
    "text_field": "prompt",
    "category_field": "prompt_harm_label",  # Primary label for training/evaluation
}

# Pre-saved training activations run ID
TRAIN_ACTIVATIONS_RUN_ID = "activations_bielik_1_5b_v3_0_instruct_wgmix_train_layer31_20260117_123045"

# Positive label for binary classification
POSITIVE_LABEL = "harmful"

# Store paths (relative to REPO_DIR)
STORE_DIR = REPO_DIR / "store"
RESULTS_STORE_DIR = STORE_DIR / "lpm_test"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_layer_signature(lm: LanguageModel, layer_num: int) -> str:
    """Get the layer signature for a given layer number."""
    layer_signature = f"llamaforcausallm_model_layers_{layer_num}"
    layer_names = lm.layers.get_layer_names()
    if layer_signature not in layer_names:
        raise ValueError(f"Layer '{layer_signature}' not found in model")
    return layer_signature


# ============================================================================
# STEP 1: Load Datasets
# ============================================================================


def step1_load_datasets() -> tuple[ClassificationDataset, ClassificationDataset]:
    """Load training and test datasets from disk."""
    logger.info("=" * 80)
    logger.info("STEP 1: Load Datasets from Disk")
    logger.info("=" * 80)

    # Load training dataset
    logger.info("Loading training dataset from: %s", TRAIN_DATASET_CONFIG["store_path"])
    train_store = LocalStore(base_path=str(STORE_DIR / TRAIN_DATASET_CONFIG["store_path"].replace("store/", "")))
    train_dataset = ClassificationDataset.from_disk(
        store=train_store,
        text_field=TRAIN_DATASET_CONFIG["text_field"],
        category_field=TRAIN_DATASET_CONFIG["category_field"],
    )
    logger.info("✅ Loaded %d training samples", len(train_dataset))

    # Load test dataset
    logger.info("Loading test dataset from: %s", TEST_DATASET_CONFIG["store_path"])
    test_store = LocalStore(base_path=str(STORE_DIR / TEST_DATASET_CONFIG["store_path"].replace("store/", "")))
    test_dataset = ClassificationDataset.from_disk(
        store=test_store,
        text_field=TEST_DATASET_CONFIG["text_field"],
        category_field=TEST_DATASET_CONFIG["category_field"],
    )
    logger.info("✅ Loaded %d test samples", len(test_dataset))

    # Log class distributions
    from collections import Counter

    logger.info("\nTraining set class distribution:")
    train_labels = [item[TRAIN_DATASET_CONFIG["category_field"]] for item in train_dataset.iter_items()]
    train_dist = Counter(train_labels)
    for label, count in sorted(train_dist.items()):
        logger.info("  %s: %d (%.1f%%)", label, count, 100 * count / len(train_labels))

    logger.info("\nTest set class distribution:")
    test_labels = [item[TEST_DATASET_CONFIG["category_field"]] for item in test_dataset.iter_items()]
    test_dist = Counter(test_labels)
    for label, count in sorted(test_dist.items()):
        logger.info("  %s: %d (%.1f%%)", label, count, 100 * count / len(test_labels))

    return train_dataset, test_dataset


# ============================================================================
# STEP 2: Train LPM with Both Distance Metrics
# ============================================================================


def step2_train_lpm_variants(train_dataset: ClassificationDataset) -> dict[str, LPM]:
    """Train LPM with both Euclidean and Mahalanobis distances using pre-saved activations."""
    logger.info("=" * 80)
    logger.info("STEP 2: Train LPM with Both Distance Metrics")
    logger.info("=" * 80)

    logger.info("Using pre-saved activations: %s", TRAIN_ACTIVATIONS_RUN_ID)

    # Store for reading activations
    activations_store = LocalStore(base_path=str(STORE_DIR))

    # Store for saving models
    results_store = LocalStore(base_path=str(RESULTS_STORE_DIR))
    results_store.base_path.mkdir(parents=True, exist_ok=True)

    lpm_variants = {}

    for distance_metric in ["euclidean", "mahalanobis"]:
        logger.info("\n--- Training LPM with %s distance ---", distance_metric)

        lpm = LPM(
            layer_number=LAYER_NUM,
            distance_metric=distance_metric,
            aggregation_method=AGGREGATION_METHOD,
            device=DEVICE,
            positive_label=POSITIVE_LABEL,
        )

        # Fit on pre-saved activations
        logger.info("Fitting LPM on pre-saved activations...")
        fit_t0 = perf_counter()

        lpm.fit(
            store=activations_store,
            run_id=TRAIN_ACTIVATIONS_RUN_ID,
            dataset=train_dataset,
            model_id=MODEL_ID,
            category_field=TRAIN_DATASET_CONFIG["category_field"],
            max_samples=None,  # Use all samples
        )

        fit_elapsed = perf_counter() - fit_t0
        logger.info("✅ LPM trained with %s distance (%.2fs)", distance_metric, fit_elapsed)
        logger.info("   Prototypes: %s", list(lpm.lpm_context.prototypes.keys()))

        # Save model
        model_path = f"models/lpm_{distance_metric}_layer{LAYER_NUM}_{_timestamp()}.pt"
        logger.info("Saving model to %s...", model_path)
        lpm.save(results_store, model_path)
        logger.info("✅ Model saved")

        lpm_variants[distance_metric] = lpm

    return lpm_variants


# ============================================================================
# STEP 3: Save Test Attention Masks
# ============================================================================


def step3_save_test_attention_masks(test_dataset: ClassificationDataset) -> str:
    """Save attention masks for test dataset."""
    logger.info("=" * 80)
    logger.info("STEP 3: Save Test Attention Masks")
    logger.info("=" * 80)

    # Load language model
    logger.info("Loading language model for attention mask extraction...")
    model_store = LocalStore(base_path=str(STORE_DIR))
    lm = LanguageModel.from_huggingface(MODEL_ID, store=model_store, device=DEVICE)

    # Get layer 0 signature for early stopping (attention masks captured by PRE_FORWARD hook)
    layer_0_signature = _get_layer_signature(lm, 0)
    logger.info("Will stop after layer 0 (attention masks captured by PRE_FORWARD hook)")

    # Generate run name
    ts = _timestamp()
    run_name = f"test_attention_masks_layer{LAYER_NUM}_{ts}"
    logger.info("Run name: %s", run_name)

    # Setup attention mask detector
    attention_mask_layer_sig = "attention_masks"
    if attention_mask_layer_sig not in lm.layers.name_to_layer:
        lm.layers.name_to_layer[attention_mask_layer_sig] = lm.context.model

    attention_mask_detector = ModelInputDetector(
        layer_signature=attention_mask_layer_sig,
        hook_id=f"attention_mask_detector_{run_name}",
        save_input_ids=False,
        save_attention_mask=True,
    )
    attention_mask_hook_id = lm.layers.register_hook(
        attention_mask_layer_sig, attention_mask_detector, HookType.PRE_FORWARD
    )

    # Process batches
    logger.info("Saving attention masks...")
    test_texts = test_dataset.get_all_texts()
    num_batches = (len(test_texts) + BATCH_SIZE - 1) // BATCH_SIZE

    save_t0 = perf_counter()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(test_texts))
        batch_texts = test_texts[start_idx:end_idx]

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx == num_batches - 1:
            logger.info(
                "Processing batch %d/%d (%d samples)...",
                batch_idx + 1,
                num_batches,
                len(batch_texts),
            )

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
            stop_after_layer=layer_0_signature,  # Stop after layer 0 (attention masks captured by PRE_FORWARD)
        )

    save_elapsed = perf_counter() - save_t0

    # Cleanup
    lm.layers.unregister_hook(attention_mask_hook_id)
    logger.info(
        "✅ Attention masks saved: %s (%.2fs, %.2f samples/s)", run_name, save_elapsed, len(test_texts) / save_elapsed
    )

    del lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_name


# ============================================================================
# STEP 4: Run Inference with Both LPM Variants
# ============================================================================


def step4_run_inference(
    lpm_variants: dict[str, LPM], test_dataset: ClassificationDataset, attention_masks_run_id: str
) -> dict[str, str]:
    """Run inference with both LPM variants."""
    logger.info("=" * 80)
    logger.info("STEP 4: Run Inference with Both LPM Variants")
    logger.info("=" * 80)

    # Store for reading attention masks
    masks_store = LocalStore(base_path=str(STORE_DIR))

    # Store for saving predictions
    results_store = LocalStore(base_path=str(RESULTS_STORE_DIR))

    prediction_paths = {}

    # Load language model
    logger.info("Loading language model for inference...")
    lm = LanguageModel.from_huggingface(MODEL_ID, store=masks_store, device=DEVICE)

    test_texts = test_dataset.get_all_texts()

    for distance_metric, lpm in lpm_variants.items():
        logger.info("\n--- Running inference with %s distance ---", distance_metric)

        # Load attention masks for inference
        logger.info("Loading attention masks from run_id=%s...", attention_masks_run_id)
        lpm.load_inference_attention_masks(masks_store, attention_masks_run_id)
        logger.info("✅ Attention masks loaded")

        # Register hook
        layer_signature = lpm.lpm_context.layer_signature
        logger.info("Registering LPM hook on layer: %s", layer_signature)
        hook_id = lm.layers.register_hook(layer_signature, lpm, HookType.FORWARD)

        # Run inference
        logger.info("Running inference on %d test samples...", len(test_texts))
        num_batches = (len(test_texts) + BATCH_SIZE - 1) // BATCH_SIZE

        inference_t0 = perf_counter()

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(test_texts))
            batch_texts = test_texts[start_idx:end_idx]

            # Run inference to trigger LPM detector hook
            _ = lm.inference.execute_inference(
                batch_texts,
                tok_kwargs={"max_length": MAX_LENGTH},
                autocast=False,
                with_controllers=True,
                stop_after_layer=layer_signature,  # Stop after target layer (skip expensive unembedding)
            )

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx == num_batches - 1:
                logger.info(
                    "   Processed %d/%d batches (%d/%d samples)",
                    batch_idx + 1,
                    num_batches,
                    end_idx,
                    len(test_texts),
                )

        inference_elapsed = perf_counter() - inference_t0
        logger.info(
            "✅ Inference complete: %d predictions (%.2fs, %.2f samples/s)",
            len(lpm.predictions),
            inference_elapsed,
            len(test_texts) / inference_elapsed,
        )

        # Unregister hook
        lm.layers.unregister_hook(hook_id)

        # Save predictions
        ts = _timestamp()
        inference_run_id = f"inference_{distance_metric}_{ts}"
        logger.info("Saving predictions to run_id=%s...", inference_run_id)
        pred_path = lpm.save_predictions(run_id=inference_run_id, store=results_store, format="parquet")
        logger.info("✅ Predictions saved: %s", pred_path)

        prediction_paths[distance_metric] = inference_run_id

    # Cleanup
    del lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return prediction_paths


# ============================================================================
# STEP 5: Evaluate and Analyze Results
# ============================================================================


def step5_evaluate_and_analyze(
    lpm_variants: dict[str, LPM], test_dataset: ClassificationDataset, prediction_paths: dict[str, str]
) -> None:
    """Evaluate both LPM variants and save analysis."""
    logger.info("=" * 80)
    logger.info("STEP 5: Evaluate and Analyze Results")
    logger.info("=" * 80)

    results_store = LocalStore(base_path=str(RESULTS_STORE_DIR))

    # Get ground truth labels
    ground_truth = [item[TEST_DATASET_CONFIG["category_field"]] for item in test_dataset.iter_items()]
    logger.info("Ground truth samples: %d", len(ground_truth))

    analysis_results = {}

    for distance_metric, inference_run_id in prediction_paths.items():
        logger.info("\n--- Analyzing %s distance results ---", distance_metric)

        # Load predictions
        pred_path = results_store._run_key(inference_run_id) / "predictions.parquet"
        predictions_df = pd.read_parquet(pred_path)
        logger.info("Loaded %d predictions", len(predictions_df))

        # Extract predicted labels
        predicted_labels = predictions_df["predicted_class"].tolist()

        # Ensure same length
        if len(predicted_labels) != len(ground_truth):
            logger.warning(
                "Mismatch: %d predictions vs %d ground truth",
                len(predicted_labels),
                len(ground_truth),
            )
            min_len = min(len(predicted_labels), len(ground_truth))
            predicted_labels = predicted_labels[:min_len]
            gt = ground_truth[:min_len]
        else:
            gt = ground_truth

        # Calculate metrics
        accuracy = accuracy_score(gt, predicted_labels)
        logger.info("Accuracy: %.4f", accuracy)

        # Classification report
        report = classification_report(gt, predicted_labels, output_dict=True)
        logger.info("\nClassification Report:")
        logger.info("\n%s", classification_report(gt, predicted_labels))

        # Confusion matrix
        cm = confusion_matrix(gt, predicted_labels)
        logger.info("\nConfusion Matrix:")
        logger.info("\n%s", cm)

        # Prepare analysis results
        analysis = {
            "distance_metric": distance_metric,
            "aggregation_method": AGGREGATION_METHOD,
            "layer_number": LAYER_NUM,
            "model_id": MODEL_ID,
            "accuracy": float(accuracy),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "num_samples": len(gt),
            "ground_truth_distribution": pd.Series(gt).value_counts().to_dict(),
            "predicted_distribution": pd.Series(predicted_labels).value_counts().to_dict(),
        }

        analysis_results[distance_metric] = analysis

        # Save analysis to run folder
        analysis_dir = results_store._run_key(inference_run_id) / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_path = analysis_dir / "evaluation_metrics.json"
        with open(json_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info("✅ Analysis saved to: %s", json_path)

        # Save detailed results as CSV
        results_df = pd.DataFrame(
            {
                "text": test_dataset.get_all_texts()[: len(gt)],
                "ground_truth": gt,
                "predicted": predicted_labels,
                "correct": [g == p for g, p in zip(gt, predicted_labels)],
            }
        )
        csv_path = analysis_dir / "detailed_results.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info("✅ Detailed results saved to: %s", csv_path)

    # Compare both variants
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: Euclidean vs Mahalanobis")
    logger.info("=" * 80)

    for metric_name in ["euclidean", "mahalanobis"]:
        if metric_name in analysis_results:
            acc = analysis_results[metric_name]["accuracy"]
            logger.info("%s accuracy: %.4f", metric_name.capitalize(), acc)

    # Save comparison
    comparison_dir = RESULTS_STORE_DIR / "analysis_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = comparison_dir / f"comparison_{_timestamp()}.json"
    with open(comparison_path, "w") as f:
        json.dump(analysis_results, f, indent=2, default=str)
    logger.info("\n✅ Comparison saved to: %s", comparison_path)


# ============================================================================
# STEP 6: Analyze Precision Matrix (Mahalanobis only)
# ============================================================================


def step6_analyze_precision_matrix(lpm_variants: dict[str, LPM]) -> None:
    """Analyze the learned precision matrix for Mahalanobis distance."""
    logger.info("=" * 80)
    logger.info("STEP 6: Analyze Precision Matrix (Mahalanobis only)")
    logger.info("=" * 80)

    if "mahalanobis" not in lpm_variants:
        logger.warning("No Mahalanobis LPM variant found, skipping precision matrix analysis")
        return

    lpm = lpm_variants["mahalanobis"]
    precision_matrix = lpm.lpm_context.precision_matrix

    if precision_matrix is None:
        logger.warning("Precision matrix is None, cannot analyze")
        return

    logger.info("\n--- Precision Matrix Analysis ---")
    logger.info("Shape: %s", precision_matrix.shape)
    logger.info("Device: %s", precision_matrix.device)
    logger.info("Dtype: %s", precision_matrix.dtype)

    # Move to CPU for analysis
    P = precision_matrix.cpu().numpy()
    d = P.shape[0]

    # Basic statistics
    logger.info("\n--- Basic Statistics ---")
    logger.info("Min value: %.6e", P.min())
    logger.info("Max value: %.6e", P.max())
    logger.info("Mean value: %.6e", P.mean())
    logger.info("Std deviation: %.6e", P.std())

    # Diagonal analysis
    logger.info("\n--- Diagonal Analysis ---")
    diag = P.diagonal()
    logger.info("Diagonal min: %.6e", diag.min())
    logger.info("Diagonal max: %.6e", diag.max())
    logger.info("Diagonal mean: %.6e", diag.mean())
    logger.info("Diagonal std: %.6e", diag.std())

    # Check if diagonal dominant
    import numpy as np

    off_diagonal = P.copy()
    np.fill_diagonal(off_diagonal, 0)
    off_diag_abs_max = np.abs(off_diagonal).max()
    diag_abs_min = np.abs(diag).min()
    logger.info("Max absolute off-diagonal: %.6e", off_diag_abs_max)
    logger.info("Min absolute diagonal: %.6e", diag_abs_min)
    if diag_abs_min > off_diag_abs_max:
        logger.info("✅ Matrix is diagonally dominant")
    else:
        logger.info("❌ Matrix is NOT diagonally dominant")

    # Sparsity analysis
    logger.info("\n--- Sparsity Analysis ---")
    zero_threshold = 1e-10
    num_zeros = np.sum(np.abs(P) < zero_threshold)
    total_elements = d * d
    sparsity = (num_zeros / total_elements) * 100
    logger.info(
        "Elements close to zero (|x| < %.0e): %d / %d (%.2f%%)", zero_threshold, num_zeros, total_elements, sparsity
    )

    # Symmetry check
    logger.info("\n--- Symmetry Check ---")
    symmetry_error = np.abs(P - P.T).max()
    logger.info("Max symmetry error |P - P^T|: %.6e", symmetry_error)
    if symmetry_error < 1e-6:
        logger.info("✅ Matrix is symmetric (within tolerance)")
    else:
        logger.info("⚠️ Matrix symmetry error exceeds tolerance")

    # Positive definiteness check (via eigenvalues)
    logger.info("\n--- Positive Definiteness Check ---")
    logger.info("Computing eigenvalues (this may take a moment for large matrices)...")
    try:
        eigenvalues = np.linalg.eigvalsh(P)  # Use eigvalsh for symmetric matrices
        logger.info("Smallest eigenvalue: %.6e", eigenvalues.min())
        logger.info("Largest eigenvalue: %.6e", eigenvalues.max())
        logger.info(
            "Condition number: %.6e", eigenvalues.max() / eigenvalues.min() if eigenvalues.min() > 0 else float("inf")
        )

        num_positive = np.sum(eigenvalues > 0)
        num_negative = np.sum(eigenvalues < 0)
        num_zero = np.sum(np.abs(eigenvalues) < 1e-10)

        logger.info("Positive eigenvalues: %d / %d", num_positive, d)
        logger.info("Negative eigenvalues: %d / %d", num_negative, d)
        logger.info("Near-zero eigenvalues: %d / %d", num_zero, d)

        if num_positive == d:
            logger.info("✅ Matrix is positive definite")
        elif num_positive == d - num_zero and num_negative == 0:
            logger.info("⚠️ Matrix is positive semi-definite")
        else:
            logger.info("❌ Matrix is NOT positive definite")
    except Exception as e:
        logger.warning("Failed to compute eigenvalues: %s", e)

    # Save analysis to file
    analysis_dir = RESULTS_STORE_DIR / "precision_matrix_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Save summary as JSON
    analysis_summary = {
        "aggregation_method": AGGREGATION_METHOD,
        "layer_number": LAYER_NUM,
        "model_id": MODEL_ID,
        "shape": list(P.shape),
        "basic_stats": {
            "min": float(P.min()),
            "max": float(P.max()),
            "mean": float(P.mean()),
            "std": float(P.std()),
        },
        "diagonal": {
            "min": float(diag.min()),
            "max": float(diag.max()),
            "mean": float(diag.mean()),
            "std": float(diag.std()),
        },
        "sparsity": {
            "num_zeros": int(num_zeros),
            "total_elements": int(total_elements),
            "sparsity_percent": float(sparsity),
        },
        "symmetry_error": float(symmetry_error),
    }

    try:
        analysis_summary["eigenvalues"] = {
            "min": float(eigenvalues.min()),
            "max": float(eigenvalues.max()),
            "condition_number": float(eigenvalues.max() / eigenvalues.min()) if eigenvalues.min() > 0 else None,
            "num_positive": int(num_positive),
            "num_negative": int(num_negative),
            "num_zero": int(num_zero),
        }
    except Exception:
        pass

    json_path = analysis_dir / f"precision_matrix_analysis_{_timestamp()}.json"
    with open(json_path, "w") as f:
        json.dump(analysis_summary, f, indent=2)
    logger.info("\n✅ Analysis summary saved to: %s", json_path)

    # Save precision matrix as numpy array for further inspection
    npy_path = analysis_dir / f"precision_matrix_{_timestamp()}.npy"
    np.save(npy_path, P)
    logger.info("✅ Precision matrix saved to: %s", npy_path)

    logger.info("\n" + "=" * 80)


# ============================================================================
# Main Execution
# ============================================================================


def main() -> int:
    """Execute complete LPM pipeline."""
    logger.info("=" * 80)
    logger.info("LPM TEST: Complete Pipeline on Cluster")
    logger.info("=" * 80)
    logger.info("Model: %s", MODEL_ID)
    logger.info("Layer: %d", LAYER_NUM)
    logger.info("Aggregation: %s", AGGREGATION_METHOD)
    logger.info("Device: %s", DEVICE)
    logger.info("Batch size: %d", BATCH_SIZE)
    logger.info("Results directory: %s", RESULTS_STORE_DIR)
    logger.info("=" * 80)

    set_seed(SEED)
    total_t0 = perf_counter()

    try:
        # Step 1: Load datasets
        train_dataset, test_dataset = step1_load_datasets()

        # Step 2: Train LPM variants using pre-saved activations
        lpm_variants = step2_train_lpm_variants(train_dataset)

        # Step 3: Save test attention masks
        test_masks_run_id = step3_save_test_attention_masks(test_dataset)

        # Step 4: Run inference
        prediction_paths = step4_run_inference(lpm_variants, test_dataset, test_masks_run_id)

        # Step 5: Evaluate and analyze
        step5_evaluate_and_analyze(lpm_variants, test_dataset, prediction_paths)

        # Step 6: Analyze precision matrix
        step6_analyze_precision_matrix(lpm_variants)

        total_elapsed = perf_counter() - total_t0
        logger.info("\n" + "=" * 80)
        logger.info("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        logger.info("   Total time: %.2f seconds (%.2f minutes)", total_elapsed, total_elapsed / 60)
        logger.info("   Results saved to: %s", RESULTS_STORE_DIR)
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error("❌ Pipeline failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
