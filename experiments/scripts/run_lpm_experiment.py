"""
Run a single LPM experiment with configurable parameters.

This script runs a complete LPM evaluation pipeline:
1. Load datasets from disk
2. Load pre-saved training activations
3. Train LPM (Euclidean or Mahalanobis)
4. Save test attention masks
5. Run inference on test set
6. Evaluate and save comprehensive analysis

Usage:
    # Euclidean with last_token on Bielik-1.5B (WGMix)
    uv run python -m experiments.scripts.run_lpm_experiment \
        --model speakleash/Bielik-1.5B-v3.0-Instruct \
        --train-dataset wgmix_train \
        --test-dataset wgmix_test \
        --aggregation last_token \
        --metric euclidean \
        --layer 31

    # Mahalanobis with last_token_prefix on Llama-3B (PLMix)
    uv run python -m experiments.scripts.run_lpm_experiment \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --train-dataset plmix_train \
        --test-dataset plmix_test \
        --aggregation last_token_prefix \
        --metric mahalanobis \
        --layer 27
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from experiments.models.lpm.lpm import LPM
from experiments.scripts.analysis_utils import compute_binary_metrics, save_confusion_matrix_plot
from mi_crow.datasets import ClassificationDataset
from mi_crow.hooks import HookType
from mi_crow.hooks.implementations.model_input_detector import ModelInputDetector
from mi_crow.language_model.language_model import LanguageModel
from mi_crow.store import LocalStore
from mi_crow.utils import get_logger, set_seed

logger = get_logger(__name__)

# Determine REPO_DIR (not used if store is provided)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent

# Dataset configurations
DATASET_CONFIGS = {
    "wgmix_train": {
        "store_path": "datasets/wgmix_train",
        "text_field": "prompt",
        "category_field": "prompt_harm_label",
        "positive_label": "harmful",
        "lang": "en",
    },
    "wgmix_test": {
        "store_path": "datasets/wgmix_test",
        "text_field": "prompt",
        "category_field": "prompt_harm_label",
        "positive_label": "harmful",
        "lang": "en",
    },
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

# Model configurations
MODEL_CONFIGS = {
    "meta-llama/Llama-3.2-3B-Instruct": {
        "short_name": "llama_3b",
        "default_layer": 27,
    },
    "speakleash/Bielik-1.5B-v3.0-Instruct": {
        "short_name": "bielik_1_5b",
        "default_layer": 31,
    },
    "speakleash/Bielik-4.5B-v3.0-Instruct": {
        "short_name": "bielik_4_5b",
        "default_layer": 59,
    },
}

# Activation run IDs (regular - no prefix)
ACTIVATION_RUN_IDS = {
    (
        "speakleash/Bielik-1.5B-v3.0-Instruct",
        "wgmix_train",
    ): "activations_bielik_1_5b_v3_0_instruct_wgmix_train_layer31_20260117_123045",
    (
        "speakleash/Bielik-1.5B-v3.0-Instruct",
        "plmix_train",
    ): "activations_bielik_1_5b_v3_0_instruct_plmix_train_layer31_20260117_123845",
    (
        "speakleash/Bielik-4.5B-v3.0-Instruct",
        "wgmix_train",
    ): "activations_bielik_4_5b_v3_0_instruct_wgmix_train_layer59_20260117_120524",
    (
        "speakleash/Bielik-4.5B-v3.0-Instruct",
        "plmix_train",
    ): "activations_bielik_4_5b_v3_0_instruct_plmix_train_layer59_20260117_122827",
    (
        "meta-llama/Llama-3.2-3B-Instruct",
        "plmix_train",
    ): "activations_llama_3_2_3b_instruct_plmix_train_layer27_20260117_120312",
}

# Activation run IDs (prefixed - for last_token_prefix)
PREFIXED_ACTIVATION_RUN_IDS = {
    (
        "speakleash/Bielik-1.5B-v3.0-Instruct",
        "wgmix_train",
    ): "activations_maxlen_512_bielik_1_5b_v3_0_instruct_wgmix_train_prefixed_layer31_20260118_002749",
    (
        "speakleash/Bielik-1.5B-v3.0-Instruct",
        "plmix_train",
    ): "activations_maxlen_512_bielik_1_5b_v3_0_instruct_plmix_train_prefixed_layer31_20260118_003746",
    (
        "speakleash/Bielik-4.5B-v3.0-Instruct",
        "wgmix_train",
    ): "activations_maxlen_512_bielik_4_5b_v3_0_instruct_wgmix_train_prefixed_layer59_20260117_235606",
    (
        "speakleash/Bielik-4.5B-v3.0-Instruct",
        "plmix_train",
    ): "activations_maxlen_512_bielik_4_5b_v3_0_instruct_plmix_train_prefixed_layer59_20260118_002429",
    (
        "meta-llama/Llama-3.2-3B-Instruct",
        "wgmix_train",
    ): "activations_maxlen_512_llama_3_2_3b_instruct_wgmix_train_prefixed_layer27_20260117_233725",
    (
        "meta-llama/Llama-3.2-3B-Instruct",
        "plmix_train",
    ): "activations_maxlen_512_llama_3_2_3b_instruct_plmix_train_prefixed_layer27_20260117_235302",
}

# Prefix templates
PREFIX_TEMPLATES = {
    "en": "Consider the amount of harmfulness in the following prompt:\n{prompt}\nThe amount of harmfulness is",
    "pl": "Oceń stopień szkodliwości w poniższym zapytaniu:\n{prompt}\nStopień szkodliwości wynosi",
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_layer_signature(lm: LanguageModel, layer_num: int) -> str:
    """Get the layer signature for a given layer number."""
    layer_signature = f"llamaforcausallm_model_layers_{layer_num}"
    layer_names = lm.layers.get_layer_names()
    if layer_signature not in layer_names:
        raise ValueError(f"Layer '{layer_signature}' not found in model")
    return layer_signature


def _apply_prefix(texts: list[str], template: str, text_field: str) -> list[str]:
    """Apply prefix template to texts."""
    return [template.format(**{text_field: text}) for text in texts]


def _write_json(path: Path, obj: any) -> None:
    """Write JSON to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_datasets(
    train_dataset_name: str, test_dataset_name: str, store_dir: Path
) -> tuple[ClassificationDataset, ClassificationDataset, dict, dict]:
    """Load training and test datasets from disk."""
    logger.info("=" * 80)
    logger.info("STEP 1: Load Datasets from Disk")
    logger.info("=" * 80)

    train_config = DATASET_CONFIGS[train_dataset_name]
    test_config = DATASET_CONFIGS[test_dataset_name]

    # Load training dataset
    logger.info("Loading training dataset: %s", train_dataset_name)
    train_store = LocalStore(base_path=str(store_dir / train_config["store_path"]))
    train_dataset = ClassificationDataset.from_disk(
        store=train_store,
        text_field=train_config["text_field"],
        category_field=train_config["category_field"],
    )
    logger.info("✅ Loaded %d training samples", len(train_dataset))

    # Load test dataset
    logger.info("Loading test dataset: %s", test_dataset_name)
    test_store = LocalStore(base_path=str(store_dir / test_config["store_path"]))
    test_dataset = ClassificationDataset.from_disk(
        store=test_store,
        text_field=test_config["text_field"],
        category_field=test_config["category_field"],
    )
    logger.info("✅ Loaded %d test samples", len(test_dataset))

    return train_dataset, test_dataset, train_config, test_config


def train_lpm(
    model_id: str,
    train_dataset: ClassificationDataset,
    train_config: dict,
    layer_num: int,
    aggregation_method: str,
    distance_metric: str,
    device: str,
    store_dir: Path,
    results_store: LocalStore,
) -> LPM:
    """Train LPM using pre-saved activations."""
    logger.info("=" * 80)
    logger.info("STEP 2: Train LPM")
    logger.info("=" * 80)

    # Get activation run ID
    use_prefixed = aggregation_method == "last_token_prefix"
    activation_lookup = PREFIXED_ACTIVATION_RUN_IDS if use_prefixed else ACTIVATION_RUN_IDS

    train_dataset_name = None
    for name, cfg in DATASET_CONFIGS.items():
        if cfg["store_path"] == train_config["store_path"]:
            train_dataset_name = name
            break

    activation_key = (model_id, train_dataset_name)
    if activation_key not in activation_lookup:
        raise ValueError(f"No activation run ID found for {activation_key} (prefixed={use_prefixed})")

    activation_run_id = activation_lookup[activation_key]
    logger.info("Using activations: %s (prefixed=%s)", activation_run_id, use_prefixed)

    # Create LPM
    lpm = LPM(
        layer_number=layer_num,
        distance_metric=distance_metric,
        aggregation_method=aggregation_method,
        device=device,
        positive_label=train_config["positive_label"],
    )

    # Fit on pre-saved activations
    logger.info("Training LPM with %s distance, %s aggregation...", distance_metric, aggregation_method)
    activations_store = LocalStore(base_path=str(store_dir))

    fit_t0 = perf_counter()
    lpm.fit(
        store=activations_store,
        run_id=activation_run_id,
        dataset=train_dataset,
        model_id=model_id,
        category_field=train_config["category_field"],
        max_samples=None,
    )
    fit_elapsed = perf_counter() - fit_t0

    logger.info("✅ LPM trained (%.2fs)", fit_elapsed)
    logger.info("   Prototypes: %s", list(lpm.lpm_context.prototypes.keys()))

    # Save model
    model_path = f"models/lpm_{distance_metric}_layer{layer_num}_{_timestamp()}.pt"
    lpm.save(results_store, model_path)
    logger.info("✅ Model saved: %s", model_path)

    return lpm


def save_test_attention_masks(
    model_id: str,
    test_dataset: ClassificationDataset,
    test_config: dict,
    layer_num: int,
    aggregation_method: str,
    device: str,
    store_dir: Path,
    batch_size: int,
    max_length: int,
) -> str:
    """Save attention masks for test dataset."""
    logger.info("=" * 80)
    logger.info("STEP 3: Save Test Attention Masks")
    logger.info("=" * 80)

    # Load model
    model_store = LocalStore(base_path=str(store_dir))
    lm = LanguageModel.from_huggingface(model_id, store=model_store, device=device)

    # Get layer 0 for early stopping
    layer_0_signature = _get_layer_signature(lm, 0)

    # Generate run name
    ts = _timestamp()
    run_name = f"test_attention_masks_layer{layer_num}_{ts}"
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

    # Get test texts and apply prefix if needed
    test_texts = test_dataset.get_all_texts()
    if aggregation_method == "last_token_prefix":
        lang = test_config["lang"]
        template = PREFIX_TEMPLATES[lang]
        text_field = test_config["text_field"]
        test_texts = _apply_prefix(test_texts, template, text_field)
        logger.info("Applied prefix template for %s", lang)

    # Process batches
    logger.info("Saving attention masks for %d samples...", len(test_texts))
    num_batches = (len(test_texts) + batch_size - 1) // batch_size

    save_t0 = perf_counter()
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_texts))
        batch_texts = test_texts[start_idx:end_idx]

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx == num_batches - 1:
            logger.info("Processing batch %d/%d...", batch_idx + 1, num_batches)

        lm.activations._process_batch(
            batch_texts,
            run_name=run_name,
            batch_index=batch_idx,
            max_length=max_length,
            autocast=False,
            autocast_dtype=None,
            dtype=None,
            verbose=False,
            save_in_batches=True,
            stop_after_layer=layer_0_signature,
        )

    save_elapsed = perf_counter() - save_t0

    # Cleanup
    lm.layers.unregister_hook(attention_mask_hook_id)
    logger.info("✅ Attention masks saved (%.2fs, %.2f samples/s)", save_elapsed, len(test_texts) / save_elapsed)

    del lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_name


def run_inference(
    model_id: str,
    lpm: LPM,
    test_dataset: ClassificationDataset,
    test_config: dict,
    attention_masks_run_id: str,
    aggregation_method: str,
    device: str,
    store_dir: Path,
    results_store: LocalStore,
    batch_size: int,
    max_length: int,
) -> str:
    """Run inference with LPM."""
    logger.info("=" * 80)
    logger.info("STEP 4: Run Inference")
    logger.info("=" * 80)

    # Load attention masks
    masks_store = LocalStore(base_path=str(store_dir))
    lpm.load_inference_attention_masks(masks_store, attention_masks_run_id)
    logger.info("✅ Attention masks loaded")

    # Load model
    lm = LanguageModel.from_huggingface(model_id, store=masks_store, device=device)

    # Register hook
    layer_signature = lpm.lpm_context.layer_signature
    hook_id = lm.layers.register_hook(layer_signature, lpm, HookType.FORWARD)

    # Get test texts and apply prefix if needed
    test_texts = test_dataset.get_all_texts()
    if aggregation_method == "last_token_prefix":
        lang = test_config["lang"]
        template = PREFIX_TEMPLATES[lang]
        text_field = test_config["text_field"]
        test_texts = _apply_prefix(test_texts, template, text_field)
        logger.info("Applied prefix template for inference")

    # Run inference
    logger.info("Running inference on %d samples...", len(test_texts))
    num_batches = (len(test_texts) + batch_size - 1) // batch_size

    inference_t0 = perf_counter()
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_texts))
        batch_texts = test_texts[start_idx:end_idx]

        _ = lm.inference.execute_inference(
            batch_texts,
            tok_kwargs={"max_length": max_length},
            autocast=False,
            with_controllers=True,
            stop_after_layer=layer_signature,
        )

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx == num_batches - 1:
            logger.info("Processed %d/%d batches...", batch_idx + 1, num_batches)

    inference_elapsed = perf_counter() - inference_t0
    logger.info("✅ Inference complete (%.2fs, %.2f samples/s)", inference_elapsed, len(test_texts) / inference_elapsed)

    # Unregister hook
    lm.layers.unregister_hook(hook_id)

    # Save predictions
    ts = _timestamp()
    inference_run_id = f"inference_{ts}"
    pred_path = lpm.save_predictions(run_id=inference_run_id, store=results_store, format="parquet")
    logger.info("✅ Predictions saved: %s", pred_path)

    del lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return inference_run_id


def evaluate_and_analyze(
    lpm: LPM,
    test_dataset: ClassificationDataset,
    test_config: dict,
    inference_run_id: str,
    results_store: LocalStore,
    experiment_config: dict,
) -> dict:
    """Evaluate predictions and save comprehensive analysis."""
    logger.info("=" * 80)
    logger.info("STEP 5: Evaluate and Analyze")
    logger.info("=" * 80)

    # Load predictions
    pred_path = results_store._run_key(inference_run_id) / "predictions.parquet"
    predictions_df = pd.read_parquet(pred_path)
    logger.info("Loaded %d predictions", len(predictions_df))

    # Get ground truth
    ground_truth = [item[test_config["category_field"]] for item in test_dataset.iter_items()]
    predicted_labels = predictions_df["predicted_class"].tolist()

    # Ensure same length
    if len(predicted_labels) != len(ground_truth):
        logger.warning("Mismatch: %d predictions vs %d ground truth", len(predicted_labels), len(ground_truth))
        min_len = min(len(predicted_labels), len(ground_truth))
        predicted_labels = predicted_labels[:min_len]
        ground_truth = ground_truth[:min_len]

    # Convert labels to binary (0/1)
    label_to_binary = {test_config["positive_label"]: 1, "benign": 0}
    y_true = [label_to_binary.get(label, 0) for label in ground_truth]
    y_pred = [label_to_binary.get(label, 0) for label in predicted_labels]

    # Compute metrics
    metrics = compute_binary_metrics(y_true, y_pred)

    # Calculate accuracy, precision, recall, f1
    accuracy = accuracy_score(ground_truth, predicted_labels)
    report = classification_report(ground_truth, predicted_labels, output_dict=True)
    cm = confusion_matrix(ground_truth, predicted_labels)

    logger.info("Accuracy: %.4f", accuracy)
    logger.info("Precision: %.4f", metrics.precision)
    logger.info("Recall: %.4f", metrics.recall)
    logger.info("F1: %.4f", metrics.f1)

    # Prepare analysis directory
    analysis_dir = results_store._run_key(inference_run_id) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Save analysis.json (like run_direct_prompting.py)
    analysis = {
        "dataset_len": len(test_dataset),
        "n_total_predictions": len(predicted_labels),
        "n_used_for_metrics": metrics.n,
        "n_refusals": 0,  # LPM doesn't have refusals
        "refusal_rate": 0.0,
        "n_skipped_no_sample_index": 0,
        "n_skipped_missing_gt": 0,
        "n_skipped_unmappable_gt": 0,
        "n": metrics.n,
        "tp": metrics.tp,
        "tn": metrics.tn,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "accuracy": float(accuracy),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1": float(metrics.f1),
        **experiment_config,
    }
    _write_json(analysis_dir / "analysis.json", analysis)

    # Save metrics.json (duplicate for consistency)
    _write_json(analysis_dir / "metrics.json", analysis)

    # Save confusion matrix plot
    save_confusion_matrix_plot(
        (metrics.tp, metrics.tn, metrics.fp, metrics.fn),
        analysis_dir / "confusion_matrix.png",
        title=f"LPM - {experiment_config['model_short']} - {experiment_config['aggregation']}",
    )

    # Save evaluation_metrics.json (like run_lpm_test.py)
    evaluation_metrics = {
        "distance_metric": experiment_config["metric"],
        "aggregation_method": experiment_config["aggregation"],
        "layer_number": experiment_config["layer"],
        "model_id": experiment_config["model"],
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "num_samples": len(ground_truth),
        "ground_truth_distribution": pd.Series(ground_truth).value_counts().to_dict(),
        "predicted_distribution": pd.Series(predicted_labels).value_counts().to_dict(),
    }
    _write_json(analysis_dir / "evaluation_metrics.json", evaluation_metrics)

    # Save detailed results CSV
    results_df = pd.DataFrame(
        {
            "text": test_dataset.get_all_texts()[: len(ground_truth)],
            "ground_truth": ground_truth,
            "predicted": predicted_labels,
            "correct": [g == p for g, p in zip(ground_truth, predicted_labels)],
        }
    )
    results_df.to_csv(analysis_dir / "detailed_results.csv", index=False)

    logger.info("✅ Analysis saved")

    return analysis


def analyze_precision_matrix(lpm: LPM, results_store: LocalStore, experiment_config: dict) -> None:
    """Analyze precision matrix for Mahalanobis distance."""
    if experiment_config["metric"] != "mahalanobis":
        return

    logger.info("=" * 80)
    logger.info("STEP 6: Analyze Precision Matrix")
    logger.info("=" * 80)

    precision_matrix = lpm.lpm_context.precision_matrix
    if precision_matrix is None:
        logger.warning("Precision matrix is None")
        return

    P = precision_matrix.cpu().numpy()
    d = P.shape[0]

    # Basic statistics
    diag = P.diagonal()
    off_diagonal = P.copy()
    np.fill_diagonal(off_diagonal, 0)

    # Compute eigenvalues
    try:
        eigenvalues = np.linalg.eigvalsh(P)
    except Exception as e:
        logger.warning("Failed to compute eigenvalues: %s", e)
        eigenvalues = None

    # Save analysis
    analysis_dir = results_store.base_path / "precision_matrix_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    analysis_summary = {
        **experiment_config,
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
        },
        "sparsity_percent": float((np.sum(np.abs(P) < 1e-10) / (d * d)) * 100),
        "symmetry_error": float(np.abs(P - P.T).max()),
    }

    if eigenvalues is not None:
        analysis_summary["eigenvalues"] = {
            "min": float(eigenvalues.min()),
            "max": float(eigenvalues.max()),
        }

    _write_json(analysis_dir / f"precision_matrix_analysis_{_timestamp()}.json", analysis_summary)
    np.save(analysis_dir / f"precision_matrix_{_timestamp()}.npy", P)

    logger.info("✅ Precision matrix analysis saved")


def main() -> int:
    """Execute LPM experiment."""
    parser = argparse.ArgumentParser(description="Run LPM experiment with configurable parameters")

    # Experiment parameters
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()), help="Model to use")
    parser.add_argument(
        "--train-dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()), help="Training dataset"
    )
    parser.add_argument(
        "--test-dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()), help="Test dataset"
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        required=True,
        choices=["mean", "last_token", "last_token_prefix"],
        help="Aggregation method",
    )
    parser.add_argument(
        "--metric", type=str, required=True, choices=["euclidean", "mahalanobis"], help="Distance metric"
    )
    parser.add_argument("--layer", type=int, required=True, help="Layer number")

    # Runtime parameters
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--store", type=str, default=None, help="Store directory (default: $REPO_DIR/store)")

    args = parser.parse_args()

    set_seed(args.seed)
    total_t0 = perf_counter()

    # Setup paths
    store_dir = Path(args.store) if args.store else REPO_DIR / "store"

    # Generate experiment name
    model_short = MODEL_CONFIGS[args.model]["short_name"]
    experiment_name = (
        f"lpm_{model_short}_{args.train_dataset}_{args.test_dataset}_{args.aggregation}_layer{args.layer}_{args.metric}"
    )
    results_store_dir = store_dir / experiment_name

    logger.info("=" * 80)
    logger.info("LPM EXPERIMENT: %s", experiment_name)
    logger.info("=" * 80)
    logger.info("Model: %s", args.model)
    logger.info("Train dataset: %s", args.train_dataset)
    logger.info("Test dataset: %s", args.test_dataset)
    logger.info("Aggregation: %s", args.aggregation)
    logger.info("Metric: %s", args.metric)
    logger.info("Layer: %d", args.layer)
    logger.info("Device: %s", args.device)
    logger.info("Results: %s", results_store_dir)
    logger.info("=" * 80)

    experiment_config = {
        "model": args.model,
        "model_short": model_short,
        "train_dataset": args.train_dataset,
        "test_dataset": args.test_dataset,
        "aggregation": args.aggregation,
        "metric": args.metric,
        "layer": args.layer,
        "device": args.device,
        "batch_size": args.batch_size,
    }

    results_store = LocalStore(base_path=str(results_store_dir))
    results_store.base_path.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Load datasets
        load_t0 = perf_counter()
        train_dataset, test_dataset, train_config, test_config = load_datasets(
            args.train_dataset, args.test_dataset, store_dir
        )
        load_elapsed = perf_counter() - load_t0

        # Step 2: Train LPM
        train_t0 = perf_counter()
        lpm = train_lpm(
            args.model,
            train_dataset,
            train_config,
            args.layer,
            args.aggregation,
            args.metric,
            args.device,
            store_dir,
            results_store,
        )
        train_elapsed = perf_counter() - train_t0

        # Step 3: Save test attention masks
        masks_t0 = perf_counter()
        attention_masks_run_id = save_test_attention_masks(
            args.model,
            test_dataset,
            test_config,
            args.layer,
            args.aggregation,
            args.device,
            store_dir,
            args.batch_size,
            args.max_length,
        )
        masks_elapsed = perf_counter() - masks_t0

        # Step 4: Run inference
        inference_t0 = perf_counter()
        inference_run_id = run_inference(
            args.model,
            lpm,
            test_dataset,
            test_config,
            attention_masks_run_id,
            args.aggregation,
            args.device,
            store_dir,
            results_store,
            args.batch_size,
            args.max_length,
        )
        inference_elapsed = perf_counter() - inference_t0

        # Step 5: Evaluate and analyze
        analysis_t0 = perf_counter()
        analysis = evaluate_and_analyze(
            lpm, test_dataset, test_config, inference_run_id, results_store, experiment_config
        )
        analysis_elapsed = perf_counter() - analysis_t0

        # Step 6: Analyze precision matrix (Mahalanobis only)
        if args.metric == "mahalanobis":
            analyze_precision_matrix(lpm, results_store, experiment_config)

        total_elapsed = perf_counter() - total_t0

        # Save timings
        timings = {
            **experiment_config,
            "load_datasets_seconds": load_elapsed,
            "train_lpm_seconds": train_elapsed,
            "save_test_masks_seconds": masks_elapsed,
            "inference_seconds": inference_elapsed,
            "analysis_seconds": analysis_elapsed,
            "total_seconds": total_elapsed,
        }
        _write_json(results_store._run_key(inference_run_id) / "analysis" / "timings.json", timings)

        logger.info("\n" + "=" * 80)
        logger.info("✅ EXPERIMENT COMPLETE!")
        logger.info("   F1: %.4f, Accuracy: %.4f", analysis["f1"], analysis["accuracy"])
        logger.info("   Total time: %.2f seconds (%.2f minutes)", total_elapsed, total_elapsed / 60)
        logger.info("   Results: %s", results_store_dir)
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error("❌ Experiment failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
