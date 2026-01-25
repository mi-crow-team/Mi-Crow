"""
Run direct prompting experiments on test datasets.

This script evaluates direct prompting baselines with multiple prompt templates
on cached test datasets (WGMix or PLMix). Datasets must be prepared first using
experiments.scripts.prepare_datasets.

The script runs all 4 prompt templates sequentially and saves results for each
in separate run directories.

Usage:
    # WildGuardMix Test with Llama (English)
    uv run python -m experiments.scripts.run_direct_prompting \
        --dataset-name wgmix_test \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --device cuda

    # PL Mix Test with Bielik (Polish)
    uv run python -m experiments.scripts.run_direct_prompting \
        --dataset-name plmix_test \
        --model speakleash/Bielik-4.5B-v3.0-Instruct \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

from experiments.baselines import create_direct_prompting_predictor
from experiments.scripts.analysis_utils import (
    compute_binary_metrics,
    map_wildguard_label_to_binary,
    save_confusion_matrix_plot,
)
from experiments.scripts.config import DATASET_CONFIGS, HARMFULNESS_DETECTION_PROMPTS
from mi_crow.datasets import ClassificationDataset
from mi_crow.store import LocalStore
from mi_crow.utils import get_logger, set_seed

logger = get_logger(__name__)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_timings(run_dir: Path, timings: Dict[str, Any]) -> None:
    _write_json(run_dir / "analysis" / "timings.json", timings)


def _evaluate_and_save_analysis(
    *,
    run_dir: Path,
    dataset: ClassificationDataset,
    category_field: str,
    predictions: List[Dict[str, Any]],
    model_name: str,
    prompt_name: str,
) -> Dict[str, Any]:
    """Evaluate predictions and save analysis artifacts.

    Returns:
        Dictionary with analysis metrics
    """
    # Align by sample_index
    y_true: List[int] = []
    y_pred: List[int] = []

    n_total_predictions = 0
    n_skipped_no_sample_index = 0
    n_skipped_missing_gt = 0
    n_skipped_unmappable_gt = 0
    n_skipped_missing_pred = 0  # Refusals

    for p in predictions:
        n_total_predictions += 1
        idx = p.get("sample_index")
        if idx is None:
            n_skipped_no_sample_index += 1
            continue

        gt_raw = dataset[int(idx)].get(category_field)
        if gt_raw is None:
            n_skipped_missing_gt += 1
            continue

        gt = map_wildguard_label_to_binary(gt_raw)
        if gt is None:
            n_skipped_unmappable_gt += 1
            continue

        pred_label = p.get("predicted_label")
        if pred_label is None:
            n_skipped_missing_pred += 1
            continue

        y_true.append(gt)
        y_pred.append(int(pred_label))

    metrics = compute_binary_metrics(y_true, y_pred)
    analysis = {
        "prompt_name": prompt_name,
        "dataset_len": len(dataset),
        "n_total_predictions": n_total_predictions,
        "n_used_for_metrics": metrics.n,
        "n_refusals": n_skipped_missing_pred,
        "refusal_rate": n_skipped_missing_pred / n_total_predictions if n_total_predictions > 0 else 0.0,
        "n_skipped_no_sample_index": n_skipped_no_sample_index,
        "n_skipped_missing_gt": n_skipped_missing_gt,
        "n_skipped_unmappable_gt": n_skipped_unmappable_gt,
        "n": metrics.n,
        "tp": metrics.tp,
        "tn": metrics.tn,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
    }

    out_analysis_dir = run_dir / "analysis"
    _write_json(out_analysis_dir / "metrics.json", analysis)
    _write_json(out_analysis_dir / "analysis.json", analysis)

    save_confusion_matrix_plot(
        (metrics.tp, metrics.tn, metrics.fp, metrics.fn),
        run_dir / "analysis" / "confusion_matrix.png",
        title=f"{model_name} - {prompt_name}",
    )

    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Run direct prompting experiments on test datasets")

    # Dataset selection
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Test dataset to use (must be prepared first with prepare_datasets.py)",
    )
    parser.add_argument("--store", type=str, default="store", help="LocalStore base path for saving results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model identifier (e.g., meta-llama/Llama-3.2-3B-Instruct)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device for inference"
    )
    parser.add_argument("--max-new-tokens", type=int, default=10, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)")

    args = parser.parse_args()

    script_t0 = perf_counter()
    set_seed(args.seed)

    # Get dataset configuration
    dataset_config = DATASET_CONFIGS[args.dataset_name]
    dataset_store_path = dataset_config["store_path"]
    text_field = dataset_config["text_field"]
    category_field = dataset_config["category_field"]

    logger.info("=" * 80)
    logger.info("Direct Prompting Experiment")
    logger.info("=" * 80)
    logger.info("Model: %s", args.model)
    logger.info("Dataset: %s", dataset_config["description"])
    logger.info("Store path: %s", dataset_store_path)
    logger.info("Text field: %s, Category field: %s", text_field, category_field)
    logger.info("Device: %s", args.device)
    logger.info("=" * 80)

    # Load dataset from disk
    logger.info("Loading dataset from disk...")
    dataset_t0 = perf_counter()

    dataset_store = LocalStore(base_path=dataset_store_path)
    dataset = ClassificationDataset.from_disk(
        store=dataset_store,
        text_field=text_field,
        category_field=category_field,
    )

    dataset_load_s = perf_counter() - dataset_t0
    logger.info("✅ Dataset loaded: %d samples (%.2fs)", len(dataset), dataset_load_s)

    # Store for saving results
    results_store = LocalStore(args.store)

    # Extract short model name for run_id
    model_short = args.model.split("/")[-1].replace(".", "_").lower()

    # Run experiments for all prompts
    all_results = []

    for prompt_name, prompt_template in HARMFULNESS_DETECTION_PROMPTS.items():
        logger.info("\n" + "=" * 80)
        logger.info("Running prompt: %s", prompt_name)
        logger.info("=" * 80)
        logger.info("Template: %s", prompt_template[:100] + "...")

        run_t0 = perf_counter()
        ts = _timestamp()
        run_id = f"direct_prompting_{prompt_name}_{model_short}_{args.dataset_name}_{ts}"

        # Create predictor
        init_t0 = perf_counter()
        predictor = create_direct_prompting_predictor(
            model_path=args.model,
            prompt_template=prompt_template,
            prompt_name=prompt_name,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        init_s = perf_counter() - init_t0
        logger.info("✅ Predictor initialized (%.2fs)", init_s)

        # Run predictions
        # Note: ClassificationDataset normalizes all text fields to "text" key in __getitem__
        predict_t0 = perf_counter()
        predictor.predict_dataset(dataset, batch_size=args.batch_size, verbose=True, text_field="text")
        predict_s = perf_counter() - predict_t0
        logger.info("✅ Predictions complete (%.2fs)", predict_s)

        # Save predictions
        save_t0 = perf_counter()
        predictor.save_predictions(run_id=run_id, store=results_store, format="parquet")
        save_s = perf_counter() - save_t0
        logger.info("✅ Predictions saved to: runs/%s", run_id)

        # Run analysis
        run_dir = Path(results_store.base_path) / "runs" / run_id
        analysis_t0 = perf_counter()
        analysis = _evaluate_and_save_analysis(
            run_dir=run_dir,
            dataset=dataset,
            category_field=category_field,
            predictions=predictor.predictions,
            model_name=predictor.model_id,
            prompt_name=prompt_name,
        )
        analysis_s = perf_counter() - analysis_t0

        total_s = perf_counter() - run_t0

        # Save timings
        timings = {
            "dataset_name": args.dataset_name,
            "model": args.model,
            "prompt_name": prompt_name,
            "dataset_load_seconds": dataset_load_s,
            "model_init_seconds": init_s,
            "predict_seconds": predict_s,
            "save_predictions_seconds": save_s,
            "analysis_seconds": analysis_s,
            "total_seconds": total_s,
        }
        _write_timings(run_dir, timings)

        # Collect results summary
        result_summary = {
            "prompt_name": prompt_name,
            "run_id": run_id,
            "f1": analysis["f1"],
            "accuracy": analysis["accuracy"],
            "precision": analysis["precision"],
            "recall": analysis["recall"],
            "refusal_rate": analysis["refusal_rate"],
            "n_refusals": analysis["n_refusals"],
            "total_seconds": total_s,
        }
        all_results.append(result_summary)

        logger.info("📊 Results for %s:", prompt_name)
        logger.info("   F1: %.4f, Accuracy: %.4f", analysis["f1"], analysis["accuracy"])
        logger.info("   Precision: %.4f, Recall: %.4f", analysis["precision"], analysis["recall"])
        logger.info(
            "   Refusal rate: %.2f%% (%d/%d)",
            analysis["refusal_rate"] * 100,
            analysis["n_refusals"],
            analysis["n_total_predictions"],
        )
        logger.info("   Time: %.2fs", total_s)

    # Save combined summary
    logger.info("\n" + "=" * 80)
    logger.info("All prompts complete - Summary")
    logger.info("=" * 80)

    summary_path = (
        Path(results_store.base_path)
        / "runs"
        / f"summary_direct_prompting_{model_short}_{args.dataset_name}_{_timestamp()}.json"
    )
    summary = {
        "model": args.model,
        "dataset_name": args.dataset_name,
        "dataset_description": dataset_config["description"],
        "total_script_seconds": perf_counter() - script_t0,
        "results": all_results,
    }
    _write_json(summary_path, summary)
    logger.info("📄 Summary saved to: %s", summary_path.relative_to(Path.cwd()))

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("Results Summary")
    logger.info("=" * 80)
    logger.info("%-10s  %8s  %8s  %8s  %8s  %10s", "Prompt", "F1", "Acc", "Prec", "Recall", "Refusal%")
    logger.info("-" * 80)
    for r in all_results:
        logger.info(
            "%-10s  %8.4f  %8.4f  %8.4f  %8.4f  %9.2f%%",
            r["prompt_name"],
            r["f1"],
            r["accuracy"],
            r["precision"],
            r["recall"],
            r["refusal_rate"] * 100,
        )
    logger.info("=" * 80)

    # Find best prompt
    best_prompt = max(all_results, key=lambda x: x["f1"])
    logger.info("🏆 Best prompt: %s (F1=%.4f)", best_prompt["prompt_name"], best_prompt["f1"])

    logger.info("\n✅ Experiment complete (total_seconds=%.2f)", perf_counter() - script_t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
