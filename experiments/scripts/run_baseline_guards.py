"""
Run baseline guard models (BielikGuard, LlamaGuard) on test datasets.

This script evaluates baseline models on cached test datasets (WGMix or PLMix).
Datasets must be prepared first using experiments.scripts.prepare_datasets.

Usage:
    # WildGuardMix Test (English)
    uv run python -m experiments.scripts.run_baseline_guards \
        --dataset-name wgmix_test --run-bielik --device cpu

    # PL Mix Test (Polish)
    uv run python -m experiments.scripts.run_baseline_guards \
        --dataset-name plmix_test --run-llama --llama-model <PATH> --device cpu
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

from experiments.baselines import create_bielik_guard, create_llama_guard
from experiments.scripts.analysis_utils import (
    compute_binary_metrics,
    map_wildguard_label_to_binary,
    save_confusion_matrix_plot,
    save_threat_category_bar,
)
from experiments.scripts.config import DATASET_CONFIGS
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
) -> None:
    # Align by sample_index
    y_true: List[int] = []
    y_pred: List[int] = []
    threat_vals: List[Any] = []

    n_total_predictions = 0
    n_skipped_no_sample_index = 0
    n_skipped_missing_gt = 0
    n_skipped_unmappable_gt = 0
    n_skipped_missing_pred = 0

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
        threat_vals.append(p.get("threat_category"))

    metrics = compute_binary_metrics(y_true, y_pred)
    analysis = {
        "dataset_len": len(dataset),
        "n_total_predictions": n_total_predictions,
        "n_used_for_metrics": metrics.n,
        "n_skipped_no_sample_index": n_skipped_no_sample_index,
        "n_skipped_missing_gt": n_skipped_missing_gt,
        "n_skipped_unmappable_gt": n_skipped_unmappable_gt,
        "n_skipped_missing_predicted_label": n_skipped_missing_pred,
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
    # Backwards-compat filename (some older runs refer to analysis.json).
    _write_json(out_analysis_dir / "analysis.json", analysis)

    save_confusion_matrix_plot(
        (metrics.tp, metrics.tn, metrics.fp, metrics.fn),
        run_dir / "analysis" / "confusion_matrix.png",
        title=f"{model_name} confusion matrix",
    )

    save_threat_category_bar(
        threat_vals,
        run_dir / "analysis" / "threat_categories.png",
        title=f"{model_name} threat categories",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline guard models on test datasets")

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

    # Model selection
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])

    parser.add_argument("--run-bielik", action="store_true", help="Run BielikGuard model")
    parser.add_argument("--bielik-model", type=str, default="speakleash/Bielik-Guard-0.1B-v1.0")
    parser.add_argument("--bielik-threshold", type=float, default=0.5)

    parser.add_argument("--run-llama", action="store_true", help="Run LlamaGuard model")
    parser.add_argument(
        "--llama-model", type=str, default="meta-llama/Llama-Guard-3-1B", help="HF model id/path for LlamaGuard"
    )

    args = parser.parse_args()

    script_t0 = perf_counter()
    set_seed(args.seed)

    # Get dataset configuration
    dataset_config = DATASET_CONFIGS[args.dataset_name]
    dataset_store_path = dataset_config["store_path"]
    text_field = dataset_config["text_field"]
    category_field = dataset_config["category_field"]

    logger.info("=" * 80)
    logger.info("Dataset: %s", dataset_config["description"])
    logger.info("Store path: %s", dataset_store_path)
    logger.info("Text field: %s, Category field: %s", text_field, category_field)
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
    ts = _timestamp()

    if not args.run_bielik and not args.run_llama:
        logger.warning("No models selected; use --run-bielik and/or --run-llama")
        return 2

    if args.run_bielik:
        run_t0 = perf_counter()

        init_t0 = perf_counter()
        bielik = create_bielik_guard(model_path=args.bielik_model, threshold=args.bielik_threshold, device=args.device)
        init_s = perf_counter() - init_t0

        run_id = f"baseline_bielik_{args.dataset_name}_{ts}"
        logger.info("Running BielikGuard -> run_id=%s", run_id)

        predict_t0 = perf_counter()
        bielik.predict_dataset(dataset, batch_size=args.batch_size, verbose=True, text_field="text")
        predict_s = perf_counter() - predict_t0

        save_t0 = perf_counter()
        bielik.save_predictions(run_id=run_id, store=results_store, format="parquet")
        save_s = perf_counter() - save_t0

        run_dir = Path(results_store.base_path) / "runs" / run_id

        analysis_t0 = perf_counter()
        _evaluate_and_save_analysis(
            run_dir=run_dir,
            dataset=dataset,
            category_field=category_field,
            predictions=bielik.predictions,
            model_name=bielik.model_id,
        )
        analysis_s = perf_counter() - analysis_t0

        total_s = perf_counter() - run_t0
        timings = {
            "dataset_name": args.dataset_name,
            "dataset_load_seconds": dataset_load_s,
            "model_init_seconds": init_s,
            "predict_seconds": predict_s,
            "save_predictions_seconds": save_s,
            "analysis_seconds": analysis_s,
            "total_seconds": total_s,
        }
        _write_timings(run_dir, timings)
        logger.info("Bielik timings (s): %s", timings)

    if args.run_llama:
        if not args.llama_model:
            raise SystemExit("--llama-model is required when --run-llama is set")

        run_t0 = perf_counter()

        init_t0 = perf_counter()
        llama = create_llama_guard(model_path=args.llama_model, device=args.device)
        init_s = perf_counter() - init_t0

        run_id = f"baseline_llamaguard_{args.dataset_name}_{ts}"
        logger.info("Running LlamaGuard -> run_id=%s", run_id)

        # Per request: use predict_batch (generation) only.
        predict_t0 = perf_counter()
        llama.predict_dataset(dataset, batch_size=args.batch_size, verbose=True, text_field="text")
        predict_s = perf_counter() - predict_t0

        save_t0 = perf_counter()
        llama.save_predictions(run_id=run_id, store=results_store, format="parquet")
        save_s = perf_counter() - save_t0

        run_dir = Path(results_store.base_path) / "runs" / run_id

        analysis_t0 = perf_counter()
        _evaluate_and_save_analysis(
            run_dir=run_dir,
            dataset=dataset,
            category_field=category_field,
            predictions=llama.predictions,
            model_name=llama.model_id,
        )
        analysis_s = perf_counter() - analysis_t0

        total_s = perf_counter() - run_t0
        timings = {
            "dataset_name": args.dataset_name,
            "dataset_load_seconds": dataset_load_s,
            "model_init_seconds": init_s,
            "predict_seconds": predict_s,
            "save_predictions_seconds": save_s,
            "analysis_seconds": analysis_s,
            "total_seconds": total_s,
        }
        _write_timings(run_dir, timings)
        logger.info("LlamaGuard timings (s): %s", timings)

    logger.info("Done (script_total_seconds=%.3f)", perf_counter() - script_t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
