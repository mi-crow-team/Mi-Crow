from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

from amber.datasets import ClassificationDataset
from amber.store import LocalStore
from amber.utils import get_logger
from experiments.baselines import create_bielik_guard, create_llama_guard
from experiments.scripts.analysis_utils import (
    compute_binary_metrics,
    map_wildguard_label_to_binary,
    save_confusion_matrix_plot,
    save_threat_category_bar,
)

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

    for p in predictions:
        idx = p.get("sample_index")
        if idx is None:
            continue
        gt_raw = dataset[int(idx)].get(category_field)
        gt = map_wildguard_label_to_binary(gt_raw)
        if gt is None:
            continue

        y_true.append(gt)
        y_pred.append(int(p.get("predicted_label")))
        threat_vals.append(p.get("threat_category"))

    metrics = compute_binary_metrics(y_true, y_pred)
    analysis = {
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

    _write_json(run_dir / "analysis" / "metrics.json", analysis)

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
    parser = argparse.ArgumentParser()

    parser.add_argument("--store", type=str, default="store", help="LocalStore base path")
    parser.add_argument("--dataset", type=str, default="allenai/wildguardmix")
    parser.add_argument("--dataset-config", type=str, default="wildguardtest")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--text-field", type=str, default="prompt")
    parser.add_argument("--category-field", type=str, default="prompt_harm_label")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])

    parser.add_argument("--run-bielik", action="store_true")
    parser.add_argument("--bielik-model", type=str, default="speakleash/Bielik-Guard-0.1B-v1.0")
    parser.add_argument("--bielik-threshold", type=float, default=0.5)

    parser.add_argument("--run-llama", action="store_true")
    parser.add_argument("--llama-model", type=str, default=None, help="HF model id/path for LlamaGuard")

    args = parser.parse_args()

    script_t0 = perf_counter()

    store = LocalStore(args.store)

    logger.info("Loading dataset %s (%s/%s)", args.dataset, args.dataset_config, args.split)
    dataset_t0 = perf_counter()
    dataset = ClassificationDataset.from_huggingface(
        repo_id=args.dataset,
        store=store,
        name=args.dataset_config,
        split=args.split,
        text_field=args.text_field,
        category_field=args.category_field,
        limit=args.limit,
    )
    dataset_load_s = perf_counter() - dataset_t0

    ts = _timestamp()

    if not args.run_bielik and not args.run_llama:
        logger.warning("No models selected; use --run-bielik and/or --run-llama")
        return 2

    if args.run_bielik:
        run_t0 = perf_counter()

        init_t0 = perf_counter()
        bielik = create_bielik_guard(model_path=args.bielik_model, threshold=args.bielik_threshold, device=args.device)
        init_s = perf_counter() - init_t0

        run_id = f"baseline_bielik_{ts}"
        logger.info("Running BielikGuard -> run_id=%s", run_id)

        predict_t0 = perf_counter()
        bielik.predict_dataset(dataset, batch_size=args.batch_size, verbose=True, text_field="text")
        predict_s = perf_counter() - predict_t0

        save_t0 = perf_counter()
        bielik.save_predictions(run_id=run_id, store=store, format="parquet")
        save_s = perf_counter() - save_t0

        run_dir = Path(store.base_path) / "runs" / run_id

        analysis_t0 = perf_counter()
        _evaluate_and_save_analysis(
            run_dir=run_dir,
            dataset=dataset,
            category_field=args.category_field,
            predictions=bielik.predictions,
            model_name=bielik.model_id,
        )
        analysis_s = perf_counter() - analysis_t0

        total_s = perf_counter() - run_t0
        timings = {
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

        run_id = f"baseline_llamaguard_{ts}"
        logger.info("Running LlamaGuard -> run_id=%s", run_id)

        # Per request: use predict_batch (generation) only.
        predict_t0 = perf_counter()
        llama.predict_dataset(dataset, batch_size=1, verbose=True, text_field="text")
        predict_s = perf_counter() - predict_t0

        save_t0 = perf_counter()
        llama.save_predictions(run_id=run_id, store=store, format="parquet")
        save_s = perf_counter() - save_t0

        run_dir = Path(store.base_path) / "runs" / run_id

        analysis_t0 = perf_counter()
        _evaluate_and_save_analysis(
            run_dir=run_dir,
            dataset=dataset,
            category_field=args.category_field,
            predictions=llama.predictions,
            model_name=llama.model_id,
        )
        analysis_s = perf_counter() - analysis_t0

        total_s = perf_counter() - run_t0
        timings = {
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
