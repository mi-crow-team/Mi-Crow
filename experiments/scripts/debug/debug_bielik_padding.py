"""Debug BielikGuard on WildGuardMix and verify batch padding/truncation.

Goal:
- Ensure adapter/model returns one prediction per input sample.
- Detect any batch where `len(preds) != len(texts)` (would explain fewer saved rows).

Example:
uv run python -m experiments.scripts.debug_bielik_padding --batch-size 32 --limit 200
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.baselines import create_bielik_guard
from experiments.baselines.guard_adapters import BielikGuardAdapter
from mi_crow.datasets import ClassificationDataset
from mi_crow.store import LocalStore
from mi_crow.utils import get_logger

logger = get_logger(__name__)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_token_lengths(adapter: BielikGuardAdapter, texts: List[str]) -> List[int]:
    tok = adapter._pipe.tokenizer  # noqa: SLF001 (debug script)
    # Prefer a fast vectorized path if available.
    try:
        enc = tok(texts, add_special_tokens=True, truncation=False, padding=False)
        return [len(ids) for ids in enc["input_ids"]]
    except Exception:
        return [len(tok.encode(t, add_special_tokens=True)) for t in texts]


def _debug_adapter_batches(
    *,
    adapter: BielikGuardAdapter,
    dataset: ClassificationDataset,
    batch_size: int,
) -> Dict[str, Any]:
    max_len = adapter._effective_max_length()  # noqa: SLF001 (debug script)

    n = len(dataset)
    mismatched_batches: List[Dict[str, Any]] = []
    total_pred_count = 0
    total_inputs = 0
    total_long = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        texts = [dataset[i]["text"] for i in range(start, end)]

        try:
            preds = adapter.predict_batch(texts)
        except Exception:
            logger.exception("Adapter crashed for batch [%d:%d)", start, end)
            mismatched_batches.append(
                {
                    "start": start,
                    "end": end,
                    "n_texts": len(texts),
                    "n_preds": None,
                    "error": "exception",
                }
            )
            continue

        total_inputs += len(texts)
        total_pred_count += len(preds)

        lengths = _safe_token_lengths(adapter, texts)
        long_count = sum(1 for L in lengths if L > max_len)
        total_long += long_count

        if len(preds) != len(texts):
            logger.error(
                "Bielik adapter length mismatch for batch [%d:%d): n_texts=%d n_preds=%d (max_len=%d, long=%d)",
                start,
                end,
                len(texts),
                len(preds),
                max_len,
                long_count,
            )
            # Save a small sample for inspection.
            example_idx: Optional[int] = None
            for i, L in enumerate(lengths):
                if L > max_len:
                    example_idx = i
                    break
            mismatched_batches.append(
                {
                    "start": start,
                    "end": end,
                    "n_texts": len(texts),
                    "n_preds": len(preds),
                    "max_len": max_len,
                    "n_long": long_count,
                    "token_len_min": int(min(lengths)) if lengths else None,
                    "token_len_max": int(max(lengths)) if lengths else None,
                    "example_long_offset": example_idx,
                    "example_long_token_len": int(lengths[example_idx]) if example_idx is not None else None,
                    "example_long_text_prefix": (texts[example_idx][:300] if example_idx is not None else None),
                }
            )

    return {
        "adapter_max_len": max_len,
        "adapter_total_inputs": total_inputs,
        "adapter_total_preds": total_pred_count,
        "adapter_total_long_texts": total_long,
        "adapter_mismatched_batches": mismatched_batches,
    }


def _debug_model_wrapper(
    *,
    model_path: str,
    threshold: float,
    device: str,
    dataset: ClassificationDataset,
    batch_size: int,
    save_predictions: bool,
    store: LocalStore,
) -> Dict[str, Any]:
    guard = create_bielik_guard(model_path=model_path, threshold=threshold, device=device)
    guard.predict_dataset(dataset, batch_size=batch_size, verbose=True, text_field="text")

    preds = guard.predictions
    n_dataset = len(dataset)
    n_preds = len(preds)

    indices = [p.get("sample_index") for p in preds if isinstance(p, dict)]
    indices_int = [int(i) for i in indices if isinstance(i, int)]
    uniq = set(indices_int)

    missing = sorted(set(range(n_dataset)) - uniq)
    dupes = len(indices_int) - len(uniq)

    pred_label_counts = Counter(
        int(p.get("predicted_label")) for p in preds if isinstance(p, dict) and "predicted_label" in p
    )

    run_id = f"debug_bielik_padding_{_timestamp()}"
    saved_path: Optional[str] = None
    if save_predictions:
        guard.save_predictions(run_id=run_id, store=store, format="parquet")
        saved_path = str(Path(store.base_path) / "runs" / run_id)

    return {
        "wrapper_model_id": guard.model_id,
        "wrapper_n_dataset": n_dataset,
        "wrapper_n_predictions": n_preds,
        "wrapper_unique_sample_indices": len(uniq),
        "wrapper_duplicate_sample_indices": dupes,
        "wrapper_missing_sample_indices": missing[:50],
        "wrapper_missing_sample_indices_count": len(missing),
        "wrapper_predicted_label_counts": dict(pred_label_counts),
        "wrapper_saved_run_dir": saved_path,
    }


def main() -> int:
    p = argparse.ArgumentParser()

    p.add_argument("--store", type=str, default="store")
    p.add_argument("--dataset", type=str, default="allenai/wildguardmix")
    p.add_argument("--dataset-config", type=str, default="wildguardtest")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--text-field", type=str, default="prompt")
    p.add_argument("--category-field", type=str, default="prompt_harm_label")
    p.add_argument("--limit", type=int, default=None)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])

    p.add_argument("--bielik-model", type=str, default="speakleash/Bielik-Guard-0.1B-v1.0")
    p.add_argument("--bielik-threshold", type=float, default=0.5)

    p.add_argument("--save-predictions", action="store_true")

    args = p.parse_args()

    store = LocalStore(args.store)

    logger.info("Loading dataset %s (%s/%s)", args.dataset, args.dataset_config, args.split)
    dataset = ClassificationDataset.from_huggingface(
        repo_id=args.dataset,
        store=store,
        name=args.dataset_config,
        split=args.split,
        text_field=args.text_field,
        category_field=args.category_field,
        limit=args.limit,
    )

    logger.info(
        "Init Bielik adapter model=%s threshold=%.3f device=%s",
        args.bielik_model,
        args.bielik_threshold,
        args.device,
    )
    adapter = BielikGuardAdapter(model_path=args.bielik_model, threshold=args.bielik_threshold, device=args.device)

    adapter_report = _debug_adapter_batches(adapter=adapter, dataset=dataset, batch_size=args.batch_size)
    wrapper_report = _debug_model_wrapper(
        model_path=args.bielik_model,
        threshold=args.bielik_threshold,
        device=args.device,
        dataset=dataset,
        batch_size=args.batch_size,
        save_predictions=args.save_predictions,
        store=store,
    )

    report: Dict[str, Any] = {
        "dataset_len": len(dataset),
        "batch_size": args.batch_size,
        "device": args.device,
        "model": args.bielik_model,
        "threshold": args.bielik_threshold,
        **adapter_report,
        **wrapper_report,
    }

    out_dir = Path(store.base_path) / "debug" / f"bielik_padding_{_timestamp()}"
    _write_json(out_dir / "report.json", report)

    logger.info("Wrote report to %s", out_dir / "report.json")
    logger.info(
        "Summary: dataset=%d adapter_preds=%s wrapper_preds=%s mismatched_batches=%d missing_indices=%d",
        report["dataset_len"],
        report.get("adapter_total_preds"),
        report.get("wrapper_n_predictions"),
        len(report.get("adapter_mismatched_batches", [])),
        report.get("wrapper_missing_sample_indices_count", -1),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
