"""Debug LlamaGuard chat template + small batched run.

Goal:
- Verify whether `tokenizer.apply_chat_template(...)` exists and succeeds.
- Print/log prompt template preview and batched-generation decoded continuations.

Example:
uv run python -m experiments.scripts.debug_llama_chat_template --limit 6 --batch-size 3
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from experiments.baselines.guard_adapters import LlamaGuardAdapter
from mi_crow.datasets import ClassificationDataset
from mi_crow.store import LocalStore
from mi_crow.utils import get_logger

logger = get_logger(__name__)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()

    p.add_argument("--store", type=str, default="store")
    p.add_argument("--dataset", type=str, default="allenai/wildguardmix")
    p.add_argument("--dataset-config", type=str, default="wildguardtest")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--text-field", type=str, default="prompt")
    p.add_argument("--category-field", type=str, default="prompt_harm_label")

    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=3)

    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--llama-model", type=str, default="meta-llama/Llama-Guard-3-1B")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)

    args = p.parse_args()

    store = LocalStore(args.store)

    logger.info("Loading dataset %s (%s/%s) limit=%s", args.dataset, args.dataset_config, args.split, args.limit)
    dataset = ClassificationDataset.from_huggingface(
        repo_id=args.dataset,
        store=store,
        name=args.dataset_config,
        split=args.split,
        text_field=args.text_field,
        category_field=args.category_field,
        limit=args.limit,
    )

    adapter = LlamaGuardAdapter(
        model_path=args.llama_model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Force very chatty logging for this debug run.
    adapter._true_batch_log_first_n = 9999  # noqa: SLF001
    adapter._true_batch_log_every_n = 1  # noqa: SLF001

    samples = [dataset[i]["text"] for i in range(len(dataset))]

    template_results: List[Dict[str, Any]] = []

    for i, text in enumerate(samples):
        messages = [{"role": "user", "content": text}]
        messages_mm = [{"role": "user", "content": [{"type": "text", "text": text}]}]

        has_apply = hasattr(adapter._tokenizer, "apply_chat_template")  # noqa: SLF001
        logger.info("Sample %d: tokenizer has apply_chat_template=%s", i, has_apply)

        applied_ok = False
        applied_preview = None
        applied_error = None

        applied_mm_ok = False
        applied_mm_preview = None
        applied_mm_error = None

        if has_apply:
            try:
                applied = adapter._tokenizer.apply_chat_template(  # noqa: SLF001
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                applied_ok = True
                applied_preview = applied[:700]
                logger.info("Sample %d: apply_chat_template preview: %r", i, applied_preview[:700])
            except Exception as e:
                applied_error = repr(e)
                logger.warning("Sample %d: apply_chat_template FAILED: %s", i, applied_error, exc_info=True)

            try:
                applied_mm = adapter._tokenizer.apply_chat_template(  # noqa: SLF001
                    messages_mm,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                applied_mm_ok = True
                applied_mm_preview = applied_mm[:700]
                logger.info("Sample %d: apply_chat_template(mm) preview: %r", i, applied_mm_preview[:700])
            except Exception as e:
                applied_mm_error = repr(e)
                logger.warning("Sample %d: apply_chat_template(mm) FAILED: %s", i, applied_mm_error, exc_info=True)

        # Also exercise adapter logic (this is where your existing log line lives).
        try:
            prompt = adapter._build_prompt(text)  # noqa: SLF001
            prompt_preview = prompt[:300]
            # Quick sanity: if the prompt doesn't include the user text, it will classify an "empty conversation".
            needle = " ".join(text.strip().split())[:80]
            hay = " ".join(prompt.split())
            prompt_contains_text = (not needle) or (needle in hay)
            if not prompt_contains_text:
                logger.warning(
                    "Sample %d: adapter prompt does NOT contain user text snippet; likely empty <BEGIN CONVERSATION>.",
                    i,
                )
        except Exception as e:
            prompt_preview = None
            prompt_contains_text = False
            logger.warning("Sample %d: adapter._build_prompt FAILED: %r", i, e, exc_info=True)

        template_results.append(
            {
                "i": i,
                "has_apply_chat_template": has_apply,
                "apply_ok": applied_ok,
                "apply_preview": applied_preview,
                "apply_error": applied_error,
                "apply_mm_ok": applied_mm_ok,
                "apply_mm_preview": applied_mm_preview,
                "apply_mm_error": applied_mm_error,
                "adapter_build_prompt_preview": prompt_preview,
                "adapter_prompt_contains_text": prompt_contains_text,
            }
        )

    # Run a tiny batched prediction to exercise the true-batching codepath.
    preds: List[Dict[str, Any]] = []
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start : start + args.batch_size]
        batch_preds = adapter.predict_batch(batch)
        preds.extend(batch_preds)

    out = {
        "model": args.llama_model,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "n_samples": len(samples),
        "batch_size": args.batch_size,
        "template_checks": template_results,
        "predictions": preds,
    }

    out_dir = Path(store.base_path) / "debug" / f"llama_chat_template_{_timestamp()}"
    _write_json(out_dir / "report.json", out)
    logger.info("Wrote report to %s", out_dir / "report.json")

    # Print a short summary to stdout for Slurm logs.
    if preds:
        logger.info("Example decoded raw_output (sample 0): %r", preds[0].get("raw_output"))
        logger.info("Example parsed threat_category (sample 0): %r", preds[0].get("threat_category"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
