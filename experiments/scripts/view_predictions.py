"""
Usage:
uv run python -m experiments.scripts.view_predictions \
  --run-id baseline_bielik_20251228_223543 \
  --add-prompt --add-label \
  --head 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def _prompt_preview(text: object, max_chars: int) -> Optional[str]:
    if text is None:
        return None
    s = str(text).replace("\n", " ").strip()
    if max_chars <= 0:
        return s
    return (s[: max_chars - 1] + "…") if len(s) > max_chars else s


def _load_hf_dataset(*, repo_id: str, config: Optional[str], split: str):
    try:
        from datasets import load_dataset
    except Exception as e:
        # Fallback for this repo: use Mi-Crow's dataset helper (it wraps HF datasets).
        try:
            from mi_crow.datasets import ClassificationDataset
            from mi_crow.store import LocalStore
        except Exception:
            raise SystemExit(
                "Reading prompts/labels requires HuggingFace `datasets` or Mi-Crow installed.\n"
                "Install e.g. `pip install datasets`, or run inside the project's environment.\n"
                f"Original error: {e}"
            )

        store = LocalStore("store")

        # Return the ClassificationDataset itself; items are dict-like.
        return ClassificationDataset.from_huggingface(
            repo_id=repo_id,
            store=store,
            name=config,
            split=split,
            text_field="prompt",
            category_field="prompt_harm_label",
        )

    kwargs = {"path": repo_id, "split": split}
    if config:
        kwargs["name"] = config
    return load_dataset(**kwargs)


def _augment_with_dataset_columns(
    *,
    rows_df,
    repo_id: str,
    config: Optional[str],
    split: str,
    prompt_column: str,
    label_column: str,
    prompt_chars: int,
    add_prompt: bool,
    add_label: bool,
):
    if "sample_index" not in rows_df.columns:
        raise SystemExit("Cannot join to dataset: missing `sample_index` column in predictions")

    ds = _load_hf_dataset(repo_id=repo_id, config=config, split=split)

    prompt_previews = []
    labels = []
    for idx in rows_df["sample_index"].tolist():
        item = ds[int(idx)]
        if add_prompt:
            value = item.get(prompt_column) if hasattr(item, "get") else item[prompt_column]
            prompt_previews.append(_prompt_preview(value, prompt_chars))
        if add_label:
            value = item.get(label_column) if hasattr(item, "get") else item[label_column]
            labels.append(value)

    if add_prompt:
        rows_df.insert(0, "prompt_preview", prompt_previews)
    if add_label:
        rows_df.insert(0, "true_label", labels)

    return rows_df


def _resolve_predictions_path(*, store: Path, run_id: Optional[str], path: Optional[str]) -> Path:
    if path:
        p = Path(path).expanduser()
        return p if p.is_absolute() else (Path.cwd() / p)
    if not run_id:
        raise SystemExit("Provide --run-id or --path")
    return store / "runs" / run_id / "predictions.parquet"


def _print_table(df, head: int) -> None:
    try:
        import pandas as pd  # noqa: F401
    except Exception:
        # pandas isn't strictly required for printing; but df will be pandas for parquet reads.
        pass

    if head > 0:
        print(df.head(head).to_string(index=False))
    else:
        print(df.to_string(index=False))


def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser(description="Preview saved predictions artifacts (parquet/json).")
    parser.add_argument("--store", type=str, default="store", help="Base store directory")
    parser.add_argument("--run-id", type=str, default=None, help="Run id under store/runs/<run_id>/")
    parser.add_argument("--path", type=str, default=None, help="Direct path to predictions.parquet or predictions.json")
    parser.add_argument("--head", type=int, default=20, help="Rows to show (use 0 for all)")
    parser.add_argument("--columns", type=str, default=None, help="Comma-separated columns to show")
    parser.add_argument("--info", action="store_true", help="Print columns/dtypes and row count")

    parser.add_argument(
        "--add-prompt", action="store_true", help="Add `prompt_preview` column (joined by sample_index)"
    )
    parser.add_argument("--add-label", action="store_true", help="Add `true_label` column (joined by sample_index)")
    parser.add_argument("--prompt-chars", type=int, default=100, help="Max chars for prompt preview")

    parser.add_argument("--dataset", type=str, default="allenai/wildguardmix", help="HF dataset repo id")
    parser.add_argument("--dataset-config", type=str, default="wildguardtest", help="HF dataset config/name (or empty)")
    parser.add_argument("--dataset-split", type=str, default="test", help="HF dataset split")
    parser.add_argument("--prompt-column", type=str, default="prompt", help="Dataset column for prompt text")
    parser.add_argument(
        "--label-column", type=str, default="prompt_harm_label", help="Dataset column for ground-truth label"
    )

    args = parser.parse_args()

    store = Path(args.store)
    pred_path = _resolve_predictions_path(store=store, run_id=args.run_id, path=args.path)

    if not pred_path.exists():
        # Common fallback if parquet wasn't available
        if pred_path.suffix == ".parquet":
            json_path = pred_path.with_suffix(".json")
            if json_path.exists():
                pred_path = json_path
            else:
                raise SystemExit(f"Not found: {pred_path} (or {json_path})")
        else:
            raise SystemExit(f"Not found: {pred_path}")

    cols = None
    if args.columns:
        cols = [c.strip() for c in args.columns.split(",") if c.strip()]

    if pred_path.suffix == ".parquet":
        try:
            import pandas as pd
        except Exception as e:
            raise SystemExit(
                "pandas+pyarrow is required to read parquet. Install e.g. `uv pip install pandas pyarrow`.\n"
                f"Original error: {e}"
            )

        df = pd.read_parquet(pred_path)

        # Join dataset columns only for the rows we're about to display.
        if args.add_prompt or args.add_label:
            display_df = df if args.head == 0 else df.head(args.head).copy()
            display_df = _augment_with_dataset_columns(
                rows_df=display_df,
                repo_id=args.dataset,
                config=(args.dataset_config or None),
                split=args.dataset_split,
                prompt_column=args.prompt_column,
                label_column=args.label_column,
                prompt_chars=args.prompt_chars,
                add_prompt=args.add_prompt,
                add_label=args.add_label,
            )
        else:
            display_df = df if args.head == 0 else df.head(args.head)
        if cols:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise SystemExit(f"Missing columns: {missing}. Available: {list(df.columns)}")
            display_df = display_df[cols]

        if args.info:
            print(f"path: {pred_path}")
            print(f"rows: {len(df)}")
            print("columns:")
            for c in df.columns:
                print(f"  - {c}: {df[c].dtype}")
            print()

        _print_table(display_df, head=0)
        return 0

    if pred_path.suffix == ".json":
        import json

        data = json.loads(pred_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit("Expected JSON predictions to be a list of dicts")

        if args.info:
            print(f"path: {pred_path}")
            print(f"rows: {len(data)}")
            if data:
                print(f"keys: {sorted(set().union(*(d.keys() for d in data if isinstance(d, dict))))}")
            print()

        rows = data if args.head == 0 else data[: args.head]

        if args.add_prompt or args.add_label:
            ds = _load_hf_dataset(repo_id=args.dataset, config=(args.dataset_config or None), split=args.dataset_split)
            for r in rows:
                idx = r.get("sample_index") if isinstance(r, dict) else None
                if idx is None:
                    continue
                item = ds[int(idx)]
                if args.add_prompt:
                    value = item.get(args.prompt_column) if hasattr(item, "get") else item[args.prompt_column]
                    r["prompt_preview"] = _prompt_preview(value, args.prompt_chars)
                if args.add_label:
                    value = item.get(args.label_column) if hasattr(item, "get") else item[args.label_column]
                    r["true_label"] = value
        if cols:
            rows = [{k: r.get(k) for k in cols} for r in rows]
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    raise SystemExit(f"Unsupported file type: {pred_path}")


if __name__ == "__main__":
    raise SystemExit(main())
