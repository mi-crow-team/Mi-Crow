#  ruff: noqa
"""Utilities for loading experiment results from disk."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def parse_lpm_run_id(run_id: str) -> Dict[str, str]:
    """Parse LPM run_id to extract parameters.

    Example: lpm_bielik_1_5b_plmix_train_plmix_test_last_token_layer31_euclidean

    Returns:
        Dictionary with keys: method, model, train_dataset, test_dataset,
        aggregation, layer, metric
    """
    pattern = r"lpm_(.+?)_(plmix|wgmix)_train_(plmix|wgmix)_test_(mean|last_token|last_token_prefix)_layer(\d+)_(euclidean|mahalanobis)"
    match = re.match(pattern, run_id)

    if not match:
        raise ValueError(f"Could not parse LPM run_id: {run_id}")

    model_part = match.group(1)

    # Map model parts to readable names
    model_map = {
        "bielik_1_5b": "Bielik-1.5B",
        "bielik_4_5b": "Bielik-4.5B",
        "llama_3b": "Llama-3.2-3B",
    }

    return {
        "method": "LPM",
        "model": model_map.get(model_part, model_part),
        "model_short": model_part,
        "train_dataset": match.group(2) + "_train",
        "test_dataset": match.group(3) + "_test",
        "aggregation": match.group(4),
        "layer": int(match.group(5)),
        "metric": match.group(6),
    }


def parse_probe_run_id(run_id: str) -> Dict[str, str]:
    """Parse probe run_id to extract parameters.

    Example: probe_bielik_1_5b_plmix_train_plmix_test_last_token_layer31

    Returns:
        Dictionary with keys: method, model, train_dataset, test_dataset,
        aggregation, layer
    """
    pattern = r"probe_(.+?)_(plmix|wgmix)_train_(plmix|wgmix)_test_(mean|last_token|last_token_prefix)_layer(\d+)"
    match = re.match(pattern, run_id)

    if not match:
        raise ValueError(f"Could not parse probe run_id: {run_id}")

    model_part = match.group(1)

    # Map model parts to readable names
    model_map = {
        "bielik_1_5b": "Bielik-1.5B",
        "bielik_4_5b": "Bielik-4.5B",
        "llama_3b": "Llama-3.2-3B",
    }

    return {
        "method": "Linear Probe",
        "model": model_map.get(model_part, model_part),
        "model_short": model_part,
        "train_dataset": match.group(2) + "_train",
        "test_dataset": match.group(3) + "_test",
        "aggregation": match.group(4),
        "layer": int(match.group(5)),
    }


def find_latest_inference_run(run_base_dir: Path) -> Optional[Path]:
    """Find the most recent inference_[timestamp] directory.

    Args:
        run_base_dir: Base directory containing runs/ subdirectory

    Returns:
        Path to the latest inference directory, or None if not found
    """
    runs_dir = run_base_dir / "runs"
    if not runs_dir.exists():
        return None

    inference_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("inference_")],
        key=lambda x: x.name,
        reverse=True,
    )

    return inference_dirs[0] if inference_dirs else None


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Load metrics.json file.

    Args:
        metrics_path: Path to metrics.json file

    Returns:
        Dictionary containing metrics
    """
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def recalculate_accuracy(metrics: Dict[str, Any]) -> float:
    """Recalculate accuracy from confusion matrix values.

    The accuracy in metrics.json for probes is sometimes incorrect.
    This recalculates it from tp, tn, fp, fn.

    Args:
        metrics: Metrics dictionary with tp, tn, fp, fn keys

    Returns:
        Corrected accuracy value
    """
    tp = metrics.get("tp", 0)
    tn = metrics.get("tn", 0)
    fp = metrics.get("fp", 0)
    fn = metrics.get("fn", 0)

    total = tp + tn + fp + fn
    if total == 0:
        return 0.0

    return (tp + tn) / total


def load_lpm_results(store_path: Path, run_ids: List[str]) -> pd.DataFrame:
    """Load all LPM results into a DataFrame.

    Args:
        store_path: Path to store directory
        run_ids: List of LPM run_ids to load

    Returns:
        DataFrame with columns: method, model, train_dataset, test_dataset,
        aggregation, layer, metric, f1, precision, recall, accuracy
    """
    results = []

    for run_id in run_ids:
        try:
            params = parse_lpm_run_id(run_id)
            run_dir = store_path / run_id

            if not run_dir.exists():
                print(f"Warning: Run directory not found: {run_dir}")
                continue

            latest_inference = find_latest_inference_run(run_dir)
            if latest_inference is None:
                print(f"Warning: No inference run found for: {run_id}")
                continue

            metrics_path = latest_inference / "analysis" / "metrics.json"
            if not metrics_path.exists():
                print(f"Warning: Metrics file not found: {metrics_path}")
                continue

            metrics = load_metrics(metrics_path)

            # Recalculate accuracy to be safe
            accuracy = recalculate_accuracy(metrics)

            result = {
                **params,
                "f1": metrics.get("f1", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "accuracy": accuracy,
                "n": metrics.get("n", 0),
            }

            results.append(result)

        except Exception as e:
            print(f"Error loading {run_id}: {e}")
            continue

    return pd.DataFrame(results)


def load_probe_results(store_path: Path, run_ids: List[str]) -> pd.DataFrame:
    """Load all Linear Probe results into a DataFrame.

    Args:
        store_path: Path to store directory
        run_ids: List of probe run_ids to load

    Returns:
        DataFrame with columns: method, model, train_dataset, test_dataset,
        aggregation, layer, f1, precision, recall, accuracy
    """
    results = []

    for run_id in run_ids:
        try:
            params = parse_probe_run_id(run_id)
            run_dir = store_path / run_id

            if not run_dir.exists():
                print(f"Warning: Run directory not found: {run_dir}")
                continue

            latest_inference = find_latest_inference_run(run_dir)
            if latest_inference is None:
                print(f"Warning: No inference run found for: {run_id}")
                continue

            metrics_path = latest_inference / "analysis" / "metrics.json"
            if not metrics_path.exists():
                print(f"Warning: Metrics file not found: {metrics_path}")
                continue

            metrics = load_metrics(metrics_path)

            # IMPORTANT: Recalculate accuracy (it's wrong in metrics.json for probes)
            accuracy = recalculate_accuracy(metrics)

            result = {
                **params,
                "f1": metrics.get("f1", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "accuracy": accuracy,
                "n": metrics.get("n", 0),
            }

            results.append(result)

        except Exception as e:
            print(f"Error loading {run_id}: {e}")
            continue

    return pd.DataFrame(results)
