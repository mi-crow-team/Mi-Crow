# ruff: noqa
"""Utilities for loading baseline experiment results from disk."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def parse_baseline_run_id(run_id: str) -> Dict[str, str]:
    """Parse baseline guard run_id to extract parameters.

    Example: baseline_bielik_wgmix_test_20260105_045726

    Returns:
        Dictionary with keys: method, model, dataset
    """
    pattern = r"baseline_(bielik|llamaguard)_(plmix|wgmix)_test_\d{8}_\d{6}"
    match = re.match(pattern, run_id)

    if not match:
        raise ValueError(f"Could not parse baseline run_id: {run_id}")

    model_part = match.group(1)
    dataset = match.group(2) + "_test"

    # Map model parts to readable names
    model_map = {
        "bielik": "Bielik-Guard-0.1B",
        "llamaguard": "Llama-Guard-3-1B",
    }

    return {
        "method": "Baseline Guard",
        "model": model_map.get(model_part, model_part),
        "model_short": model_part,
        "test_dataset": dataset,
    }


def parse_direct_prompting_run_id(run_id: str) -> Dict[str, str]:
    """Parse direct prompting run_id to extract parameters.

    Example: direct_prompting_prompt_0_bielik-4_5b-v3_0-instruct_plmix_test_20260105_213956

    Returns:
        Dictionary with keys: method, model, dataset, prompt_id
    """
    pattern = r"direct_prompting_prompt_(\d+)_(bielik-4_5b-v3_0-instruct|llama-3_2-3b-instruct)_(plmix|wgmix)_test_\d{8}_\d{6}"
    match = re.match(pattern, run_id)

    if not match:
        raise ValueError(f"Could not parse direct prompting run_id: {run_id}")

    prompt_id = int(match.group(1))
    model_part = match.group(2)
    dataset = match.group(3) + "_test"

    # Map model parts to readable names
    model_map = {
        "bielik-4_5b-v3_0-instruct": "Bielik-4.5B-Prompted",
        "llama-3_2-3b-instruct": "Llama-3.2-3B-Prompted",
    }

    return {
        "method": "Direct Prompting",
        "model": model_map.get(model_part, model_part),
        "model_short": model_part,
        "test_dataset": dataset,
        "prompt_id": prompt_id,
    }


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Load metrics.json file.

    Args:
        metrics_path: Path to metrics.json file

    Returns:
        Dictionary containing metrics
    """
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_baseline_results(store_path: Path, run_ids: List[str]) -> pd.DataFrame:
    """Load all baseline guard results into a DataFrame.

    Args:
        store_path: Path to store/runs directory
        run_ids: List of baseline run_ids to load

    Returns:
        DataFrame with columns: method, model, test_dataset, f1, precision, recall, accuracy
    """
    results = []

    for run_id in run_ids:
        try:
            params = parse_baseline_run_id(run_id)
            run_dir = store_path / "runs" / run_id

            if not run_dir.exists():
                print(f"Warning: Run directory not found: {run_dir}")
                continue

            metrics_path = run_dir / "analysis" / "metrics.json"
            if not metrics_path.exists():
                print(f"Warning: Metrics file not found: {metrics_path}")
                continue

            metrics = load_metrics(metrics_path)

            result = {
                **params,
                "f1": metrics.get("f1", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "accuracy": metrics.get("accuracy", 0.0),
                "n": metrics.get("n", 0),
            }

            results.append(result)

        except Exception as e:
            print(f"Error loading {run_id}: {e}")
            continue

    return pd.DataFrame(results)


def load_direct_prompting_results(store_path: Path, run_ids: List[str]) -> pd.DataFrame:
    """Load all direct prompting results into a DataFrame.

    Args:
        store_path: Path to store/runs directory
        run_ids: List of direct prompting run_ids to load

    Returns:
        DataFrame with columns: method, model, test_dataset, prompt_id,
        f1, precision, recall, accuracy
    """
    results = []

    for run_id in run_ids:
        try:
            params = parse_direct_prompting_run_id(run_id)
            run_dir = store_path / "runs" / run_id

            if not run_dir.exists():
                print(f"Warning: Run directory not found: {run_dir}")
                continue

            metrics_path = run_dir / "analysis" / "metrics.json"
            if not metrics_path.exists():
                print(f"Warning: Metrics file not found: {metrics_path}")
                continue

            metrics = load_metrics(metrics_path)

            result = {
                **params,
                "f1": metrics.get("f1", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "accuracy": metrics.get("accuracy", 0.0),
                "n": metrics.get("n", 0),
            }

            results.append(result)

        except Exception as e:
            print(f"Error loading {run_id}: {e}")
            continue

    return pd.DataFrame(results)
