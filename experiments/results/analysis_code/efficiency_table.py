# ruff: noqa
"""Create efficiency comparison tables showing FLOPs and F1 metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd


def load_flops_data(flops_json_path: Path) -> Dict:
    """Load FLOPs data from JSON file.

    Args:
        flops_json_path: Path to flops.json file

    Returns:
        Dictionary containing FLOPs data
    """
    with open(flops_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_efficiency_comparison_table(
    lpm_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    prompting_df: pd.DataFrame,
    flops_data: Dict,
    output_path: Path,
) -> pd.DataFrame:
    """Create a comprehensive efficiency comparison table.

    This table compares mechanistic interpretability methods (LPM, Linear Probe)
    with baseline models (Guards, Prompted LLMs) in terms of computational
    efficiency (FLOPs) and performance (F1 score).

    Args:
        lpm_df: DataFrame with LPM results
        probe_df: DataFrame with Linear Probe results
        baseline_df: DataFrame with baseline guard results
        prompting_df: DataFrame with direct prompting results
        flops_data: Dictionary with FLOPs information
        output_path: Path to save the output CSV

    Returns:
        DataFrame with the efficiency comparison table
    """
    # Filter for Mean aggregation only for MI methods
    lpm_mean = lpm_df[lpm_df["aggregation"] == "mean"].copy()
    probe_mean = probe_df[probe_df["aggregation"] == "mean"].copy()

    # Create mapping from method names to FLOPs
    flops_map = {}

    # Add baseline LLM monitors
    for method in flops_data["baseline_llm_monitors"]["methods"]:
        flops_map[method["name"]] = method["flops"]

    # Add MI methods
    for method in flops_data["mechanistic_interpretability_methods"]["methods"]:
        flops_map[method["name"]] = method["flops"]

    # Prepare results list
    results = []

    # Helper function to average F1 across datasets
    def get_avg_f1(df, model_filter, **kwargs):
        """Get average F1 across plmix and wgmix datasets."""
        filtered = df[df["model_short"] == model_filter]
        for key, value in kwargs.items():
            filtered = filtered[filtered[key] == value]

        if len(filtered) == 0:
            return None

        return filtered["f1"].mean()

    # ========================================================================
    # Mechanistic Interpretability Methods
    # ========================================================================

    # LPM(Bielik-1.5B) Mahalanobis
    f1 = get_avg_f1(lpm_mean, "bielik_1_5b", metric="mahalanobis")
    if f1 is not None:
        results.append(
            {
                "Method": "LPM(Bielik-1.5B) Mahalanobis",
                "FLOPs": flops_map.get("lpm_bielik-1.5B_mahalanobis", 0),
                "F1": f1,
                "Category": "MI Method",
            }
        )

    # LPM(Bielik-1.5B) Euclidean
    f1 = get_avg_f1(lpm_mean, "bielik_1_5b", metric="euclidean")
    if f1 is not None:
        results.append(
            {
                "Method": "LPM(Bielik-1.5B) Euclidean",
                "FLOPs": flops_map.get("lpm_bielik-1.5B_euclidean", 0),
                "F1": f1,
                "Category": "MI Method",
            }
        )

    # LPM(Llama-3B) Mahalanobis
    f1 = get_avg_f1(lpm_mean, "llama_3b", metric="mahalanobis")
    if f1 is not None:
        results.append(
            {
                "Method": "LPM(Llama-3B) Mahalanobis",
                "FLOPs": flops_map.get("lpm_llama-3B_mahalanobis", 0),
                "F1": f1,
                "Category": "MI Method",
            }
        )

    # LPM(Llama-3B) Euclidean
    f1 = get_avg_f1(lpm_mean, "llama_3b", metric="euclidean")
    if f1 is not None:
        results.append(
            {
                "Method": "LPM(Llama-3B) Euclidean",
                "FLOPs": flops_map.get("lpm_llama-3B_euclidean", 0),
                "F1": f1,
                "Category": "MI Method",
            }
        )

    # Linear Probe(Bielik-1.5B)
    f1 = get_avg_f1(probe_mean, "bielik_1_5b")
    if f1 is not None:
        results.append(
            {
                "Method": "Linear Probe(Bielik-1.5B)",
                "FLOPs": flops_map.get("linear_probe_bielik-1.5B", 0),
                "F1": f1,
                "Category": "MI Method",
            }
        )

    # Linear Probe(Llama-3B)
    f1 = get_avg_f1(probe_mean, "llama_3b")
    if f1 is not None:
        results.append(
            {
                "Method": "Linear Probe(Llama-3B)",
                "FLOPs": flops_map.get("linear_probe_llama-3B", 0),
                "F1": f1,
                "Category": "MI Method",
            }
        )

    # ========================================================================
    # Baseline Guard Models
    # ========================================================================

    # Llama-Guard-3-1B
    f1_avg = baseline_df[baseline_df["model_short"] == "llamaguard"]["f1"].mean()
    if pd.notna(f1_avg):
        results.append(
            {
                "Method": "Llama-Guard-3-1B",
                "FLOPs": flops_map.get("Llama-Guard-3-1B", 0),
                "F1": f1_avg,
                "Category": "Baseline Guard",
            }
        )

    # Bielik-Guard-0.1B
    f1_avg = baseline_df[baseline_df["model_short"] == "bielik"]["f1"].mean()
    if pd.notna(f1_avg):
        results.append(
            {
                "Method": "Bielik-Guard-0.1B",
                "FLOPs": flops_map.get("Bielik-Guard-0.1B", 0),
                "F1": f1_avg,
                "Category": "Baseline Guard",
            }
        )

    # ========================================================================
    # Direct Prompting (Averaged across prompts)
    # ========================================================================

    # Bielik-4.5B-Prompted (average across all prompts and datasets)
    bielik_prompting = prompting_df[prompting_df["model_short"] == "bielik-4_5b-v3_0-instruct"]
    if len(bielik_prompting) > 0:
        f1_avg = bielik_prompting["f1"].mean()
        results.append(
            {
                "Method": "Bielik-4.5B-Prompted",
                "FLOPs": flops_map.get("Bielik-4.5B-v3-Instruct-Prompted", 0),
                "F1": f1_avg,
                "Category": "Direct Prompting",
            }
        )

    # Llama-3.2-3B-Prompted (average across all prompts and datasets)
    llama_prompting = prompting_df[prompting_df["model_short"] == "llama-3_2-3b-instruct"]
    if len(llama_prompting) > 0:
        f1_avg = llama_prompting["f1"].mean()
        results.append(
            {
                "Method": "Llama-3.2-3B-Prompted",
                "FLOPs": flops_map.get("Llama-3.2-3B-Instruct-Prompted", 0),
                "F1": f1_avg,
                "Category": "Direct Prompting",
            }
        )

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by FLOPs (ascending)
    df = df.sort_values("FLOPs")

    # Format FLOPs in scientific notation
    df["FLOPs_formatted"] = df["FLOPs"].apply(lambda x: f"{x:.2e}")

    # Format F1 as percentage with 2 decimals
    df["F1_formatted"] = df["F1"].apply(lambda x: f"{x * 100:.2f}%")

    # Reorder columns for better readability
    output_df = df[["Method", "FLOPs_formatted", "F1_formatted", "Category"]].copy()
    output_df.columns = ["Method", "FLOPs", "F1", "Category"]

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"✅ Saved efficiency comparison table to: {output_path}")

    return output_df


def create_latex_efficiency_table(
    df: pd.DataFrame,
    output_path: Path,
    caption: str = "Efficiency Comparison: FLOPs vs F1 Score",
    label: str = "tab:efficiency_comparison",
) -> None:
    """Create LaTeX version of the efficiency comparison table.

    Args:
        df: DataFrame with efficiency comparison data
        output_path: Path to save the LaTeX file
        caption: Table caption
        label: LaTeX label for referencing
    """
    # Create LaTeX table
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append("\\begin{tabular}{llll}")
    latex.append("\\toprule")
    latex.append("Method & FLOPs & F1 & Category \\\\")
    latex.append("\\midrule")

    for _, row in df.iterrows():
        method = row["Method"].replace("_", "\\_")
        flops = row["FLOPs"]
        f1 = row["F1"]
        category = row["Category"]
        latex.append(f"{method} & {flops} & {f1} & {category} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex))

    print(f"✅ Saved LaTeX efficiency table to: {output_path}")
