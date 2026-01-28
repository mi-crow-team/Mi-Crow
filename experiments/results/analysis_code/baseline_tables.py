# ruff: noqa
"""Create baseline results tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def create_baseline_results_table(
    baseline_df: pd.DataFrame,
    prompting_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Create comprehensive baseline results table.

    Table structure:
    - Rows: LlamaGuard, BielikGuard, Llama-3B (prompts 0-3 + Avg), Bielik-4.5B (prompts 0-3 + Avg)
    - Columns: Model, PLMix, WGMix, Average
    - Metric: F1 score

    Args:
        baseline_df: DataFrame with guard baseline results
        prompting_df: DataFrame with direct prompting results
        output_path: Path to save the CSV table

    Returns:
        DataFrame with the baseline results table
    """
    results = []

    # ========================================================================
    # Guard Models
    # ========================================================================

    # LlamaGuard
    llama_guard = baseline_df[baseline_df["model_short"] == "llamaguard"]
    if len(llama_guard) > 0:
        plmix = llama_guard[llama_guard["test_dataset"] == "plmix_test"]["f1"].values
        wgmix = llama_guard[llama_guard["test_dataset"] == "wgmix_test"]["f1"].values

        plmix_f1 = plmix[0] if len(plmix) > 0 else None
        wgmix_f1 = wgmix[0] if len(wgmix) > 0 else None
        avg_f1 = (plmix_f1 + wgmix_f1) / 2 if plmix_f1 and wgmix_f1 else None

        results.append(
            {
                "Model": "LlamaGuard-1B",
                "PLMix": plmix_f1,
                "WGMix": wgmix_f1,
                "Average": avg_f1,
            }
        )

    # BielikGuard
    bielik_guard = baseline_df[baseline_df["model_short"] == "bielik"]
    if len(bielik_guard) > 0:
        plmix = bielik_guard[bielik_guard["test_dataset"] == "plmix_test"]["f1"].values
        wgmix = bielik_guard[bielik_guard["test_dataset"] == "wgmix_test"]["f1"].values

        plmix_f1 = plmix[0] if len(plmix) > 0 else None
        wgmix_f1 = wgmix[0] if len(wgmix) > 0 else None
        avg_f1 = (plmix_f1 + wgmix_f1) / 2 if plmix_f1 and wgmix_f1 else None

        results.append(
            {
                "Model": "BielikGuard-0.1B",
                "PLMix": plmix_f1,
                "WGMix": wgmix_f1,
                "Average": avg_f1,
            }
        )

    # ========================================================================
    # Prompted Models
    # ========================================================================

    for model_short, model_name in [
        ("llama-3_2-3b-instruct", "Llama-3B"),
        ("bielik-4_5b-v3_0-instruct", "Bielik-4.5B"),
    ]:
        model_data = prompting_df[prompting_df["model_short"] == model_short]

        if len(model_data) == 0:
            continue

        # Results for each prompt (0-3)
        for prompt_id in range(4):
            prompt_data = model_data[model_data["prompt_id"] == prompt_id]

            plmix = prompt_data[prompt_data["test_dataset"] == "plmix_test"]["f1"].values
            wgmix = prompt_data[prompt_data["test_dataset"] == "wgmix_test"]["f1"].values

            plmix_f1 = plmix[0] if len(plmix) > 0 else None
            wgmix_f1 = wgmix[0] if len(wgmix) > 0 else None
            avg_f1 = (plmix_f1 + wgmix_f1) / 2 if plmix_f1 and wgmix_f1 else None

            results.append(
                {
                    "Model": f"{model_name} (Prompt {prompt_id})",
                    "PLMix": plmix_f1,
                    "WGMix": wgmix_f1,
                    "Average": avg_f1,
                }
            )

        # Average across all prompts
        plmix_values = []
        wgmix_values = []

        for prompt_id in range(4):
            prompt_data = model_data[model_data["prompt_id"] == prompt_id]
            plmix = prompt_data[prompt_data["test_dataset"] == "plmix_test"]["f1"].values
            wgmix = prompt_data[prompt_data["test_dataset"] == "wgmix_test"]["f1"].values

            if len(plmix) > 0:
                plmix_values.append(plmix[0])
            if len(wgmix) > 0:
                wgmix_values.append(wgmix[0])

        plmix_avg = sum(plmix_values) / len(plmix_values) if plmix_values else None
        wgmix_avg = sum(wgmix_values) / len(wgmix_values) if wgmix_values else None
        overall_avg = (plmix_avg + wgmix_avg) / 2 if plmix_avg and wgmix_avg else None

        results.append(
            {
                "Model": f"{model_name} (Avg)",
                "PLMix": plmix_avg,
                "WGMix": wgmix_avg,
                "Average": overall_avg,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(results)

    # Format F1 scores as percentages
    for col in ["PLMix", "WGMix", "Average"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "N/A")

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved baseline results table to: {output_path}")

    return df


def create_latex_baseline_table(
    df: pd.DataFrame,
    output_path: Path,
    caption: str = "Baseline Results: F1 Scores across Models and Datasets",
    label: str = "tab:baseline_results",
) -> None:
    """Create LaTeX version of baseline results table.

    Args:
        df: DataFrame with baseline results
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
    latex.append("Model & PLMix & WGMix & Average \\\\")
    latex.append("\\midrule")

    for _, row in df.iterrows():
        model = row["Model"].replace("_", "\\_")
        plmix = row["PLMix"]
        wgmix = row["WGMix"]
        average = row["Average"]

        # Add separator before prompted model groups
        if "(Prompt 0)" in model:
            latex.append("\\midrule")

        latex.append(f"{model} & {plmix} & {wgmix} & {average} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex))

    print(f"✅ Saved LaTeX baseline table to: {output_path}")
