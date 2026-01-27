"""Table generation functions for thesis results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def create_lpm_results_table(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Create comprehensive LPM results table.

    Columns: Model, Aggregation, Metric, PLMix (F1, Precision, Recall, Accuracy),
             WGMix (F1, Precision, Recall, Accuracy)

    Args:
        df: DataFrame with LPM results
        output_path: Path to save CSV file

    Returns:
        Formatted DataFrame
    """
    # Filter to LPM only
    lpm_df = df[df["method"] == "LPM"].copy()

    # Pivot data to have one row per (model, aggregation, metric)
    table_rows = []

    models = sorted(lpm_df["model"].unique())
    aggregations = ["mean", "last_token", "last_token_prefix"]
    metrics = ["euclidean", "mahalanobis"]

    for model in models:
        for agg in aggregations:
            for metric in metrics:
                row = {
                    "Model": model,
                    "Aggregation": agg.replace("_", " ").title(),
                    "Metric": metric.title(),
                }

                # PLMix metrics
                plmix_data = lpm_df[
                    (lpm_df["model"] == model)
                    & (lpm_df["aggregation"] == agg)
                    & (lpm_df["metric"] == metric)
                    & (lpm_df["test_dataset"] == "plmix_test")
                ]

                if len(plmix_data) > 0:
                    row["PLMix F1"] = plmix_data["f1"].values[0]
                    row["PLMix Precision"] = plmix_data["precision"].values[0]
                    row["PLMix Recall"] = plmix_data["recall"].values[0]
                    row["PLMix Accuracy"] = plmix_data["accuracy"].values[0]
                else:
                    row["PLMix F1"] = None
                    row["PLMix Precision"] = None
                    row["PLMix Recall"] = None
                    row["PLMix Accuracy"] = None

                # WGMix metrics
                wgmix_data = lpm_df[
                    (lpm_df["model"] == model)
                    & (lpm_df["aggregation"] == agg)
                    & (lpm_df["metric"] == metric)
                    & (lpm_df["test_dataset"] == "wgmix_test")
                ]

                if len(wgmix_data) > 0:
                    row["WGMix F1"] = wgmix_data["f1"].values[0]
                    row["WGMix Precision"] = wgmix_data["precision"].values[0]
                    row["WGMix Recall"] = wgmix_data["recall"].values[0]
                    row["WGMix Accuracy"] = wgmix_data["accuracy"].values[0]
                else:
                    row["WGMix F1"] = None
                    row["WGMix Precision"] = None
                    row["WGMix Recall"] = None
                    row["WGMix Accuracy"] = None

                table_rows.append(row)

    table_df = pd.DataFrame(table_rows)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"✅ Saved LPM results table: {output_path}")

    return table_df


def create_probe_results_table(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Create comprehensive Linear Probe results table.

    Columns: Model, Aggregation, PLMix (F1, Precision, Recall, Accuracy),
             WGMix (F1, Precision, Recall, Accuracy)

    Args:
        df: DataFrame with Linear Probe results
        output_path: Path to save CSV file

    Returns:
        Formatted DataFrame
    """
    # Filter to Linear Probe only
    probe_df = df[df["method"] == "Linear Probe"].copy()

    # Pivot data to have one row per (model, aggregation)
    table_rows = []

    models = sorted(probe_df["model"].unique())
    aggregations = ["mean", "last_token", "last_token_prefix"]

    for model in models:
        for agg in aggregations:
            row = {
                "Model": model,
                "Aggregation": agg.replace("_", " ").title(),
            }

            # PLMix metrics
            plmix_data = probe_df[
                (probe_df["model"] == model)
                & (probe_df["aggregation"] == agg)
                & (probe_df["test_dataset"] == "plmix_test")
            ]

            if len(plmix_data) > 0:
                row["PLMix F1"] = plmix_data["f1"].values[0]
                row["PLMix Precision"] = plmix_data["precision"].values[0]
                row["PLMix Recall"] = plmix_data["recall"].values[0]
                row["PLMix Accuracy"] = plmix_data["accuracy"].values[0]
            else:
                row["PLMix F1"] = None
                row["PLMix Precision"] = None
                row["PLMix Recall"] = None
                row["PLMix Accuracy"] = None

            # WGMix metrics
            wgmix_data = probe_df[
                (probe_df["model"] == model)
                & (probe_df["aggregation"] == agg)
                & (probe_df["test_dataset"] == "wgmix_test")
            ]

            if len(wgmix_data) > 0:
                row["WGMix F1"] = wgmix_data["f1"].values[0]
                row["WGMix Precision"] = wgmix_data["precision"].values[0]
                row["WGMix Recall"] = wgmix_data["recall"].values[0]
                row["WGMix Accuracy"] = wgmix_data["accuracy"].values[0]
            else:
                row["WGMix F1"] = None
                row["WGMix Precision"] = None
                row["WGMix Recall"] = None
                row["WGMix Accuracy"] = None

            table_rows.append(row)

    table_df = pd.DataFrame(table_rows)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"✅ Saved Linear Probe results table: {output_path}")

    return table_df


def create_best_results_summary(
    lpm_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Create summary table showing best results for each method and dataset.

    Args:
        lpm_df: DataFrame with LPM results
        probe_df: DataFrame with Linear Probe results
        output_path: Path to save CSV file

    Returns:
        Summary DataFrame
    """
    summary_rows = []

    for dataset in ["plmix_test", "wgmix_test"]:
        dataset_label = dataset.replace("_test", "").upper()

        # Best LPM
        lpm_subset = lpm_df[lpm_df["test_dataset"] == dataset]
        if len(lpm_subset) > 0:
            best_lpm = lpm_subset.loc[lpm_subset["f1"].idxmax()]
            summary_rows.append(
                {
                    "Dataset": dataset_label,
                    "Method": "LPM",
                    "F1": best_lpm["f1"],
                    "Precision": best_lpm["precision"],
                    "Recall": best_lpm["recall"],
                    "Accuracy": best_lpm["accuracy"],
                    "Model": best_lpm["model"],
                    "Aggregation": best_lpm["aggregation"],
                    "Config": f"{best_lpm['metric']}",
                }
            )

        # Best Linear Probe
        probe_subset = probe_df[probe_df["test_dataset"] == dataset]
        if len(probe_subset) > 0:
            best_probe = probe_subset.loc[probe_subset["f1"].idxmax()]
            summary_rows.append(
                {
                    "Dataset": dataset_label,
                    "Method": "Linear Probe",
                    "F1": best_probe["f1"],
                    "Precision": best_probe["precision"],
                    "Recall": best_probe["recall"],
                    "Accuracy": best_probe["accuracy"],
                    "Model": best_probe["model"],
                    "Aggregation": best_probe["aggregation"],
                    "Config": "-",
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"✅ Saved best results summary: {output_path}")

    return summary_df


def create_latex_table(df: pd.DataFrame, output_path: Path, caption: str, label: str):
    """Create LaTeX-formatted table from DataFrame.

    Args:
        df: DataFrame to convert
        output_path: Path to save .tex file
        caption: Table caption
        label: LaTeX label for referencing
    """
    latex_str = df.to_latex(
        index=False,
        float_format="%.4f",
        caption=caption,
        label=label,
        escape=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str, encoding="utf-8")
    print(f"✅ Saved LaTeX table: {output_path}")
