# ruff: noqa
"""Additional visualization functions for thesis.

New visualizations requested:
1. Consolidated aggregation impact
2. Detailed method comparison across all configs
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from visualizations import save_figure, setup_plotting_style


def plot_aggregation_impact_consolidated(
    lpm_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    output_path: Path,
) -> plt.Figure:
    """Consolidated Aggregation Impact Visualization.

    Shows impact of aggregation methods across all experiments in a compact format.
    Focus on probe/wgmix where differences are most significant.

    Args:
        lpm_df: DataFrame with LPM results
        probe_df: DataFrame with Linear Probe results
        output_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    setup_plotting_style()

    # Calculate aggregation variance for each method×dataset combination
    lpm_stats = []
    probe_stats = []

    for dataset in ["plmix_test", "wgmix_test"]:
        # LPM: Use mahalanobis metric (best performing)
        lpm_subset = lpm_df[(lpm_df["test_dataset"] == dataset) & (lpm_df["metric"] == "mahalanobis")]
        for model in lpm_subset["model"].unique():
            model_data = lpm_subset[lpm_subset["model"] == model]
            agg_f1s = []
            for agg in ["mean", "last_token", "last_token_prefix"]:
                agg_data = model_data[model_data["aggregation"] == agg]
                if len(agg_data) > 0:
                    agg_f1s.append(agg_data["f1"].values[0])

            if len(agg_f1s) == 3:
                lpm_stats.append(
                    {
                        "dataset": dataset.replace("_test", "").upper(),
                        "model": model,
                        "mean_f1": np.mean(agg_f1s),
                        "variance": np.var(agg_f1s),
                        "range": max(agg_f1s) - min(agg_f1s),
                    }
                )

        # Linear Probe
        probe_subset = probe_df[probe_df["test_dataset"] == dataset]
        for model in probe_subset["model"].unique():
            model_data = probe_subset[probe_subset["model"] == model]
            agg_f1s = []
            for agg in ["mean", "last_token", "last_token_prefix"]:
                agg_data = model_data[model_data["aggregation"] == agg]
                if len(agg_data) > 0:
                    agg_f1s.append(agg_data["f1"].values[0])

            if len(agg_f1s) == 3:
                probe_stats.append(
                    {
                        "dataset": dataset.replace("_test", "").upper(),
                        "model": model,
                        "mean_f1": np.mean(agg_f1s),
                        "variance": np.var(agg_f1s),
                        "range": max(agg_f1s) - min(agg_f1s),
                    }
                )

    lpm_stats_df = pd.DataFrame(lpm_stats)
    probe_stats_df = pd.DataFrame(probe_stats)

    # Create figure with 2 subplots: variance and actual performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Aggregation impact (variance/range)
    methods = ["LPM", "Linear Probe"]
    datasets = ["PLMIX", "WGMIX"]
    colors = sns.color_palette("Set2", n_colors=2)

    x_pos = np.arange(len(datasets))
    width = 0.35

    lpm_ranges = [lpm_stats_df[lpm_stats_df["dataset"] == ds]["range"].mean() for ds in datasets]
    probe_ranges = [probe_stats_df[probe_stats_df["dataset"] == ds]["range"].mean() for ds in datasets]

    ax1.bar(x_pos - width / 2, lpm_ranges, width, label="LPM", color=colors[0])
    ax1.bar(x_pos + width / 2, probe_ranges, width, label="Linear Probe", color=colors[1])

    # Add value labels
    for i, (lpm_r, probe_r) in enumerate(zip(lpm_ranges, probe_ranges)):
        ax1.text(x_pos[i] - width / 2, lpm_r, f"{lpm_r:.3f}", ha="center", va="bottom", fontsize=9)
        ax1.text(x_pos[i] + width / 2, probe_r, f"{probe_r:.3f}", ha="center", va="bottom", fontsize=9)

    ax1.set_xlabel("Dataset", fontsize=10)
    ax1.set_ylabel("Aggregation Impact (F1 Range)", fontsize=10)
    ax1.set_title("Sensitivity to Aggregation Method", fontsize=11, fontweight="bold", pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(datasets)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Right plot: Detailed probe/wgmix breakdown (where differences are largest)
    probe_wgmix = probe_df[probe_df["test_dataset"] == "wgmix_test"]
    aggregations = ["mean", "last_token", "last_token_prefix"]
    agg_labels = {"mean": "Mean", "last_token": "Last", "last_token_prefix": "Last+Prefix"}

    models = ["Bielik-1.5B", "Bielik-4.5B", "Llama-3.2-3B"]
    x_pos_detailed = np.arange(len(models))
    width_detailed = 0.25

    agg_colors = sns.color_palette("Set2", n_colors=3)

    for i, agg in enumerate(aggregations):
        agg_data = probe_wgmix[probe_wgmix["aggregation"] == agg]
        f1_values = []
        for model in models:
            model_data = agg_data[agg_data["model"] == model]
            if len(model_data) > 0:
                f1_values.append(model_data["f1"].values[0])
            else:
                f1_values.append(0.0)

        bars = ax2.bar(
            x_pos_detailed + i * width_detailed - width_detailed,
            f1_values,
            width_detailed,
            label=agg_labels[agg],
            color=agg_colors[i],
        )

        # Add value labels
        for bar, f1 in zip(bars, f1_values):
            if f1 > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    f1,
                    f"{f1:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax2.set_xlabel("Model", fontsize=10)
    ax2.set_ylabel("F1 Score", fontsize=10)
    ax2.set_title("Probe/WGMix: Largest Aggregation Effect", fontsize=11, fontweight="bold", pad=15)
    ax2.set_xticks(x_pos_detailed)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0.0, 1.1)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)

    return fig


def plot_method_comparison_detailed(
    lpm_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    output_path: Path,
) -> plt.Figure:
    """Detailed Method Comparison Across All Configurations.

    Shows LPM vs Probe performance for all model×dataset×aggregation combinations
    in a heatmap format to see patterns across configurations.

    Args:
        lpm_df: DataFrame with LPM results (mahalanobis only)
        probe_df: DataFrame with Linear Probe results
        output_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    setup_plotting_style()

    # Prepare data for heatmap: rows = config, cols = method
    configs = []
    lpm_scores = []
    probe_scores = []

    for dataset in ["plmix_test", "wgmix_test"]:
        for model in ["Bielik-1.5B", "Bielik-4.5B", "Llama-3.2-3B"]:
            for agg in ["mean", "last_token", "last_token_prefix"]:
                config_name = (
                    f"{dataset.replace('_test', '').upper()}/{model.split('-')[0]}/{agg.replace('_', ' ').title()}"
                )

                # LPM (mahalanobis)
                lpm_row = lpm_df[
                    (lpm_df["test_dataset"] == dataset)
                    & (lpm_df["model"] == model)
                    & (lpm_df["aggregation"] == agg)
                    & (lpm_df["metric"] == "mahalanobis")
                ]
                lpm_f1 = lpm_row["f1"].values[0] if len(lpm_row) > 0 else 0.0

                # Probe
                probe_row = probe_df[
                    (probe_df["test_dataset"] == dataset)
                    & (probe_df["model"] == model)
                    & (probe_df["aggregation"] == agg)
                ]
                probe_f1 = probe_row["f1"].values[0] if len(probe_row) > 0 else 0.0

                configs.append(config_name)
                lpm_scores.append(lpm_f1)
                probe_scores.append(probe_f1)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame({"Configuration": configs, "LPM": lpm_scores, "Linear Probe": probe_scores})

    # Create grouped bar chart (vertical)
    fig, ax = plt.subplots(figsize=(10, 12))

    x_pos = np.arange(len(configs))
    width = 0.35
    colors = sns.color_palette("Set2", n_colors=2)

    ax.barh(x_pos - width / 2, lpm_scores, width, label="LPM (Mahalanobis)", color=colors[0])
    ax.barh(x_pos + width / 2, probe_scores, width, label="Linear Probe", color=colors[1])

    ax.set_ylabel("Configuration", fontsize=10)
    ax.set_xlabel("F1 Score", fontsize=10)
    ax.set_title("LPM vs. Linear Probe: All Configurations", fontsize=11, fontweight="bold", pad=15)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(configs, fontsize=7)
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)

    return fig
