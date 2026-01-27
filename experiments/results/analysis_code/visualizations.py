# ruff: noqa
"""Visualization functions for thesis-ready plots."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Configure matplotlib for LaTeX-style output
def setup_plotting_style(use_latex: bool = False):
    """Set up matplotlib and seaborn for thesis-ready plots.

    Following visualization rules from experiments/.llm_context/experiments/visualization_rules.md

    Args:
        use_latex: Whether to use LaTeX for text rendering (requires LaTeX installation)
                  Default: False (not required for thesis-quality plots)
    """
    # Seaborn setup FIRST (so we can override its font settings)
    sns.set_context("paper", font_scale=1.0)
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # Use serif fonts (professional appearance without LaTeX dependency)
    if use_latex and shutil.which("latex") is not None:
        try:
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                }
            )
            print("✅ Using LaTeX for text rendering")
        except Exception as e:
            print(f"⚠️  Could not enable LaTeX: {e}")
            # Fallback to explicit serif fonts
            plt.rcParams.update(
                {
                    "font.family": "serif",
                    "font.serif": [
                        "DejaVu Serif",
                        "Liberation Serif",
                        "Bitstream Vera Serif",
                        "Times New Roman",
                        "serif",
                    ],
                    "font.size": 10,
                    "axes.labelsize": 10,
                    "axes.titlesize": 11,
                    "xtick.labelsize": 9,
                    "ytick.labelsize": 9,
                    "legend.fontsize": 9,
                }
            )
    else:
        # Default: explicit serif fonts without LaTeX (thesis-appropriate)
        # List multiple serif fonts as fallbacks
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["DejaVu Serif", "Liberation Serif", "Bitstream Vera Serif", "Times New Roman", "serif"],
                "font.size": 10,
                "axes.labelsize": 10,
                "axes.titlesize": 11,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
            }
        )


def save_figure(fig: plt.Figure, output_path: Path, dpi: int = 300):
    """Save figure with high resolution.

    Args:
        fig: Matplotlib figure
        output_path: Path to save figure
        dpi: Dots per inch (default: 300 for print quality)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"✅ Saved figure: {output_path}")


def plot_lpm_metric_comparison(
    df: pd.DataFrame,
    output_path: Path,
    aggregation: str = "all",
    show_whiskers: bool = True,
) -> plt.Figure:
    """Figure 1: LPM Metric Comparison (Euclidean vs. Mahalanobis).

    Creates grouped bar chart comparing Euclidean vs. Mahalanobis distance
    metrics for LPM across models and datasets.

    Args:
        df: DataFrame with LPM results
        output_path: Path to save figure
        aggregation: Which aggregation method to use (default: 'all')
                    If 'all', uses mean across all aggregation methods with whiskers
        show_whiskers: Whether to show min/max whiskers across aggregations

    Returns:
        Matplotlib figure
    """
    setup_plotting_style()

    # Filter to LPM only
    lpm_df = df[df["method"] == "LPM"].copy()

    # Prepare data based on aggregation parameter
    if aggregation == "all":
        # Use mean across all aggregation methods
        grouped = lpm_df.groupby(["model", "test_dataset", "metric"])["f1"].agg(["mean", "min", "max"]).reset_index()
        plot_df = grouped.rename(columns={"mean": "f1"})
        whiskers_df = grouped if show_whiskers else None
    else:
        # Filter to specific aggregation
        plot_df = lpm_df[lpm_df["aggregation"] == aggregation].copy()
        whiskers_df = None

    # Create figure with two subplots (PLMix and WGMix)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    datasets = [("plmix_test", "PLMix"), ("wgmix_test", "WGMix")]
    models = ["Bielik-1.5B", "Bielik-4.5B", "Llama-3.2-3B"]
    metrics = ["euclidean", "mahalanobis"]

    metric_labels = {"euclidean": "Euclidean", "mahalanobis": "Mahalanobis"}
    colors = sns.color_palette("Set2", n_colors=2)

    for ax, (dataset, dataset_label) in zip(axes, datasets):
        x_pos = np.arange(len(models))
        width = 0.35

        for i, metric in enumerate(metrics):
            metric_data = plot_df[(plot_df["test_dataset"] == dataset) & (plot_df["metric"] == metric)]

            # Ensure models are in correct order
            f1_values = []
            f1_errors = []  # For whiskers: [[lower_errors], [upper_errors]]

            for model in models:
                model_data = metric_data[metric_data["model"] == model]
                if len(model_data) > 0:
                    f1_mean = model_data["f1"].values[0]
                    f1_values.append(f1_mean)

                    # Calculate whisker errors if data available
                    if whiskers_df is not None:
                        whisker_row = whiskers_df[
                            (whiskers_df["test_dataset"] == dataset)
                            & (whiskers_df["metric"] == metric)
                            & (whiskers_df["model"] == model)
                        ]
                        if len(whisker_row) > 0:
                            f1_min = whisker_row["min"].values[0]
                            f1_max = whisker_row["max"].values[0]
                            f1_errors.append([f1_mean - f1_min, f1_max - f1_mean])
                        else:
                            f1_errors.append([0.0, 0.0])
                    else:
                        f1_errors.append([0.0, 0.0])
                else:
                    f1_values.append(0.0)
                    f1_errors.append([0.0, 0.0])

            # Convert errors to format for matplotlib: [[lower], [upper]]
            errors_array = np.array(f1_errors).T if f1_errors else None

            # Plot bars with optional error bars
            if errors_array is not None and np.any(errors_array > 0):
                bars = ax.bar(
                    x_pos + i * width - width / 2,
                    f1_values,
                    width,
                    label=metric_labels[metric],
                    color=colors[i],
                    yerr=errors_array,
                    capsize=3,
                    error_kw={"linewidth": 1.5, "ecolor": "black", "alpha": 0.7},
                )
            else:
                bars = ax.bar(
                    x_pos + i * width - width / 2,
                    f1_values,
                    width,
                    label=metric_labels[metric],
                    color=colors[i],
                )

            # Add value labels with leader lines when whiskers present
            for j, (bar, f1) in enumerate(zip(bars, f1_values)):
                height = bar.get_height()
                if height > 0:
                    bar_x = bar.get_x() + bar.get_width() / 2.0

                    # If whiskers present, use leader line from bar top to offset label
                    if errors_array is not None and np.any(errors_array > 0):
                        upper_error = errors_array[1][j] if j < len(errors_array[1]) else 0.0
                        whisker_top = height + upper_error

                        # Position label to the left with vertical offset
                        label_x = bar_x + 0.01  # Offset to the left
                        label_y = whisker_top + 0.06  # Slightly above whisker top

                        # Draw leader line with arrow from label to bar mean
                        ax.annotate(
                            f"{f1:.3f}",
                            xy=(bar_x, height),  # Arrow points at bar top (mean)
                            xytext=(label_x, label_y),  # Label position
                            ha="right",
                            va="center",
                            fontsize=8,
                            arrowprops=dict(
                                arrowstyle="-|>",
                                color="gray",
                                linewidth=0.7,
                                alpha=0.6,
                                shrinkA=0,
                                shrinkB=0,
                            ),
                        )
                    else:
                        # No whiskers - simple text above bar
                        ax.text(
                            bar_x,
                            height,
                            f"{f1:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel("F1 Score" if dataset == "plmix_test" else "", fontsize=10)
        ax.set_title(dataset_label, fontsize=11, fontweight="bold", pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models)  # No rotation - fits without it
        ax.set_ylim(0.0, 1.1)  # Full scale with margin for labels
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    title_suffix = "Mean Across Aggregations" if aggregation == "all" else f"{aggregation.replace('_', ' ').title()}"
    fig.suptitle(
        f"LPM: Euclidean vs. Mahalanobis Distance ({title_suffix})",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    save_figure(fig, output_path)

    return fig


def plot_aggregation_impact(
    df: pd.DataFrame,
    output_dir: Path,
    method: str = "LPM",
    metric: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Figure]:
    """Figure 2: Impact of Aggregation Methods.

    Creates grouped bar charts showing impact of aggregation methods
    (mean, last_token, last_token_prefix) across models and datasets.

    Args:
        df: DataFrame with results
        output_dir: Directory to save figures
        method: "LPM" or "Linear Probe"
        metric: For LPM, which distance metric to use (default: "mahalanobis")

    Returns:
        Tuple of (plmix_figure, wgmix_figure)
    """
    setup_plotting_style()

    # Filter data
    method_df = df[df["method"] == method].copy()

    if method == "LPM":
        metric = metric or "mahalanobis"
        method_df = method_df[method_df["metric"] == metric]
        title_suffix = f"({metric.title()})"
    else:
        title_suffix = ""

    # Create separate figures for each dataset
    datasets = [("plmix_test", "PLMix"), ("wgmix_test", "WGMix")]
    models = ["Bielik-1.5B", "Bielik-4.5B", "Llama-3.2-3B"]
    aggregations = ["mean", "last_token", "last_token_prefix"]
    aggregation_labels = {
        "mean": "Mean",
        "last_token": "Last Token",
        "last_token_prefix": "Last Token + Prefix",
    }

    colors = sns.color_palette("Set2", n_colors=3)
    figures = []

    for dataset, dataset_label in datasets:
        fig, ax = plt.subplots(figsize=(8, 5))

        x_pos = np.arange(len(models))
        width = 0.25

        for i, agg in enumerate(aggregations):
            agg_data = method_df[(method_df["test_dataset"] == dataset) & (method_df["aggregation"] == agg)]

            # Ensure models are in correct order
            f1_values = []
            for model in models:
                model_data = agg_data[agg_data["model"] == model]
                if len(model_data) > 0:
                    f1_values.append(model_data["f1"].values[0])
                else:
                    f1_values.append(0.0)

            bars = ax.bar(
                x_pos + i * width - width,
                f1_values,
                width,
                label=aggregation_labels[agg],
                color=colors[i],
            )

            # Add value labels
            for bar, f1 in zip(bars, f1_values):
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{f1:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel("F1 Score", fontsize=10)
        ax.set_title(
            f"{method} - Aggregation Methods: {dataset_label} {title_suffix}",
            fontsize=11,
            fontweight="bold",
            pad=15,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models)  # No rotation needed
        ax.set_ylim(0.0, 1.1)  # Full scale with margin
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        # Save figure
        method_slug = method.lower().replace(" ", "_")
        dataset_slug = dataset.replace("_test", "")
        output_path = output_dir / f"aggregation_impact_{method_slug}_{dataset_slug}.png"
        save_figure(fig, output_path)

        figures.append(fig)

    return tuple(figures)


def plot_method_comparison(
    lpm_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    output_path: Path,
) -> plt.Figure:
    """Figure 3: Linear Probe vs. LPM (Method Battle).

    Compares best performing LPM vs. Linear Probe for each dataset.
    Includes horizontal lines showing mean performance across all experiments
    to visualize stability (gap between best and mean).

    Args:
        lpm_df: DataFrame with LPM results
        probe_df: DataFrame with Linear Probe results
        output_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    setup_plotting_style()

    # Calculate overall mean F1 for each method (stability indicator)
    lpm_mean_f1 = lpm_df["f1"].mean()
    probe_mean_f1 = probe_df["f1"].mean()

    # Find best F1 for each method and dataset
    results = []

    for dataset in ["plmix_test", "wgmix_test"]:
        # Best LPM
        lpm_subset = lpm_df[lpm_df["test_dataset"] == dataset]
        if len(lpm_subset) > 0:
            best_lpm = lpm_subset.loc[lpm_subset["f1"].idxmax()]
            results.append(
                {
                    "dataset": dataset.replace("_test", "").upper(),
                    "method": "LPM",
                    "f1": best_lpm["f1"],
                    "config": f"{best_lpm['model']}, {best_lpm['aggregation']}, {best_lpm['metric']}",
                }
            )

        # Best Linear Probe
        probe_subset = probe_df[probe_df["test_dataset"] == dataset]
        if len(probe_subset) > 0:
            best_probe = probe_subset.loc[probe_subset["f1"].idxmax()]
            results.append(
                {
                    "dataset": dataset.replace("_test", "").upper(),
                    "method": "Linear Probe",
                    "f1": best_probe["f1"],
                    "config": f"{best_probe['model']}, {best_probe['aggregation']}",
                }
            )

    plot_df = pd.DataFrame(results)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))

    datasets = ["PLMIX", "WGMIX"]
    methods = ["LPM", "Linear Probe"]
    colors = sns.color_palette("Set2", n_colors=2)

    x_pos = np.arange(len(datasets))
    width = 0.35

    for i, method in enumerate(methods):
        method_data = plot_df[plot_df["method"] == method]

        f1_values = []
        configs = []
        for dataset in datasets:
            dataset_data = method_data[method_data["dataset"] == dataset]
            if len(dataset_data) > 0:
                f1_values.append(dataset_data["f1"].values[0])
                configs.append(dataset_data["config"].values[0])
            else:
                f1_values.append(0.0)
                configs.append("")

        bars = ax.bar(
            x_pos + i * width - width / 2,
            f1_values,
            width,
            label=method,
            color=colors[i],
        )

        # Add value labels
        for bar, f1 in zip(bars, f1_values):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{f1:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    # Add horizontal dashed lines for mean performance (stability indicator)
    ax.axhline(
        lpm_mean_f1,
        color=colors[0],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"LPM Mean (all configs): {lpm_mean_f1:.3f}",
    )
    ax.axhline(
        probe_mean_f1,
        color=colors[1],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Probe Mean (all configs): {probe_mean_f1:.3f}",
    )

    ax.set_xlabel("Dataset", fontsize=10)
    ax.set_ylabel("F1 Score (Best Configuration)", fontsize=10)
    ax.set_title("Method Comparison: LPM vs. Linear Probe", fontsize=11, fontweight="bold", pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets)  # No rotation needed for 2 labels
    ax.set_ylim(0.0, 1.1)  # Full scale with margin
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)

    return fig
