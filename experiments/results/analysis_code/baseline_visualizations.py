# ruff: noqa
"""Visualization functions for baseline results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_plotting_style():
    """Set up matplotlib and seaborn for thesis-ready plots.

    Following visualization rules from experiments/.llm_context/experiments/visualization_rules.md
    """
    # Seaborn setup FIRST (so we can override its font settings)
    sns.set_context("paper", font_scale=1.0)
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # Use serif fonts (professional appearance without LaTeX dependency)
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


def plot_baseline_bar_chart(
    baseline_df: pd.DataFrame,
    prompting_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create bar chart showing baseline results with prompt variation whiskers.

    Args:
        baseline_df: DataFrame with guard baseline results
        prompting_df: DataFrame with direct prompting results
        output_path: Path to save the plot
    """
    setup_plotting_style()

    # Prepare data
    data = {"plmix_test": {}, "wgmix_test": {}}

    # Process guard models
    for model_short, model_name in [("llamaguard", "LlamaGuard"), ("bielik", "BielikGuard")]:
        model_data = baseline_df[baseline_df["model_short"] == model_short]
        for dataset in ["plmix_test", "wgmix_test"]:
            f1 = model_data[model_data["test_dataset"] == dataset]["f1"].values
            if len(f1) > 0:
                data[dataset][model_name] = {"mean": f1[0], "min": f1[0], "max": f1[0]}

    # Process prompted models
    for model_short, model_name in [
        ("llama-3_2-3b-instruct", "Llama-3B\n(Prompted)"),
        ("bielik-4_5b-v3_0-instruct", "Bielik-4.5B\n(Prompted)"),
    ]:
        model_data = prompting_df[prompting_df["model_short"] == model_short]
        for dataset in ["plmix_test", "wgmix_test"]:
            dataset_data = model_data[model_data["test_dataset"] == dataset]
            f1_values = dataset_data["f1"].values
            if len(f1_values) > 0:
                data[dataset][model_name] = {
                    "mean": f1_values.mean(),
                    "min": f1_values.min(),
                    "max": f1_values.max(),
                }

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    datasets = [("plmix_test", "PLMix"), ("wgmix_test", "WGMix")]
    colors = sns.color_palette("Set2", 4)

    for ax, (dataset_key, dataset_label) in zip(axes, datasets):
        dataset_data = data[dataset_key]
        models = list(dataset_data.keys())
        x_pos = np.arange(len(models))

        means = [dataset_data[m]["mean"] for m in models]
        mins = [dataset_data[m]["min"] for m in models]
        maxs = [dataset_data[m]["max"] for m in models]

        # Calculate error bars (distance from mean to min/max)
        errors = [[means[i] - mins[i], maxs[i] - means[i]] for i in range(len(means))]
        errors_array = np.array(errors).T

        # Assign colors: guards get different colors, prompted models get different colors
        bar_colors = [colors[0], colors[1], colors[2], colors[3]]

        # Plot bars with error bars
        bars = ax.bar(
            x_pos,
            means,
            color=bar_colors,
            yerr=errors_array,
            capsize=4,
            error_kw={"linewidth": 1.5, "ecolor": "black", "alpha": 0.7},
        )

        # Add value labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel("F1 Score" if dataset_key == "plmix_test" else "", fontsize=10)
        ax.set_title(dataset_label, fontsize=11, fontweight="bold", pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylim(0.0, 1.1)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Baseline Results: Guards vs Prompted LLMs", fontsize=12, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved baseline bar chart to: {output_path}")
    plt.close()


def plot_baseline_heatmap(
    baseline_df: pd.DataFrame,
    prompting_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create heatmap showing F1 scores across models, prompts, and datasets.

    Args:
        baseline_df: DataFrame with guard baseline results
        prompting_df: DataFrame with direct prompting results
        output_path: Path to save the plot
    """
    setup_plotting_style()

    # Prepare data for heatmap
    rows = []

    # Guard models
    for model_short, model_name in [("llamaguard", "LlamaGuard-1B"), ("bielik", "BielikGuard-0.1B")]:
        model_data = baseline_df[baseline_df["model_short"] == model_short]
        plmix = model_data[model_data["test_dataset"] == "plmix_test"]["f1"].values
        wgmix = model_data[model_data["test_dataset"] == "wgmix_test"]["f1"].values

        rows.append(
            {
                "Model": model_name,
                "Variant": "N/A",
                "PLMix": plmix[0] if len(plmix) > 0 else np.nan,
                "WGMix": wgmix[0] if len(wgmix) > 0 else np.nan,
            }
        )

    # Prompted models
    for model_short, model_name in [
        ("llama-3_2-3b-instruct", "Llama-3B"),
        ("bielik-4_5b-v3_0-instruct", "Bielik-4.5B"),
    ]:
        model_data = prompting_df[prompting_df["model_short"] == model_short]

        # Individual prompts
        for prompt_id in range(4):
            prompt_data = model_data[model_data["prompt_id"] == prompt_id]
            plmix = prompt_data[prompt_data["test_dataset"] == "plmix_test"]["f1"].values
            wgmix = prompt_data[prompt_data["test_dataset"] == "wgmix_test"]["f1"].values

            rows.append(
                {
                    "Model": model_name,
                    "Variant": f"Prompt {prompt_id}",
                    "PLMix": plmix[0] if len(plmix) > 0 else np.nan,
                    "WGMix": wgmix[0] if len(wgmix) > 0 else np.nan,
                }
            )

        # Average
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

        rows.append(
            {
                "Model": model_name,
                "Variant": "Average",
                "PLMix": np.mean(plmix_values) if plmix_values else np.nan,
                "WGMix": np.mean(wgmix_values) if wgmix_values else np.nan,
            }
        )

    df = pd.DataFrame(rows)

    # Create row labels
    df["Row_Label"] = df.apply(
        lambda row: row["Model"] if row["Variant"] == "N/A" else f"{row['Model']} ({row['Variant']})", axis=1
    )

    # Pivot for heatmap
    heatmap_data = df[["Row_Label", "PLMix", "WGMix"]].set_index("Row_Label")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(6, 9))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        cbar_kws={"label": "F1 Score"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    ax.set_xlabel("Dataset", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)
    ax.set_title("Baseline F1 Scores Heatmap", fontsize=11, fontweight="bold", pad=15)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved baseline heatmap to: {output_path}")
    plt.close()


def plot_prompt_stability(
    prompting_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create violin plot showing prompt stability for prompted models.

    Args:
        prompting_df: DataFrame with direct prompting results
        output_path: Path to save the plot
    """
    setup_plotting_style()

    # Prepare data
    plot_data = []

    for model_short, model_name in [
        ("llama-3_2-3b-instruct", "Llama-3B"),
        ("bielik-4_5b-v3_0-instruct", "Bielik-4.5B"),
    ]:
        model_data = prompting_df[prompting_df["model_short"] == model_short]

        for dataset in ["plmix_test", "wgmix_test"]:
            dataset_label = "PLMix" if dataset == "plmix_test" else "WGMix"
            dataset_data = model_data[model_data["test_dataset"] == dataset]

            for _, row in dataset_data.iterrows():
                plot_data.append(
                    {
                        "Model": model_name,
                        "Dataset": dataset_label,
                        "Prompt": f"Prompt {row['prompt_id']}",
                        "F1": row["f1"],
                    }
                )

    df = pd.DataFrame(plot_data)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    datasets = ["PLMix", "WGMix"]
    colors = sns.color_palette("Set2", 2)

    for ax, dataset in zip(axes, datasets):
        dataset_data = df[df["Dataset"] == dataset]

        # Create violin plot
        sns.violinplot(
            data=dataset_data,
            x="Model",
            y="F1",
            palette=colors,
            ax=ax,
            inner="box",
        )

        # Overlay individual prompt points
        sns.stripplot(
            data=dataset_data,
            x="Model",
            y="F1",
            color="black",
            alpha=0.6,
            size=5,
            ax=ax,
        )

        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel("F1 Score" if dataset == "PLMix" else "", fontsize=10)
        ax.set_title(f"{dataset}: Prompt Stability", fontsize=11, fontweight="bold", pad=15)
        ax.set_ylim(0.0, 1.1)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Prompt Variation Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved prompt stability plot to: {output_path}")
    plt.close()
