#!/usr/bin/env python3
# ruff: noqa
"""
Visualize efficiency comparison: FLOPs vs F1 Score.

This script creates a scatter plot showing the trade-off between
computational efficiency (FLOPs) and performance (F1 score) for
different safety classification methods.

Usage:
    python visualize_efficiency_comparison.py [--table-path PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import sys
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


def parse_flops(flops_str: str) -> float:
    """Parse FLOPs string in scientific notation to float.

    Args:
        flops_str: String like "2.30e+05"

    Returns:
        Float value
    """
    return float(flops_str)


def parse_f1(f1_str: str) -> float:
    """Parse F1 percentage string to float.

    Args:
        f1_str: String like "74.19%"

    Returns:
        Float value (e.g., 74.19)
    """
    return float(f1_str.rstrip("%"))


def get_method_type(method: str) -> str:
    """Determine the method type from the method name.

    Args:
        method: Method name

    Returns:
        One of: 'linear_probe', 'lpm_euclidean', 'lpm_mahalanobis',
        'llama_guard', 'bielik_guard', 'prompted'
    """
    if "Linear Probe" in method:
        return "linear_probe"
    elif "LPM" in method and "Euclidean" in method:
        return "lpm_euclidean"
    elif "LPM" in method and "Mahalanobis" in method:
        return "lpm_mahalanobis"
    elif "Llama-Guard" in method:
        return "llama_guard"
    elif "Bielik-Guard" in method:
        return "bielik_guard"
    elif "Prompted" in method:
        return "prompted"
    else:
        return "other"


def get_short_name(method: str) -> str:
    """Get short annotation name for the method.

    Args:
        method: Full method name

    Returns:
        Short name for annotation
    """
    name_map = {
        "Linear Probe(Bielik-1.5B)": "Probe (Bielik-1.5B)",
        "Linear Probe(Llama-3B)": "Probe (Llama-3B)",
        "LPM(Bielik-1.5B) Euclidean": "LPM (Bielik-1.5B)",
        "LPM(Bielik-1.5B) Mahalanobis": "LPM (Bielik-1.5B)",
        "LPM(Llama-3B) Euclidean": "LPM (Llama-3B)",
        "LPM(Llama-3B) Mahalanobis": "LPM (Llama-3B)",
        "Llama-Guard-3-1B": "LlamaGuard",
        "Bielik-Guard-0.1B": "BielikGuard",
        "Llama-3.2-3B-Prompted": "Llama-3B",
        "Bielik-4.5B-Prompted": "Bielik-4.5B",
    }
    return name_map.get(method, method)


def create_efficiency_plot(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (10, 7),
    dpi: int = 300,
) -> None:
    """Create efficiency comparison scatter plot.

    Args:
        df: DataFrame with columns: Method, FLOPs, F1, Category
        output_path: Path to save the plot
        figsize: Figure size in inches
        dpi: Resolution in dots per inch
    """
    # Setup plotting style (following visualization_rules.md)
    setup_plotting_style()

    # Parse data
    df = df.copy()
    df["FLOPs_numeric"] = df["FLOPs"].apply(parse_flops)
    df["F1_numeric"] = df["F1"].apply(parse_f1)
    df["Method_Type"] = df["Method"].apply(get_method_type)
    df["Short_Name"] = df["Method"].apply(get_short_name)

    # Define colors using Set2 palette
    set2_colors = sns.color_palette("Set2", 8)
    colors = {
        "linear_probe": set2_colors[0],  # Blue
        "lpm_euclidean": set2_colors[1],  # Orange
        "lpm_mahalanobis": set2_colors[2],  # Green
        "llama_guard": set2_colors[3],  # Red
        "bielik_guard": set2_colors[4],  # Purple
        "prompted": set2_colors[5],  # Brown
    }

    # Define markers
    markers = {
        "linear_probe": "o",  # Circle
        "lpm_euclidean": "o",  # Circle
        "lpm_mahalanobis": "o",  # Circle
        "llama_guard": "+",  # Plus
        "bielik_guard": "D",  # Diamond
        "prompted": "o",  # Circle
    }

    # Define marker sizes (larger for better visibility)
    marker_sizes = {
        "linear_probe": 180,
        "lpm_euclidean": 180,
        "lpm_mahalanobis": 180,
        "llama_guard": 300,  # Larger for + marker
        "bielik_guard": 180,
        "prompted": 180,
    }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot each method type
    plotted_types = set()
    for _, row in df.iterrows():
        method_type = row["Method_Type"]
        x = row["FLOPs_numeric"]
        y = row["F1_numeric"]
        color = colors[method_type]

        # Plot point (no border for filled markers, special handling for +)
        if markers[method_type] == "+":
            # For + marker, use c parameter (it uses this for the marker color)
            ax.scatter(
                x,
                y,
                c=[color],
                marker=markers[method_type],
                s=marker_sizes[method_type],
                alpha=0.9,
                linewidths=2.5,
                zorder=3,
            )
        else:
            # For filled markers
            ax.scatter(
                x,
                y,
                c=[color],
                marker=markers[method_type],
                s=marker_sizes[method_type],
                alpha=0.9,
                edgecolors="none",
                linewidths=0,
                zorder=3,
            )

        # Add annotation with better offset to avoid overlap
        # Use offset in data coordinates for better control
        offset_factor_x = 0.25  # Factor for log scale
        offset_y = 3.5  # Points offset in y direction

        # Adjust offset direction based on position to avoid crowding
        if method_type == "llama_guard":
            offset_factor_x = 0.15
            offset_y = 4.5
        elif method_type == "bielik_guard":
            offset_factor_x = -0.4
            offset_y = 3
        elif "Bielik-1.5B" in row["Method"]:
            offset_y = -5
        elif "Llama-3B" in row["Method"] and "Prompted" not in row["Method"]:
            # LPM (Llama-3B) - move left and up (rotate counter-clockwise ~30 degrees)
            if "Euclidean" in row["Method"]:
                offset_factor_x = 0.2
                offset_y = 5
            else:
                offset_y = 4
        elif "Prompted" in row["Method"]:
            # Prompted LLMs - point left and slightly up (~20 degree angle)
            offset_factor_x = -0.35
            offset_y = 2

        # Calculate x offset in log space
        x_offset = x * (10**offset_factor_x)
        y_offset = y + offset_y

        # Darken color for text (multiply RGB by 0.7 for darker shade)
        import matplotlib.colors as mcolors

        rgb = mcolors.to_rgb(color)
        darker_color = tuple(max(0, c * 0.7) for c in rgb)

        ax.annotate(
            row["Short_Name"],
            xy=(x, y),
            xytext=(x_offset, y_offset),
            fontsize=11,
            fontweight="bold",  # Make bold
            color=darker_color,  # Darker version of marker color
            ha="left" if offset_factor_x > 0 else "right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="none"),
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                alpha=0.5,
                linewidth=1,
            ),
            zorder=4,
        )

        plotted_types.add(method_type)

    # Set log scale for x-axis
    ax.set_xscale("log")

    # Configure x-axis
    ax.set_xlabel("FLOPs per sample (log scale)", fontsize=13)
    ax.set_xlim(1e5, 5e12)

    # Configure y-axis
    ax.set_ylabel("F1 Score", fontsize=13)
    ax.set_ylim(55, 95)

    # Add grid (y-axis only, following visualization rules)
    ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5, zorder=1)

    # Create legend
    legend_elements = []

    # Add color legend (method types)
    from matplotlib.lines import Line2D

    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors["linear_probe"],
            markersize=9,
            label="Linear Probe",
            markeredgecolor="none",
            markeredgewidth=0,
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors["lpm_euclidean"],
            markersize=9,
            label="LPM (Euclidean)",
            markeredgecolor="none",
            markeredgewidth=0,
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors["lpm_mahalanobis"],
            markersize=9,
            label="LPM (Mahalanobis)",
            markeredgecolor="none",
            markeredgewidth=0,
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors["prompted"],
            markersize=9,
            label="Prompted LLM",
            markeredgecolor="none",
            markeredgewidth=0,
        )
    )

    # Add guard markers
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="+",
            color="w",
            markerfacecolor=colors["llama_guard"],
            markeredgecolor=colors["llama_guard"],
            markersize=11,
            label="LlamaGuard (Guard)",
            markeredgewidth=2,
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=colors["bielik_guard"],
            markersize=8,
            label="BielikGuard (Guard)",
            markeredgecolor="none",
            markeredgewidth=0,
        )
    )

    ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=10,
        framealpha=0.95,
        edgecolor="gray",
        fancybox=False,
    )

    # Add title
    ax.set_title(
        "Efficiency vs Performance: Mechanistic Interpretability Methods vs Baselines",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"✅ Saved efficiency comparison plot to: {output_path}")

    plt.close()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize efficiency comparison table (FLOPs vs F1 Score)")
    parser.add_argument(
        "--table-path",
        type=Path,
        default=Path(__file__).parent.parent / "tables" / "efficiency_comparison.csv",
        help="Path to efficiency comparison CSV table (default: ../tables/efficiency_comparison.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "visualizations" / "efficiency_comparison.png",
        help="Path to save the output plot (default: ../visualizations/efficiency_comparison.png)",
    )
    args = parser.parse_args()

    table_path = Path(args.table_path).resolve()
    if not table_path.exists():
        print(f"❌ Table file not found: {table_path}")
        return 1

    output_path = Path(args.output).resolve()

    print("=" * 80)
    print("Efficiency Comparison Visualization")
    print("=" * 80)
    print(f"Input table: {table_path}")
    print(f"Output plot: {output_path}")
    print()

    # Load data
    print("📊 Loading efficiency comparison table...")
    df = pd.read_csv(table_path)
    print(f"  ✅ Loaded {len(df)} methods")
    print()

    # Create visualization
    print("🎨 Creating visualization...")
    create_efficiency_plot(df, output_path)
    print()

    print("=" * 80)
    print("✅ Visualization complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
