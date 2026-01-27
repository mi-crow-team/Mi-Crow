#!/usr/bin/env python3
# ruff: noqa
"""
Analyze LPM and Linear Probe experiment results.

This script loads results from all LPM and Linear Probe experiments,
creates visualizations for the thesis, and generates result tables.

Usage:
    python analyze_lpm_probe_results.py [--store-path PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.results.analysis_code.result_loader import (
    load_lpm_results,
    load_probe_results,
)
from experiments.results.analysis_code.tables import (
    create_best_results_summary,
    create_latex_table,
    create_lpm_results_table,
    create_probe_results_table,
)
from experiments.results.analysis_code.visualizations import (
    plot_aggregation_impact,
    plot_lpm_metric_comparison,
    plot_method_comparison,
)
from experiments.results.analysis_code.new_visualizations import (
    plot_aggregation_impact_consolidated,
    plot_method_comparison_detailed,
)

# LPM run_ids (from where_results_are_saved.md)
LPM_RUN_IDS = [
    "lpm_bielik_1_5b_plmix_train_plmix_test_last_token_layer31_euclidean",
    "lpm_bielik_1_5b_plmix_train_plmix_test_last_token_layer31_mahalanobis",
    "lpm_bielik_1_5b_plmix_train_plmix_test_last_token_prefix_layer31_euclidean",
    "lpm_bielik_1_5b_plmix_train_plmix_test_last_token_prefix_layer31_mahalanobis",
    "lpm_bielik_1_5b_plmix_train_plmix_test_mean_layer31_euclidean",
    "lpm_bielik_1_5b_plmix_train_plmix_test_mean_layer31_mahalanobis",
    "lpm_bielik_1_5b_wgmix_train_wgmix_test_last_token_layer31_euclidean",
    "lpm_bielik_1_5b_wgmix_train_wgmix_test_last_token_layer31_mahalanobis",
    "lpm_bielik_1_5b_wgmix_train_wgmix_test_last_token_prefix_layer31_euclidean",
    "lpm_bielik_1_5b_wgmix_train_wgmix_test_last_token_prefix_layer31_mahalanobis",
    "lpm_bielik_1_5b_wgmix_train_wgmix_test_mean_layer31_euclidean",
    "lpm_bielik_1_5b_wgmix_train_wgmix_test_mean_layer31_mahalanobis",
    "lpm_bielik_4_5b_plmix_train_plmix_test_last_token_layer59_euclidean",
    "lpm_bielik_4_5b_plmix_train_plmix_test_last_token_layer59_mahalanobis",
    "lpm_bielik_4_5b_plmix_train_plmix_test_last_token_prefix_layer59_euclidean",
    "lpm_bielik_4_5b_plmix_train_plmix_test_last_token_prefix_layer59_mahalanobis",
    "lpm_bielik_4_5b_plmix_train_plmix_test_mean_layer59_euclidean",
    "lpm_bielik_4_5b_plmix_train_plmix_test_mean_layer59_mahalanobis",
    "lpm_bielik_4_5b_wgmix_train_wgmix_test_last_token_layer59_euclidean",
    "lpm_bielik_4_5b_wgmix_train_wgmix_test_last_token_layer59_mahalanobis",
    "lpm_bielik_4_5b_wgmix_train_wgmix_test_last_token_prefix_layer59_euclidean",
    "lpm_bielik_4_5b_wgmix_train_wgmix_test_last_token_prefix_layer59_mahalanobis",
    "lpm_bielik_4_5b_wgmix_train_wgmix_test_mean_layer59_euclidean",
    "lpm_bielik_4_5b_wgmix_train_wgmix_test_mean_layer59_mahalanobis",
    "lpm_llama_3b_plmix_train_plmix_test_last_token_layer27_euclidean",
    "lpm_llama_3b_plmix_train_plmix_test_last_token_layer27_mahalanobis",
    "lpm_llama_3b_plmix_train_plmix_test_last_token_prefix_layer27_euclidean",
    "lpm_llama_3b_plmix_train_plmix_test_last_token_prefix_layer27_mahalanobis",
    "lpm_llama_3b_plmix_train_plmix_test_mean_layer27_euclidean",
    "lpm_llama_3b_plmix_train_plmix_test_mean_layer27_mahalanobis",
    "lpm_llama_3b_wgmix_train_wgmix_test_last_token_layer27_euclidean",
    "lpm_llama_3b_wgmix_train_wgmix_test_last_token_layer27_mahalanobis",
    "lpm_llama_3b_wgmix_train_wgmix_test_last_token_prefix_layer27_euclidean",
    "lpm_llama_3b_wgmix_train_wgmix_test_last_token_prefix_layer27_mahalanobis",
    "lpm_llama_3b_wgmix_train_wgmix_test_mean_layer27_euclidean",
    "lpm_llama_3b_wgmix_train_wgmix_test_mean_layer27_mahalanobis",
]

# Linear Probe run_ids (from where_results_are_saved.md)
PROBE_RUN_IDS = [
    "probe_bielik_1_5b_plmix_train_plmix_test_last_token_layer31",
    "probe_bielik_1_5b_plmix_train_plmix_test_last_token_prefix_layer31",
    "probe_bielik_1_5b_plmix_train_plmix_test_mean_layer31",
    "probe_bielik_1_5b_wgmix_train_wgmix_test_last_token_layer31",
    "probe_bielik_1_5b_wgmix_train_wgmix_test_last_token_prefix_layer31",
    "probe_bielik_1_5b_wgmix_train_wgmix_test_mean_layer31",
    "probe_bielik_4_5b_plmix_train_plmix_test_last_token_layer59",
    "probe_bielik_4_5b_plmix_train_plmix_test_last_token_prefix_layer59",
    "probe_bielik_4_5b_plmix_train_plmix_test_mean_layer59",
    "probe_bielik_4_5b_wgmix_train_wgmix_test_last_token_layer59",
    "probe_bielik_4_5b_wgmix_train_wgmix_test_last_token_prefix_layer59",
    "probe_bielik_4_5b_wgmix_train_wgmix_test_mean_layer59",
    "probe_llama_3b_plmix_train_plmix_test_last_token_layer27",
    "probe_llama_3b_plmix_train_plmix_test_last_token_prefix_layer27",
    "probe_llama_3b_plmix_train_plmix_test_mean_layer27",
    "probe_llama_3b_wgmix_train_wgmix_test_last_token_layer27",
    "probe_llama_3b_wgmix_train_wgmix_test_last_token_prefix_layer27",
    "probe_llama_3b_wgmix_train_wgmix_test_mean_layer27",
]


def main():
    parser = argparse.ArgumentParser(description="Analyze LPM and Linear Probe results")
    parser.add_argument(
        "--store-path",
        type=str,
        default="store",
        help="Path to store directory (default: store)",
    )
    args = parser.parse_args()

    store_path = Path(args.store_path)
    results_dir = Path("experiments/results")
    viz_dir = results_dir / "visualizations"
    tables_dir = results_dir / "tables"

    print("=" * 80)
    print("LPM and Linear Probe Results Analysis")
    print("=" * 80)
    print(f"Store path: {store_path}")
    print(f"Visualizations: {viz_dir}")
    print(f"Tables: {tables_dir}")
    print()

    # ========================================================================
    # Load Results
    # ========================================================================

    print("Loading LPM results...")
    lpm_df = load_lpm_results(store_path, LPM_RUN_IDS)
    print(f"✅ Loaded {len(lpm_df)} LPM results")
    print()

    print("Loading Linear Probe results...")
    probe_df = load_probe_results(store_path, PROBE_RUN_IDS)
    print(f"✅ Loaded {len(probe_df)} Linear Probe results")
    print()

    if len(lpm_df) == 0 and len(probe_df) == 0:
        print("❌ No results found! Check store path and run_ids.")
        return 1

    # ========================================================================
    # Create Visualizations
    # ========================================================================

    print("Creating visualizations...")
    print()

    if len(lpm_df) > 0:
        # Figure 1: LPM Metric Comparison
        print("📊 Figure 1: LPM Metric Comparison (Euclidean vs. Mahalanobis)")
        plot_lpm_metric_comparison(
            lpm_df,
            viz_dir / "fig1_lpm_metric_comparison_mean.png",
            aggregation="mean",
            show_whiskers=True,
        )
        print()

        # Figure 2a: LPM Aggregation Impact
        print("📊 Figure 2a: LPM Aggregation Impact")
        plot_aggregation_impact(
            lpm_df,
            viz_dir,
            method="LPM",
            metric="mahalanobis",
        )
        print()

    if len(probe_df) > 0:
        # Figure 2b: Linear Probe Aggregation Impact
        print("📊 Figure 2b: Linear Probe Aggregation Impact")
        plot_aggregation_impact(
            probe_df,
            viz_dir,
            method="Linear Probe",
        )
        print()

    if len(lpm_df) > 0 and len(probe_df) > 0:
        # Figure 3: Method Comparison (with stability lines)
        print("📊 Figure 3: Method Comparison (LPM vs. Linear Probe with Stability)")
        plot_method_comparison(
            lpm_df,
            probe_df,
            viz_dir / "fig3_method_comparison.png",
        )
        print()

        # Figure 4: Consolidated Aggregation Impact
        print("📊 Figure 4: Consolidated Aggregation Impact")
        plot_aggregation_impact_consolidated(
            lpm_df,
            probe_df,
            viz_dir / "fig4_aggregation_consolidated.png",
        )
        print()

        # Figure 5: Detailed Method Comparison (All Configurations)
        print("📊 Figure 5: Detailed Method Comparison (All Configurations)")
        plot_method_comparison_detailed(
            lpm_df,
            probe_df,
            viz_dir / "fig5_method_comparison_detailed.png",
        )
        print()

    # ========================================================================
    # Create Tables
    # ========================================================================

    print("Creating tables...")
    print()

    if len(lpm_df) > 0:
        print("📋 LPM Results Table")
        lpm_table = create_lpm_results_table(lpm_df, tables_dir / "lpm_results.csv")

        # Also create LaTeX version
        create_latex_table(
            lpm_table,
            tables_dir / "lpm_results.tex",
            caption="LPM Results: F1, Precision, Recall, and Accuracy across Models, Aggregations, and Distance Metrics",
            label="tab:lpm_results",
        )
        print()

    if len(probe_df) > 0:
        print("📋 Linear Probe Results Table")
        probe_table = create_probe_results_table(probe_df, tables_dir / "probe_results.csv")

        # Also create LaTeX version
        create_latex_table(
            probe_table,
            tables_dir / "probe_results.tex",
            caption="Linear Probe Results: F1, Precision, Recall, and Accuracy across Models and Aggregations",
            label="tab:probe_results",
        )
        print()

    if len(lpm_df) > 0 and len(probe_df) > 0:
        print("📋 Best Results Summary")
        summary_table = create_best_results_summary(
            lpm_df,
            probe_df,
            tables_dir / "best_results_summary.csv",
        )

        # Also create LaTeX version
        create_latex_table(
            summary_table,
            tables_dir / "best_results_summary.tex",
            caption="Best Results Summary: Top Performing Configurations for Each Method and Dataset",
            label="tab:best_results",
        )
        print()

    # ========================================================================
    # Summary Statistics
    # ========================================================================

    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    if len(lpm_df) > 0:
        print("\nLPM Results:")
        print(f"  Total experiments: {len(lpm_df)}")
        print(f"  Mean F1: {lpm_df['f1'].mean():.4f}")
        print(f"  Best F1: {lpm_df['f1'].max():.4f}")
        print(f"  Worst F1: {lpm_df['f1'].min():.4f}")

        best_lpm = lpm_df.loc[lpm_df["f1"].idxmax()]
        print("\n  Best Configuration:")
        print(f"    Model: {best_lpm['model']}")
        print(f"    Dataset: {best_lpm['test_dataset']}")
        print(f"    Aggregation: {best_lpm['aggregation']}")
        print(f"    Metric: {best_lpm['metric']}")
        print(f"    F1: {best_lpm['f1']:.4f}")

    if len(probe_df) > 0:
        print("\nLinear Probe Results:")
        print(f"  Total experiments: {len(probe_df)}")
        print(f"  Mean F1: {probe_df['f1'].mean():.4f}")
        print(f"  Best F1: {probe_df['f1'].max():.4f}")
        print(f"  Worst F1: {probe_df['f1'].min():.4f}")

        best_probe = probe_df.loc[probe_df["f1"].idxmax()]
        print("\n  Best Configuration:")
        print(f"    Model: {best_probe['model']}")
        print(f"    Dataset: {best_probe['test_dataset']}")
        print(f"    Aggregation: {best_probe['aggregation']}")
        print(f"    F1: {best_probe['f1']:.4f}")

    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
