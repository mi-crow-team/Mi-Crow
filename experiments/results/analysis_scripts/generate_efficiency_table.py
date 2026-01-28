#!/usr/bin/env python3
# ruff: noqa
"""
Generate efficiency comparison table for mechanistic interpretability methods.

This script creates a comprehensive table comparing MI methods (LPM, Linear Probe)
with baseline models (Guards, Prompted LLMs) in terms of:
- Computational efficiency (FLOPs)
- Performance (F1 score averaged across datasets)

For MI methods, only Mean aggregation is considered.
For prompted baselines, results are averaged across all prompts.
For all methods, results are averaged across plmix and wgmix datasets.

Usage:
    python generate_efficiency_table.py [--store-path PATH] [--flops-json PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.results.analysis_code.baseline_loader import (
    load_baseline_results,
    load_direct_prompting_results,
)
from experiments.results.analysis_code.efficiency_table import (
    create_efficiency_comparison_table,
    create_latex_efficiency_table,
    load_flops_data,
)
from experiments.results.analysis_code.result_loader import (
    load_lpm_results,
    load_probe_results,
)

# ============================================================================
# Run IDs (from where_baseline_results_are_saved.md and where_results_are_saved.md)
# ============================================================================

# Baseline Guard run_ids
BASELINE_RUN_IDS = [
    "baseline_bielik_wgmix_test_20260105_045726",
    "baseline_bielik_plmix_test_20260105_045953",
    "baseline_llamaguard_plmix_test_20260105_050006",
    "baseline_llamaguard_wgmix_test_20260105_045759",
]

# Direct Prompting run_ids
DIRECT_PROMPTING_RUN_IDS = [
    # Bielik-4.5B
    "direct_prompting_prompt_0_bielik-4_5b-v3_0-instruct_plmix_test_20260105_213956",
    "direct_prompting_prompt_0_bielik-4_5b-v3_0-instruct_wgmix_test_20260105_213006",
    "direct_prompting_prompt_1_bielik-4_5b-v3_0-instruct_plmix_test_20260105_214024",
    "direct_prompting_prompt_1_bielik-4_5b-v3_0-instruct_wgmix_test_20260105_213422",
    "direct_prompting_prompt_2_bielik-4_5b-v3_0-instruct_plmix_test_20260105_214049",
    "direct_prompting_prompt_2_bielik-4_5b-v3_0-instruct_wgmix_test_20260105_213620",
    "direct_prompting_prompt_3_bielik-4_5b-v3_0-instruct_plmix_test_20260105_214109",
    "direct_prompting_prompt_3_bielik-4_5b-v3_0-instruct_wgmix_test_20260105_213750",
    # Llama-3.2-3B
    "direct_prompting_prompt_0_llama-3_2-3b-instruct_plmix_test_20260105_212849",
    "direct_prompting_prompt_0_llama-3_2-3b-instruct_wgmix_test_20260105_212500",
    "direct_prompting_prompt_1_llama-3_2-3b-instruct_plmix_test_20260105_212909",
    "direct_prompting_prompt_1_llama-3_2-3b-instruct_wgmix_test_20260105_212603",
    "direct_prompting_prompt_2_llama-3_2-3b-instruct_plmix_test_20260105_212927",
    "direct_prompting_prompt_2_llama-3_2-3b-instruct_wgmix_test_20260105_212653",
    "direct_prompting_prompt_3_llama-3_2-3b-instruct_plmix_test_20260105_212943",
    "direct_prompting_prompt_3_llama-3_2-3b-instruct_wgmix_test_20260105_212738",
]

# LPM run_ids
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

# Linear Probe run_ids
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


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate efficiency comparison table for MI methods vs baselines")
    parser.add_argument(
        "--store-path",
        type=Path,
        default="store",
        help="Path to store directory (default: ../../store from script location)",
    )
    parser.add_argument(
        "--flops-json",
        type=Path,
        default=Path(__file__).parent.parent.parent / ".llm_context" / "experiments" / "flops.json",
        help="Path to FLOPs JSON file (default: experiments/.llm_context/experiments/flops.json)",
    )
    args = parser.parse_args()

    store_path = Path(args.store_path).resolve()
    if not store_path.exists():
        print(f"❌ Store path does not exist: {store_path}")
        return 1

    # Define output directories
    results_dir = Path(__file__).parent.parent
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Path to flops.json
    flops_json_path = Path(args.flops_json).resolve()
    if not flops_json_path.exists():
        print(f"❌ FLOPs file not found: {flops_json_path}")
        return 1

    print("=" * 80)
    print("Efficiency Comparison Table Generator")
    print("=" * 80)
    print(f"Store path: {store_path}")
    print(f"Tables output: {tables_dir}")
    print(f"FLOPs data: {flops_json_path}")
    print()

    # ========================================================================
    # Load FLOPs Data
    # ========================================================================

    print("📊 Loading FLOPs data...")
    flops_data = load_flops_data(flops_json_path)
    print(f"  ✅ Loaded {len(flops_data['baseline_llm_monitors']['methods'])} baseline methods")
    print(f"  ✅ Loaded {len(flops_data['mechanistic_interpretability_methods']['methods'])} MI methods")
    print()

    # ========================================================================
    # Load Experiment Results
    # ========================================================================

    print("📂 Loading experiment results...")
    print()

    # Load MI method results
    print("  Loading LPM results...")
    lpm_df = load_lpm_results(store_path, LPM_RUN_IDS)
    print(f"    ✅ Loaded {len(lpm_df)} LPM experiments")

    print("  Loading Linear Probe results...")
    probe_df = load_probe_results(store_path, PROBE_RUN_IDS)
    print(f"    ✅ Loaded {len(probe_df)} Linear Probe experiments")

    # Load baseline results
    print("  Loading baseline guard results...")
    baseline_df = load_baseline_results(store_path, BASELINE_RUN_IDS)
    print(f"    ✅ Loaded {len(baseline_df)} baseline guard experiments")

    print("  Loading direct prompting results...")
    prompting_df = load_direct_prompting_results(store_path, DIRECT_PROMPTING_RUN_IDS)
    print(f"    ✅ Loaded {len(prompting_df)} direct prompting experiments")
    print()

    # ========================================================================
    # Create Efficiency Comparison Table
    # ========================================================================

    print("📋 Creating efficiency comparison table...")
    print()

    efficiency_table = create_efficiency_comparison_table(
        lpm_df=lpm_df,
        probe_df=probe_df,
        baseline_df=baseline_df,
        prompting_df=prompting_df,
        flops_data=flops_data,
        output_path=tables_dir / "efficiency_comparison.csv",
    )

    # Also create LaTeX version
    create_latex_efficiency_table(
        efficiency_table,
        tables_dir / "efficiency_comparison.tex",
        caption="Efficiency Comparison: Computational Cost (FLOPs) vs Performance (F1 Score)",
        label="tab:efficiency_comparison",
    )

    # ========================================================================
    # Display Table
    # ========================================================================

    print()
    print("=" * 80)
    print("Efficiency Comparison Table")
    print("=" * 80)
    print()
    print(efficiency_table.to_string(index=False))
    print()

    # ========================================================================
    # Summary Statistics
    # ========================================================================

    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print()

    # Calculate average F1 by category
    for category in efficiency_table["Category"].unique():
        category_data = efficiency_table[efficiency_table["Category"] == category]
        # Parse F1 percentages back to float
        f1_values = category_data["F1"].str.rstrip("%").astype(float) / 100
        avg_f1 = f1_values.mean()
        print(f"{category}:")
        print(f"  Average F1: {avg_f1 * 100:.2f}%")
        print(f"  Number of methods: {len(category_data)}")
        print()

    print("=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
