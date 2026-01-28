#!/usr/bin/env python3
# ruff: noqa
"""
Analyze baseline results: Guards and Prompted LLMs.

This script creates comprehensive tables and visualizations for baseline
safety classification methods including:
- Guard models (LlamaGuard-1B, BielikGuard-0.1B)
- Prompted LLMs (Llama-3B, Bielik-4.5B with multiple prompts)

Generates:
1. Detailed results table (CSV and LaTeX)
2. Bar chart with prompt variation whiskers
3. Heatmap showing F1 across all configurations
4. Prompt stability analysis (violin plots)

Usage:
    python analyze_baseline_results.py [--store-path PATH]
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
from experiments.results.analysis_code.baseline_tables import (
    create_baseline_results_table,
    create_latex_baseline_table,
)
from experiments.results.analysis_code.baseline_visualizations import (
    plot_baseline_bar_chart,
    plot_baseline_heatmap,
    plot_prompt_stability,
)

# ============================================================================
# Run IDs (from where_baseline_results_are_saved.md)
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


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze baseline results (Guards and Prompted LLMs)")
    parser.add_argument(
        "--store-path",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent / "store",
        help="Path to store directory (default: ../../store from script location)",
    )
    args = parser.parse_args()

    store_path = Path(args.store_path).resolve()
    if not store_path.exists():
        print(f"❌ Store path does not exist: {store_path}")
        return 1

    # Define output directories
    results_dir = Path(__file__).parent.parent
    tables_dir = results_dir / "tables"
    viz_dir = results_dir / "visualizations"
    tables_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Baseline Results Analysis")
    print("=" * 80)
    print(f"Store path: {store_path}")
    print(f"Tables output: {tables_dir}")
    print(f"Visualizations output: {viz_dir}")
    print()

    # ========================================================================
    # Load Experiment Results
    # ========================================================================

    print("📂 Loading baseline results...")
    print()

    print("  Loading guard baseline results...")
    baseline_df = load_baseline_results(store_path, BASELINE_RUN_IDS)
    print(f"    ✅ Loaded {len(baseline_df)} guard experiments")

    print("  Loading direct prompting results...")
    prompting_df = load_direct_prompting_results(store_path, DIRECT_PROMPTING_RUN_IDS)
    print(f"    ✅ Loaded {len(prompting_df)} direct prompting experiments")
    print()

    # ========================================================================
    # Create Results Table
    # ========================================================================

    print("📋 Creating baseline results table...")
    print()

    results_table = create_baseline_results_table(
        baseline_df=baseline_df,
        prompting_df=prompting_df,
        output_path=tables_dir / "baseline_results.csv",
    )

    # Also create LaTeX version
    create_latex_baseline_table(
        results_table,
        tables_dir / "baseline_results.tex",
        caption="Baseline Results: F1 Scores for Guards and Prompted LLMs across Datasets",
        label="tab:baseline_results",
    )

    # Display table
    print()
    print("=" * 80)
    print("Baseline Results Table")
    print("=" * 80)
    print()
    print(results_table.to_string(index=False))
    print()

    # ========================================================================
    # Create Visualizations
    # ========================================================================

    print("=" * 80)
    print("Creating Visualizations")
    print("=" * 80)
    print()

    print("📊 Visualization 1: Bar Chart with Prompt Whiskers")
    plot_baseline_bar_chart(
        baseline_df=baseline_df,
        prompting_df=prompting_df,
        output_path=viz_dir / "baseline_bar_chart.png",
    )
    print()

    print("📊 Visualization 2: F1 Score Heatmap")
    plot_baseline_heatmap(
        baseline_df=baseline_df,
        prompting_df=prompting_df,
        output_path=viz_dir / "baseline_heatmap.png",
    )
    print()

    print("📊 Visualization 3: Prompt Stability Analysis")
    plot_prompt_stability(
        prompting_df=prompting_df,
        output_path=viz_dir / "prompt_stability.png",
    )
    print()

    # ========================================================================
    # Summary Statistics
    # ========================================================================

    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print()

    # Guard models
    print("Guard Models:")
    for model_short, model_name in [("llamaguard", "LlamaGuard-1B"), ("bielik", "BielikGuard-0.1B")]:
        model_data = baseline_df[baseline_df["model_short"] == model_short]
        avg_f1 = model_data["f1"].mean()
        print(f"  {model_name}: Avg F1 = {avg_f1:.4f}")
    print()

    # Prompted models
    print("Prompted Models (averaged across prompts and datasets):")
    for model_short, model_name in [
        ("llama-3_2-3b-instruct", "Llama-3B"),
        ("bielik-4_5b-v3_0-instruct", "Bielik-4.5B"),
    ]:
        model_data = prompting_df[prompting_df["model_short"] == model_short]
        avg_f1 = model_data["f1"].mean()
        std_f1 = model_data["f1"].std()
        min_f1 = model_data["f1"].min()
        max_f1 = model_data["f1"].max()
        print(f"  {model_name}:")
        print(f"    Mean F1: {avg_f1:.4f}")
        print(f"    Std F1:  {std_f1:.4f}")
        print(f"    Range:   {min_f1:.4f} - {max_f1:.4f}")
    print()

    print("=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
