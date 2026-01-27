#!/usr/bin/env python3
# ruff: noqa
"""Quick test to verify new visualization enhancements work."""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis_code"))

from experiments.results.analysis_code.new_visualizations import (
    plot_aggregation_impact_consolidated,
    plot_method_comparison_detailed,
)
from experiments.results.analysis_code.visualizations import (
    plot_lpm_metric_comparison,
    plot_method_comparison,
)

print("=" * 80)
print("Testing New Visualization Enhancements")
print("=" * 80)

# Create mock data
print("\n1. Creating mock data...")

lpm_data = []
probe_data = []

for dataset in ["plmix_test", "wgmix_test"]:
    for model in ["Bielik-1.5B", "Bielik-4.5B", "Llama-3.2-3B"]:
        for agg in ["mean", "last_token", "last_token_prefix"]:
            # LPM: Two metrics
            for metric in ["euclidean", "mahalanobis"]:
                f1 = np.random.uniform(0.6, 0.85)
                lpm_data.append(
                    {
                        "method": "LPM",
                        "model": model,
                        "test_dataset": dataset,
                        "aggregation": agg,
                        "metric": metric,
                        "f1": f1,
                        "precision": f1 + 0.02,
                        "recall": f1 - 0.01,
                        "accuracy": f1 + 0.01,
                    }
                )

            # Linear Probe
            f1 = np.random.uniform(0.65, 0.8)
            probe_data.append(
                {
                    "method": "Linear Probe",
                    "model": model,
                    "test_dataset": dataset,
                    "aggregation": agg,
                    "f1": f1,
                    "precision": f1 + 0.02,
                    "recall": f1 - 0.01,
                    "accuracy": f1 + 0.01,
                }
            )

lpm_df = pd.DataFrame(lpm_data)
probe_df = pd.DataFrame(probe_data)

print(f"✅ Created {len(lpm_df)} LPM rows, {len(probe_df)} Probe rows")

# Test output directory
output_dir = Path("experiments/results/visualizations/test")
output_dir.mkdir(parents=True, exist_ok=True)

# Test Figure 1 (enhanced with whiskers)
print("\n2. Testing Figure 1 (LPM metric comparison with whiskers)...")
try:
    plot_lpm_metric_comparison(
        lpm_df,
        output_dir / "test_fig1_whiskers.png",
        aggregation="all",
        show_whiskers=True,
    )
    print("✅ Figure 1 generated successfully")
except Exception as e:
    print(f"❌ Figure 1 failed: {e}")

# Test Figure 3 (enhanced with stability lines)
print("\n3. Testing Figure 3 (method comparison with stability lines)...")
try:
    plot_method_comparison(
        lpm_df,
        probe_df,
        output_dir / "test_fig3_stability.png",
    )
    print("✅ Figure 3 generated successfully")
except Exception as e:
    print(f"❌ Figure 3 failed: {e}")

# Test Figure 4 (consolidated aggregation impact)
print("\n4. Testing Figure 4 (consolidated aggregation impact)...")
try:
    plot_aggregation_impact_consolidated(
        lpm_df,
        probe_df,
        output_dir / "test_fig4_consolidated.png",
    )
    print("✅ Figure 4 generated successfully")
except Exception as e:
    print(f"❌ Figure 4 failed: {e}")

# Test Figure 5 (detailed method comparison)
print("\n5. Testing Figure 5 (detailed method comparison)...")
try:
    plot_method_comparison_detailed(
        lpm_df,
        probe_df,
        output_dir / "test_fig5_detailed.png",
    )
    print("✅ Figure 5 generated successfully")
except Exception as e:
    print(f"❌ Figure 5 failed: {e}")

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
print(f"\nTest outputs saved to: {output_dir.absolute()}")
print("\nNext steps:")
print("1. Review test plots to verify they look correct")
print("2. Run full analysis: python experiments/results/analysis_scripts/analyze_lpm_probe_results.py")
print("3. Integrate selected figures into thesis")
