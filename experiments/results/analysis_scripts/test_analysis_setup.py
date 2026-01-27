#!/usr/bin/env python3
"""
Quick test script to verify analysis code setup.

This script checks if the store directory has the expected structure
and if we can load at least one result file.

Usage:
    python test_analysis_setup.py [--store-path PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.results.analysis_code.result_loader import (
    find_latest_inference_run,
    load_metrics,
    parse_lpm_run_id,
    parse_probe_run_id,
    recalculate_accuracy,
)


def test_lpm_run(store_path: Path, run_id: str):
    """Test loading a single LPM run."""
    print(f"\nTesting LPM run: {run_id}")

    try:
        params = parse_lpm_run_id(run_id)
        print(f"  ✅ Parsed parameters: {params}")
    except Exception as e:
        print(f"  ❌ Failed to parse run_id: {e}")
        return False

    run_dir = store_path / run_id
    if not run_dir.exists():
        print(f"  ❌ Run directory not found: {run_dir}")
        return False
    print(f"  ✅ Run directory exists: {run_dir}")

    latest_inference = find_latest_inference_run(run_dir)
    if latest_inference is None:
        print("  ❌ No inference run found")
        return False
    print(f"  ✅ Latest inference: {latest_inference.name}")

    metrics_path = latest_inference / "analysis" / "metrics.json"
    if not metrics_path.exists():
        print(f"  ❌ Metrics file not found: {metrics_path}")
        return False
    print("  ✅ Metrics file exists")

    try:
        metrics = load_metrics(metrics_path)
        print("  ✅ Loaded metrics:")
        print(f"     F1: {metrics.get('f1', 'N/A'):.4f}")
        print(f"     Precision: {metrics.get('precision', 'N/A'):.4f}")
        print(f"     Recall: {metrics.get('recall', 'N/A'):.4f}")
        print(f"     Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        return True
    except Exception as e:
        print(f"  ❌ Failed to load metrics: {e}")
        return False


def test_probe_run(store_path: Path, run_id: str):
    """Test loading a single Linear Probe run."""
    print(f"\nTesting Probe run: {run_id}")

    try:
        params = parse_probe_run_id(run_id)
        print(f"  ✅ Parsed parameters: {params}")
    except Exception as e:
        print(f"  ❌ Failed to parse run_id: {e}")
        return False

    run_dir = store_path / run_id
    if not run_dir.exists():
        print(f"  ❌ Run directory not found: {run_dir}")
        return False
    print(f"  ✅ Run directory exists: {run_dir}")

    latest_inference = find_latest_inference_run(run_dir)
    if latest_inference is None:
        print("  ❌ No inference run found")
        return False
    print(f"  ✅ Latest inference: {latest_inference.name}")

    metrics_path = latest_inference / "analysis" / "metrics.json"
    if not metrics_path.exists():
        print(f"  ❌ Metrics file not found: {metrics_path}")
        return False
    print("  ✅ Metrics file exists")

    try:
        metrics = load_metrics(metrics_path)
        accuracy_original = metrics.get("accuracy", 0.0)
        accuracy_recalc = recalculate_accuracy(metrics)

        print("  ✅ Loaded metrics:")
        print(f"     F1: {metrics.get('f1', 'N/A'):.4f}")
        print(f"     Precision: {metrics.get('precision', 'N/A'):.4f}")
        print(f"     Recall: {metrics.get('recall', 'N/A'):.4f}")
        print(f"     Accuracy (original): {accuracy_original:.4f}")
        print(f"     Accuracy (recalculated): {accuracy_recalc:.4f}")

        if abs(accuracy_original - accuracy_recalc) > 0.001:
            print("  ⚠️  Accuracy mismatch detected (as expected for probes)")

        return True
    except Exception as e:
        print(f"  ❌ Failed to load metrics: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test analysis setup")
    parser.add_argument(
        "--store-path",
        type=str,
        default="store",
        help="Path to store directory",
    )
    args = parser.parse_args()

    store_path = Path(args.store_path)

    print("=" * 80)
    print("Analysis Setup Test")
    print("=" * 80)
    print(f"Store path: {store_path}")

    if not store_path.exists():
        print(f"\n❌ Store path does not exist: {store_path}")
        print("   Please run experiments first or provide correct --store-path")
        return 1

    # List all LPM and probe directories
    lpm_dirs = sorted([d.name for d in store_path.iterdir() if d.is_dir() and d.name.startswith("lpm_")])
    probe_dirs = sorted([d.name for d in store_path.iterdir() if d.is_dir() and d.name.startswith("probe_")])

    print(f"\nFound {len(lpm_dirs)} LPM directories")
    print(f"Found {len(probe_dirs)} Probe directories")

    # Test first LPM run if available
    if lpm_dirs:
        success = test_lpm_run(store_path, lpm_dirs[0])
        if not success:
            print("\n⚠️  LPM test failed - check if experiments have completed")
    else:
        print("\n⚠️  No LPM directories found - have experiments been run?")

    # Test first Probe run if available
    if probe_dirs:
        success = test_probe_run(store_path, probe_dirs[0])
        if not success:
            print("\n⚠️  Probe test failed - check if experiments have completed")
    else:
        print("\n⚠️  No Probe directories found - have experiments been run?")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

    if lpm_dirs or probe_dirs:
        print("\n✅ Store structure looks good!")
        print("   You can now run: python experiments/results/analysis_scripts/analyze_lpm_probe_results.py")
        return 0
    else:
        print("\n❌ No experiment results found")
        print("   Please run the experiments first")
        return 1


if __name__ == "__main__":
    sys.exit(main())
