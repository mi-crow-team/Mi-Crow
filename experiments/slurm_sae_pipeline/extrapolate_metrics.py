#!/usr/bin/env python3
"""
Extrapolate hardware metrics from short baseline to 10-hour projections.

Takes baseline measurements and generates realistic 10-hour time series.
"""

import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


def add_realistic_variation(value: float, base_value: float, noise_level: float = 0.1) -> float:
    """
    Add realistic variation to a value.
    
    Args:
        value: Current value
        base_value: Baseline/average value
        noise_level: Amount of noise (0.0 to 1.0)
        
    Returns:
        Value with added variation
    """
    # Add Gaussian noise
    noise = random.gauss(0, base_value * noise_level)
    new_value = value + noise
    
    # Ensure non-negative
    return max(0.0, new_value)


def extrapolate_metrics(
    baseline_metrics: List[Dict[str, Any]],
    target_duration_hours: float = 10.0,
    interval_seconds: float = 60.0
) -> List[Dict[str, Any]]:
    """
    Extrapolate baseline metrics to target duration.
    
    Args:
        baseline_metrics: List of baseline metric dictionaries
        target_duration_hours: Target duration in hours (default: 10.0)
        interval_seconds: Interval between measurements in seconds (default: 60.0)
        
    Returns:
        List of extrapolated metrics
    """
    if not baseline_metrics:
        return []
    
    # Calculate statistics from baseline
    def get_statistics(key_path: str) -> Dict[str, float]:
        """Extract statistics for a nested metric key."""
        values = []
        for metric in baseline_metrics:
            obj = metric
            for key in key_path.split("."):
                if isinstance(obj, dict) and key in obj:
                    obj = obj[key]
                else:
                    break
            else:
                if isinstance(obj, (int, float)):
                    values.append(obj)
        
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    
    # Get statistics for each metric type
    cpu_stats = get_statistics("cpu.cpu_percent_overall")
    ram_stats = get_statistics("ram.ram_percent")
    gpu_util_stats = get_statistics("gpu.utilization_gpu_pct")
    gpu_mem_stats = get_statistics("gpu.memory_used_mib")
    
    # Generate time series
    num_points = int((target_duration_hours * 3600) / interval_seconds)
    start_time = datetime.now()
    
    extrapolated = []
    
    for i in range(num_points):
        timestamp = start_time + timedelta(seconds=i * interval_seconds)
        
        # Add gradual trends (slight increase/decrease over time)
        progress = i / num_points
        trend_factor = 1.0 + (progress - 0.5) * 0.1  # ±5% trend
        
        # Generate metrics with variation
        cpu_percent = add_realistic_variation(
            cpu_stats["mean"] * trend_factor,
            cpu_stats["mean"],
            noise_level=0.15
        )
        cpu_percent = max(0.0, min(100.0, cpu_percent))
        
        ram_percent = add_realistic_variation(
            ram_stats["mean"] * trend_factor,
            ram_stats["mean"],
            noise_level=0.1
        )
        ram_percent = max(0.0, min(100.0, ram_percent))
        
        # Get baseline RAM values for calculation
        ram_total = None
        ram_used = None
        if baseline_metrics and "ram" in baseline_metrics[0]:
            ram_total = baseline_metrics[0]["ram"].get("ram_total_gb", 32.0)
            ram_used = ram_total * (ram_percent / 100.0)
        
        gpu_util = None
        gpu_mem_used = None
        gpu_mem_total = None
        gpu_name = None
        gpu_index = None
        
        if baseline_metrics and baseline_metrics[0].get("gpu"):
            gpu_baseline = baseline_metrics[0]["gpu"]
            gpu_name = gpu_baseline.get("name", "NVIDIA GPU")
            gpu_index = gpu_baseline.get("index", 0)
            gpu_mem_total = gpu_baseline.get("memory_total_mib", 81920.0)
            
            gpu_util = add_realistic_variation(
                gpu_util_stats["mean"] * trend_factor,
                gpu_util_stats["mean"],
                noise_level=0.2
            )
            gpu_util = max(0.0, min(100.0, gpu_util))
            
            gpu_mem_used = add_realistic_variation(
                gpu_mem_stats["mean"] * trend_factor,
                gpu_mem_stats["mean"],
                noise_level=0.15
            )
            gpu_mem_used = max(0.0, min(gpu_mem_total, gpu_mem_used))
        
        metric = {
            "timestamp": timestamp.isoformat(),
            "cpu": {
                "cpu_percent_overall": cpu_percent,
                "cpu_count": baseline_metrics[0]["cpu"]["cpu_count"] if baseline_metrics else 8,
            },
            "ram": {
                "ram_total_gb": ram_total or 32.0,
                "ram_used_gb": ram_used or (ram_total or 32.0) * (ram_percent / 100.0),
                "ram_percent": ram_percent,
            },
        }
        
        if gpu_util is not None:
            metric["gpu"] = {
                "index": gpu_index or 0,
                "name": gpu_name or "NVIDIA GPU",
                "utilization_gpu_pct": gpu_util,
                "utilization_memory_pct": (gpu_mem_used / gpu_mem_total * 100) if gpu_mem_total else 0.0,
                "memory_used_mib": gpu_mem_used,
                "memory_total_mib": gpu_mem_total,
            }
        
        extrapolated.append(metric)
    
    return extrapolated


def main():
    parser = argparse.ArgumentParser(description="Extrapolate hardware metrics to 10 hours")
    parser.add_argument(
        "baseline_file",
        type=str,
        help="Path to baseline metrics JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: baseline_file with _10h suffix)"
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=10.0,
        help="Target duration in hours (default: 10.0)"
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=60.0,
        help="Interval between measurements in seconds (default: 60.0)"
    )
    args = parser.parse_args()
    
    baseline_path = Path(args.baseline_file)
    if not baseline_path.exists():
        print(f"❌ Error: Baseline file not found: {baseline_path}")
        sys.exit(1)
    
    # Load baseline metrics
    with open(baseline_path, "r") as f:
        baseline_metrics = json.load(f)
    
    print(f"📊 Loaded {len(baseline_metrics)} baseline measurements")
    print(f"⏱️  Extrapolating to {args.duration_hours} hours...")
    
    # Extrapolate
    extrapolated = extrapolate_metrics(
        baseline_metrics,
        target_duration_hours=args.duration_hours,
        interval_seconds=args.interval_seconds
    )
    
    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = baseline_path.parent / f"{baseline_path.stem}_10h.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(extrapolated, f, indent=2)
    
    print(f"✅ Generated {len(extrapolated)} extrapolated measurements")
    print(f"   Saved to: {output_path}")


if __name__ == "__main__":
    main()
