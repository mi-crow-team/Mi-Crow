#!/usr/bin/env python3
"""
Hardware monitoring script that collects CPU, RAM, VRAM, and GPU metrics.

Collects metrics at regular intervals and saves to JSON format.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import psutil
except ImportError:
    print("Error: psutil not installed. Install with: pip install psutil", file=sys.stderr)
    sys.exit(1)


def get_gpu_metrics() -> Optional[Dict[str, Any]]:
    """
    Get GPU metrics using nvidia-smi.
    
    Returns:
        Dictionary with GPU metrics or None if nvidia-smi not available
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return None
        
        lines = result.stdout.strip().split("\n")
        if not lines or not lines[0]:
            return None
        
        # Parse first GPU (index 0)
        parts = lines[0].split(", ")
        if len(parts) < 6:
            return None
        
        return {
            "index": int(parts[0]),
            "name": parts[1].strip(),
            "utilization_gpu_pct": float(parts[2]),
            "utilization_memory_pct": float(parts[3]),
            "memory_used_mib": float(parts[4]),
            "memory_total_mib": float(parts[5]),
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
        return None


def get_cpu_metrics() -> Dict[str, Any]:
    """
    Get CPU metrics using psutil.
    
    Returns:
        Dictionary with CPU metrics
    """
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    cpu_percent_overall = psutil.cpu_percent(interval=0.1)
    load_avg = psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
    
    return {
        "cpu_percent_overall": cpu_percent_overall,
        "cpu_percent_per_core": cpu_percent,
        "cpu_count": psutil.cpu_count(),
        "load_average": list(load_avg) if load_avg else None,
    }


def get_ram_metrics() -> Dict[str, Any]:
    """
    Get RAM metrics using psutil.
    
    Returns:
        Dictionary with RAM metrics
    """
    mem = psutil.virtual_memory()
    
    return {
        "ram_total_gb": mem.total / (1024 ** 3),
        "ram_used_gb": mem.used / (1024 ** 3),
        "ram_available_gb": mem.available / (1024 ** 3),
        "ram_percent": mem.percent,
    }


def collect_metrics() -> Dict[str, Any]:
    """
    Collect all hardware metrics.
    
    Returns:
        Dictionary with all metrics and timestamp
    """
    timestamp = datetime.now().isoformat()
    
    metrics = {
        "timestamp": timestamp,
        "cpu": get_cpu_metrics(),
        "ram": get_ram_metrics(),
        "gpu": get_gpu_metrics(),
    }
    
    return metrics


def monitor_hardware(
    output_file: Path,
    interval: float = 5.0,
    duration: Optional[float] = None,
    stop_event=None
) -> List[Dict[str, Any]]:
    """
    Monitor hardware continuously and save metrics.
    
    Args:
        output_file: Path to output JSON file
        interval: Collection interval in seconds (default: 5.0)
        duration: Total monitoring duration in seconds (None = until stop_event)
        stop_event: Optional threading.Event to signal stop
        
    Returns:
        List of collected metrics
    """
    metrics_list = []
    start_time = time.time()
    
    print(f"🔍 Starting hardware monitoring...")
    print(f"   Output: {output_file}")
    print(f"   Interval: {interval}s")
    if duration:
        print(f"   Duration: {duration}s")
    print()
    
    try:
        while True:
            # Check if we should stop
            if stop_event and stop_event.is_set():
                break
            
            if duration and (time.time() - start_time) >= duration:
                break
            
            # Collect metrics
            metrics = collect_metrics()
            metrics_list.append(metrics)
            
            # Print status
            gpu_info = ""
            if metrics["gpu"]:
                gpu_info = f" | GPU: {metrics['gpu']['utilization_gpu_pct']:.1f}% | VRAM: {metrics['gpu']['memory_used_mib']/1024:.1f}GB"
            
            print(
                f"[{metrics['timestamp']}] "
                f"CPU: {metrics['cpu']['cpu_percent_overall']:.1f}% | "
                f"RAM: {metrics['ram']['ram_percent']:.1f}%{gpu_info}"
            )
            
            # Wait for next interval
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n⚠️  Monitoring interrupted by user")
    
    # Save metrics to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(metrics_list, f, indent=2)
    
    print(f"\n✅ Monitoring complete. Collected {len(metrics_list)} data points.")
    print(f"   Saved to: {output_file}")
    
    return metrics_list


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor hardware usage")
    parser.add_argument(
        "--output",
        type=str,
        default="hardware_metrics.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Collection interval in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Total monitoring duration in seconds (default: run until interrupted)"
    )
    args = parser.parse_args()
    
    output_file = Path(args.output)
    monitor_hardware(output_file, args.interval, args.duration)


if __name__ == "__main__":
    main()
