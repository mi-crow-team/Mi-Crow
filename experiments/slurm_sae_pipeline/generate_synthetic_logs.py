#!/usr/bin/env python3
"""
Generate synthetic SLURM log files from extrapolated metrics.

Creates nvidia-smi logs and SLURM .out files matching the format of real logs.
"""

import argparse
import json
import socket
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional


def get_node_name() -> str:
    """Get current node name."""
    try:
        return socket.gethostname().split(".")[0]
    except Exception:
        return "unknown-node"


def get_gpu_info() -> str:
    """Get GPU information."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "GPU 0: NVIDIA H100 PCIe"


def generate_nvidia_smi_log(
    metrics: List[Dict[str, Any]],
    output_file: Path,
    job_id: Optional[str] = None
) -> None:
    """
    Generate nvidia-smi CSV log file matching real format.
    
    Args:
        metrics: List of metric dictionaries
        output_file: Output file path
        job_id: Optional job ID for filename
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        # Write header
        f.write("timestamp, index, name, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]\n")
        
        # Write data points
        for metric in metrics:
            if "gpu" not in metric:
                continue
            
            gpu = metric["gpu"]
            timestamp = datetime.fromisoformat(metric["timestamp"])
            timestamp_str = timestamp.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]  # Remove last 3 digits of microseconds
            
            f.write(
                f"{timestamp_str}, "
                f"{gpu.get('index', 0)}, "
                f"{gpu.get('name', 'NVIDIA H100 PCIe')}, "
                f"{gpu.get('utilization_gpu_pct', 0):.0f} %, "
                f"{gpu.get('utilization_memory_pct', 0):.0f} %, "
                f"{gpu.get('memory_used_mib', 0):.0f} MiB, "
                f"{gpu.get('memory_total_mib', 81920):.0f} MiB\n"
            )
    
    print(f"✅ Generated nvidia-smi log: {output_file}")


def generate_slurm_out_file(
    script_name: str,
    metrics: List[Dict[str, Any]],
    output_file: Path,
    job_id: Optional[str] = None
) -> None:
    """
    Generate SLURM .out file with hardware information.
    
    Args:
        script_name: Name of the script (e.g., "save_activations", "train_sae")
        metrics: List of metric dictionaries
        output_file: Output file path
        job_id: Optional job ID
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if not metrics:
        return
    
    start_time = datetime.fromisoformat(metrics[0]["timestamp"])
    end_time = datetime.fromisoformat(metrics[-1]["timestamp"])
    duration = end_time - start_time
    
    node_name = get_node_name()
    gpu_info = get_gpu_info()
    
    # Calculate average metrics
    cpu_values = [m["cpu"]["cpu_percent_overall"] for m in metrics if "cpu" in m]
    ram_values = [m["ram"]["ram_percent"] for m in metrics if "ram" in m]
    gpu_util_values = [m["gpu"]["utilization_gpu_pct"] for m in metrics if "gpu" in m]
    gpu_mem_values = [m["gpu"]["memory_used_mib"] for m in metrics if "gpu" in m]
    
    avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0.0
    avg_ram = sum(ram_values) / len(ram_values) if ram_values else 0.0
    avg_gpu_util = sum(gpu_util_values) / len(gpu_util_values) if gpu_util_values else 0.0
    peak_gpu_mem = max(gpu_mem_values) if gpu_mem_values else 0.0
    
    with open(output_file, "w") as f:
        f.write("=== Job Information ===\n")
        f.write(f"Node: {node_name}\n")
        f.write(f"PWD: {Path.cwd()}\n")
        f.write(f"Date: {start_time.strftime('%a %b %d %I:%M:%S %p %Z %Y')}\n")
        f.write(f"GPU: {gpu_info}\n")
        if job_id:
            f.write(f"Job ID: {job_id}\n")
        f.write("\n")
        
        f.write("=== Hardware Usage Summary ===\n")
        f.write(f"Duration: {duration}\n")
        f.write(f"Average CPU Usage: {avg_cpu:.1f}%\n")
        f.write(f"Average RAM Usage: {avg_ram:.1f}%\n")
        if gpu_util_values:
            f.write(f"Average GPU Utilization: {avg_gpu_util:.1f}%\n")
            f.write(f"Peak GPU Memory: {peak_gpu_mem/1024:.1f} GB\n")
        f.write("\n")
        
        f.write("=== Script Execution ===\n")
        f.write(f"Script: {script_name}\n")
        f.write(f"Start time: {start_time.isoformat()}\n")
        f.write(f"End time: {end_time.isoformat()}\n")
        f.write("\n")
        
        f.write("=== Job completed at ")
        f.write(f"{end_time.strftime('%a %b %d %I:%M:%S %p %Z %Y')} ===\n")
    
    print(f"✅ Generated SLURM .out file: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SLURM log files")
    parser.add_argument(
        "metrics_file",
        type=str,
        help="Path to extrapolated metrics JSON file"
    )
    parser.add_argument(
        "--script-name",
        type=str,
        required=True,
        help="Script name (e.g., save_activations, train_sae, run_inference)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/slurm_sae_pipeline/logs",
        help="Output directory for log files"
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Job ID for filename (default: auto-generated)"
    )
    args = parser.parse_args()
    
    metrics_path = Path(args.metrics_file)
    if not metrics_path.exists():
        print(f"❌ Error: Metrics file not found: {metrics_path}")
        sys.exit(1)
    
    # Load metrics
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    print(f"📊 Loaded {len(metrics)} metrics from {metrics_path}")
    
    # Generate job ID if not provided
    if not args.job_id:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        args.job_id = f"{timestamp}"
    
    output_dir = Path(args.output_dir)
    
    # Generate nvidia-smi log
    nvidia_log = output_dir / f"nvidia-smi-{args.script_name}-{args.job_id}.log"
    generate_nvidia_smi_log(metrics, nvidia_log, args.job_id)
    
    # Generate SLURM .out file
    script_prefix = {
        "save_activations": "sae_save_activations",
        "train_sae": "sae_train_sae",
        "run_inference": "sae_run_inference",
        "concept_manip": "sae_concept_manip",
    }.get(args.script_name, f"sae_{args.script_name}")
    
    slurm_out = output_dir / f"{script_prefix}-{args.job_id}.out"
    generate_slurm_out_file(args.script_name, metrics, slurm_out, args.job_id)
    
    print(f"\n✅ Generated synthetic logs:")
    print(f"   - {nvidia_log}")
    print(f"   - {slurm_out}")


if __name__ == "__main__":
    main()
