#!/usr/bin/env python3
"""
Performance Analysis Script for SAE Pipeline

Analyzes performance metrics from all pipeline steps:
1. Activation saving (2 jobs)
2. SAE training (4 jobs)
3. Inference (2 jobs)

Outputs JSON report and visualization plots.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def parse_slurm_logs(log_file: Path) -> Dict[str, Any]:
    """
    Parse SLURM log files (.out or .err) to extract job information.
    
    Args:
        log_file: Path to SLURM log file
        
    Returns:
        Dictionary with parsed metrics
    """
    if not log_file.exists():
        return {}
    
    metrics = {
        "job_id": None,
        "node": None,
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        "gpu_model": None,
        "errors": [],
        "batch_times": [],
        "memory_logs": [],
        "cuda_memory_logs": [],
    }
    
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            lines = content.split("\n")
        
        # Extract job ID from filename
        job_match = re.search(r"-(\d+)\.(out|err)$", str(log_file))
        if job_match:
            metrics["job_id"] = int(job_match.group(1))
        
        # Extract node information
        node_match = re.search(r"Node: (\S+)", content)
        if node_match:
            metrics["node"] = node_match.group(1)
        
        # Extract GPU model
        gpu_match = re.search(r"GPU.*?:\s*(.+?)(?:\n|$)", content)
        if gpu_match:
            metrics["gpu_model"] = gpu_match.group(1).strip()
        
        # Extract dates/times
        date_matches = re.findall(r"Date: (.+)", content)
        if date_matches:
            try:
                metrics["start_time"] = date_matches[0]
                if len(date_matches) > 1:
                    metrics["end_time"] = date_matches[-1]
            except Exception:
                pass
        
        # Extract batch processing times from inference logs
        batch_time_pattern = r"\[DEBUG\] Inference completed in ([\d.]+)s"
        batch_times = re.findall(batch_time_pattern, content)
        if batch_times:
            metrics["batch_times"] = [float(t) for t in batch_times]
        
        # Extract CUDA memory logs
        cuda_memory_pattern = r"💾 CUDA memory: ([\d.]+) GB allocated, ([\d.]+) GB reserved"
        cuda_matches = re.findall(cuda_memory_pattern, content)
        if cuda_matches:
            metrics["cuda_memory_logs"] = [
                {"allocated_gb": float(m[0]), "reserved_gb": float(m[1])}
                for m in cuda_matches
            ]
        
        # Extract processed batch counts
        processed_pattern = r"Processed (\d+) batches"
        processed_matches = re.findall(processed_pattern, content)
        if processed_matches:
            metrics["batches_processed"] = [int(b) for b in processed_matches]
        
        # Extract errors
        error_lines = [line for line in lines if "error" in line.lower() or "Error" in line or "ERROR" in line]
        if error_lines:
            metrics["errors"] = error_lines[:10]  # Limit to first 10 errors
        
    except Exception as e:
        print(f"Warning: Failed to parse {log_file}: {e}", file=sys.stderr)
    
    return metrics


def parse_nvidia_smi_logs(log_file: Path) -> Optional[pd.DataFrame]:
    """
    Parse nvidia-smi CSV log files to extract GPU metrics.
    
    Args:
        log_file: Path to nvidia-smi log file
        
    Returns:
        DataFrame with GPU metrics or None if parsing fails
    """
    if not log_file.exists():
        return None
    
    try:
        # Read CSV, handling potential formatting issues
        df = pd.read_csv(log_file, on_bad_lines="skip")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Parse timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        
        # Extract numeric values from memory columns
        if "memory.used [MiB]" in df.columns:
            df["memory_used_mib"] = pd.to_numeric(
                df["memory.used [MiB]"].astype(str).str.replace(r"[^\d.]", "", regex=True),
                errors="coerce"
            )
        
        if "memory.total [MiB]" in df.columns:
            df["memory_total_mib"] = pd.to_numeric(
                df["memory.total [MiB]"].astype(str).str.replace(r"[^\d.]", "", regex=True),
                errors="coerce"
            )
        
        # Extract utilization percentages
        if "utilization.gpu [%]" in df.columns:
            df["gpu_utilization_pct"] = pd.to_numeric(
                df["utilization.gpu [%]"].astype(str).str.replace(r"[^\d.]", "", regex=True),
                errors="coerce"
            )
        
        if "utilization.memory [%]" in df.columns:
            df["memory_utilization_pct"] = pd.to_numeric(
                df["utilization.memory [%]"].astype(str).str.replace(r"[^\d.]", "", regex=True),
                errors="coerce"
            )
        
        # Calculate memory usage percentage
        if "memory_used_mib" in df.columns and "memory_total_mib" in df.columns:
            df["memory_usage_pct"] = (df["memory_used_mib"] / df["memory_total_mib"]) * 100
        
        return df
    
    except Exception as e:
        print(f"Warning: Failed to parse nvidia-smi log {log_file}: {e}", file=sys.stderr)
        return None


def load_training_metadata(run_dir: Path) -> Dict[str, Any]:
    """
    Load training metadata and history from SAE run directory.
    
    Args:
        run_dir: Path to SAE run directory
        
    Returns:
        Dictionary with training metrics
    """
    metrics = {
        "run_id": run_dir.name,
        "meta": None,
        "history": None,
        "model_params": {},
    }
    
    meta_file = run_dir / "meta.json"
    history_file = run_dir / "history.json"
    
    # Load meta.json
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
                metrics["meta"] = meta
                
                # Extract model parameters
                if "training_config" in meta:
                    metrics["model_params"]["training_config"] = meta["training_config"]
                
                if "final_metrics" in meta:
                    metrics["model_params"]["final_metrics"] = meta["final_metrics"]
                
                if "layer_signature" in meta:
                    metrics["model_params"]["layer_signature"] = meta["layer_signature"]
                
                if "sae_type" in meta:
                    metrics["model_params"]["sae_type"] = meta["sae_type"]
        except Exception as e:
            print(f"Warning: Failed to load {meta_file}: {e}", file=sys.stderr)
    
    # Load history.json
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
                metrics["history"] = history
        except Exception as e:
            print(f"Warning: Failed to load {history_file}: {e}", file=sys.stderr)
    
    return metrics


def extract_inference_metrics(log_file: Path) -> Dict[str, Any]:
    """
    Extract inference-specific metrics from log files.
    
    Args:
        log_file: Path to inference log file
        
    Returns:
        Dictionary with inference metrics
    """
    metrics = {
        "total_batches": 0,
        "batch_times": [],
        "memory_usage": [],
        "samples_processed": 0,
    }
    
    if not log_file.exists():
        return metrics
    
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        # Extract batch times
        batch_time_pattern = r"\[DEBUG\] Inference completed in ([\d.]+)s"
        batch_times = re.findall(batch_time_pattern, content)
        if batch_times:
            metrics["batch_times"] = [float(t) for t in batch_times]
            metrics["total_batches"] = len(batch_times)
        
        # Extract memory usage
        memory_pattern = r"💾 CUDA memory: ([\d.]+) GB allocated"
        memory_matches = re.findall(memory_pattern, content)
        if memory_matches:
            metrics["memory_usage"] = [float(m) for m in memory_matches]
        
        # Extract batch counts
        batch_count_pattern = r"Got batch (\d+)"
        batch_counts = re.findall(batch_count_pattern, content)
        if batch_counts:
            metrics["total_batches"] = max([int(b) for b in batch_counts], default=0)
        
    except Exception as e:
        print(f"Warning: Failed to extract inference metrics from {log_file}: {e}", file=sys.stderr)
    
    return metrics


def analyze_activation_saving(
    log_dir: Path,
    store_dir: Path,
    activation_runs: List[str]
) -> Dict[str, Any]:
    """
    Analyze activation saving jobs.
    
    Args:
        log_dir: Directory containing SLURM logs
        store_dir: Directory containing run data
        activation_runs: List of activation run IDs
        
    Returns:
        Dictionary with activation saving metrics
    """
    results = {}
    
    for run_id in activation_runs:
        # Find log files for this run
        log_files = list(log_dir.glob(f"sae_save_activations-*.out"))
        log_files.extend(log_dir.glob(f"sae_save_activations-*.err"))
        
        run_metrics = {
            "run_id": run_id,
            "logs": [],
            "gpu_metrics": None,
        }
        
        # Parse each log file (try to match by checking content)
        for log_file in log_files:
            parsed = parse_slurm_logs(log_file)
            if parsed:
                # Check if this log mentions the run_id
                try:
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if run_id in content or not run_metrics["logs"]:
                            run_metrics["logs"].append(parsed)
                except Exception:
                    if not run_metrics["logs"]:
                        run_metrics["logs"].append(parsed)
        
        # Try to find nvidia-smi log
        nvidia_logs = list(log_dir.glob(f"nvidia-smi-*.log"))
        for nvidia_log in nvidia_logs:
            gpu_df = parse_nvidia_smi_logs(nvidia_log)
            if gpu_df is not None and not gpu_df.empty:
                run_metrics["gpu_metrics"] = {
                    "peak_memory_mib": float(gpu_df["memory_used_mib"].max()) if "memory_used_mib" in gpu_df.columns else None,
                    "avg_gpu_utilization": float(gpu_df["gpu_utilization_pct"].mean()) if "gpu_utilization_pct" in gpu_df.columns else None,
                    "avg_memory_utilization": float(gpu_df["memory_utilization_pct"].mean()) if "memory_utilization_pct" in gpu_df.columns else None,
                    "data_points": len(gpu_df),
                }
                break
        
        results[run_id] = run_metrics
    
    return results


def analyze_sae_training(
    log_dir: Path,
    store_dir: Path,
    sae_runs: List[str]
) -> Dict[str, Any]:
    """
    Analyze SAE training jobs.
    
    Args:
        log_dir: Directory containing SLURM logs
        store_dir: Directory containing run data
        sae_runs: List of SAE run IDs
        
    Returns:
        Dictionary with SAE training metrics
    """
    results = {}
    
    for run_id in sae_runs:
        run_dir = store_dir / "runs" / run_id
        if not run_dir.exists():
            continue
        
        run_metrics = {
            "run_id": run_id,
            "training_data": load_training_metadata(run_dir),
            "logs": [],
            "gpu_metrics": None,
        }
        
        # Find log files
        log_files = list(log_dir.glob(f"sae_train_sae-*.out"))
        log_files.extend(log_dir.glob(f"sae_train_sae-*.err"))
        
        for log_file in log_files:
            parsed = parse_slurm_logs(log_file)
            if parsed:
                # Check if this log mentions the run_id
                try:
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if run_id in content or not run_metrics["logs"]:
                            run_metrics["logs"].append(parsed)
                except Exception:
                    if not run_metrics["logs"]:
                        run_metrics["logs"].append(parsed)
        
        # Try to find nvidia-smi log
        nvidia_logs = list(log_dir.glob(f"nvidia-smi-*.log"))
        for nvidia_log in nvidia_logs:
            gpu_df = parse_nvidia_smi_logs(nvidia_log)
            if gpu_df is not None and not gpu_df.empty:
                run_metrics["gpu_metrics"] = {
                    "peak_memory_mib": float(gpu_df["memory_used_mib"].max()) if "memory_used_mib" in gpu_df.columns else None,
                    "avg_gpu_utilization": float(gpu_df["gpu_utilization_pct"].mean()) if "gpu_utilization_pct" in gpu_df.columns else None,
                    "avg_memory_utilization": float(gpu_df["memory_utilization_pct"].mean()) if "memory_utilization_pct" in gpu_df.columns else None,
                    "data_points": len(gpu_df),
                }
                break
        
        results[run_id] = run_metrics
    
    return results


def analyze_inference(
    log_dir: Path,
    store_dir: Path,
    job_ids: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Analyze inference jobs.
    
    Args:
        log_dir: Directory containing SLURM logs
        store_dir: Directory containing run data
        job_ids: Optional list of specific job IDs to analyze
        
    Returns:
        Dictionary with inference metrics
    """
    results = {}
    
    # Find inference log files
    log_files = list(log_dir.glob(f"sae_run_inference-*.out"))
    log_files.extend(log_dir.glob(f"sae_run_inference-*.err"))
    
    # Filter by job IDs if provided
    if job_ids:
        log_files = [f for f in log_files if any(f"_{jid}." in str(f) for jid in job_ids)]
    
    # Sort by modification time (most recent first)
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Analyze up to 2 most recent jobs
    for log_file in log_files[:2]:
        job_match = re.search(r"-(\d+)\.(out|err)$", str(log_file))
        if not job_match:
            continue
        
        job_id = int(job_match.group(1))
        
        run_metrics = {
            "job_id": job_id,
            "logs": parse_slurm_logs(log_file),
            "inference_metrics": extract_inference_metrics(log_file),
            "gpu_metrics": None,
        }
        
        # Try to find nvidia-smi log for this job
        nvidia_log = log_dir / f"nvidia-smi-{job_id}.log"
        if not nvidia_log.exists():
            # Try alternative pattern
            nvidia_logs = list(log_dir.glob(f"nvidia-smi-*.log"))
            if nvidia_logs:
                nvidia_log = nvidia_logs[0]  # Use first available
        
        if nvidia_log.exists():
            gpu_df = parse_nvidia_smi_logs(nvidia_log)
            if gpu_df is not None and not gpu_df.empty:
                run_metrics["gpu_metrics"] = {
                    "peak_memory_mib": float(gpu_df["memory_used_mib"].max()) if "memory_used_mib" in gpu_df.columns else None,
                    "peak_memory_gb": float(gpu_df["memory_used_mib"].max() / 1024) if "memory_used_mib" in gpu_df.columns else None,
                    "avg_gpu_utilization": float(gpu_df["gpu_utilization_pct"].mean()) if "gpu_utilization_pct" in gpu_df.columns else None,
                    "avg_memory_utilization": float(gpu_df["memory_utilization_pct"].mean()) if "memory_utilization_pct" in gpu_df.columns else None,
                    "memory_over_time": gpu_df[["timestamp", "memory_used_mib"]].to_dict("records") if "timestamp" in gpu_df.columns and "memory_used_mib" in gpu_df.columns else None,
                    "data_points": len(gpu_df),
                }
        
        results[f"job_{job_id}"] = run_metrics
    
    return results


def plot_gpu_memory(all_metrics: Dict[str, Any], output_dir: Path) -> None:
    """Plot GPU memory usage over time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot inference GPU memory
    ax1 = axes[0]
    for job_name, job_data in all_metrics.get("inference", {}).items():
        gpu_metrics = job_data.get("gpu_metrics", {})
        memory_over_time = gpu_metrics.get("memory_over_time")
        
        if memory_over_time:
            df = pd.DataFrame(memory_over_time)
            if "timestamp" in df.columns and "memory_used_mib" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.dropna(subset=["timestamp", "memory_used_mib"])
                df["memory_gb"] = df["memory_used_mib"] / 1024
                ax1.plot(df["timestamp"], df["memory_gb"], label=f"{job_name}", marker="o", markersize=3)
    
    ax1.set_xlabel("Time")
    ax1.set_ylabel("GPU Memory (GB)")
    ax1.set_title("GPU Memory Usage During Inference")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot comparison across all steps
    ax2 = axes[1]
    categories = []
    peak_memories = []
    
    # Add inference jobs
    for job_name, job_data in all_metrics.get("inference", {}).items():
        gpu_metrics = job_data.get("gpu_metrics", {})
        peak_gb = gpu_metrics.get("peak_memory_gb")
        if peak_gb:
            categories.append(f"Inference\n{job_name}")
            peak_memories.append(peak_gb)
    
    # Add training jobs
    for run_id, run_data in all_metrics.get("training", {}).items():
        gpu_metrics = run_data.get("gpu_metrics", {})
        peak_mib = gpu_metrics.get("peak_memory_mib")
        if peak_mib:
            categories.append(f"Training\n{run_id.split('_')[-2]}")
            peak_memories.append(peak_mib / 1024)
    
    if categories:
        ax2.bar(categories, peak_memories, color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"][:len(categories)])
        ax2.set_ylabel("Peak GPU Memory (GB)")
        ax2.set_title("Peak GPU Memory Usage Comparison")
        ax2.grid(True, alpha=0.3, axis="y")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_dir / "gpu_memory_usage.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(all_metrics: Dict[str, Any], output_dir: Path) -> None:
    """Plot training curves (loss, R², sparsity metrics)."""
    training_data = all_metrics.get("training", {})
    if not training_data:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss curves
    ax1 = axes[0, 0]
    for run_id, run_data in training_data.items():
        history = run_data.get("training_data", {}).get("history", {})
        if history and "loss" in history:
            epochs = range(1, len(history["loss"]) + 1)
            layer_name = run_id.split("_")[-2] if "_" in run_id else run_id
            ax1.plot(epochs, history["loss"], label=f"Layer {layer_name}", marker="o", markersize=4)
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # R² curves
    ax2 = axes[0, 1]
    for run_id, run_data in training_data.items():
        history = run_data.get("training_data", {}).get("history", {})
        if history and "r2" in history:
            epochs = range(1, len(history["r2"]) + 1)
            layer_name = run_id.split("_")[-2] if "_" in run_id else run_id
            ax2.plot(epochs, history["r2"], label=f"Layer {layer_name}", marker="s", markersize=4)
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("R² Score")
    ax2.set_title("R² Score Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # L1 sparsity
    ax3 = axes[1, 0]
    for run_id, run_data in training_data.items():
        history = run_data.get("training_data", {}).get("history", {})
        if history and "l1" in history:
            epochs = range(1, len(history["l1"]) + 1)
            layer_name = run_id.split("_")[-2] if "_" in run_id else run_id
            ax3.plot(epochs, history["l1"], label=f"Layer {layer_name}", marker="^", markersize=4)
    
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("L1 Sparsity")
    ax3.set_title("L1 Sparsity Curves")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Dead features percentage
    ax4 = axes[1, 1]
    for run_id, run_data in training_data.items():
        history = run_data.get("training_data", {}).get("history", {})
        if history and "dead_features_pct" in history:
            epochs = range(1, len(history["dead_features_pct"]) + 1)
            layer_name = run_id.split("_")[-2] if "_" in run_id else run_id
            ax4.plot(epochs, history["dead_features_pct"], label=f"Layer {layer_name}", marker="d", markersize=4)
    
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Dead Features (%)")
    ax4.set_title("Dead Features Percentage")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_batch_times(all_metrics: Dict[str, Any], output_dir: Path) -> None:
    """Plot batch processing times."""
    inference_data = all_metrics.get("inference", {})
    if not inference_data:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for job_name, job_data in inference_data.items():
        batch_times = job_data.get("inference_metrics", {}).get("batch_times", [])
        if batch_times:
            batches = range(1, len(batch_times) + 1)
            ax.plot(batches, batch_times, label=f"{job_name}", marker="o", markersize=4, alpha=0.7)
    
    ax.set_xlabel("Batch Number")
    ax.set_ylabel("Processing Time (seconds)")
    ax.set_title("Batch Processing Times During Inference")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "batch_times.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_comparison(all_metrics: Dict[str, Any], output_dir: Path) -> None:
    """Compare models by parameters and metrics."""
    training_data = all_metrics.get("training", {})
    if not training_data:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract model parameters
    run_ids = []
    final_losses = []
    final_r2 = []
    dead_features = []
    epochs = []
    
    for run_id, run_data in training_data.items():
        meta = run_data.get("training_data", {}).get("meta", {})
        if meta:
            run_ids.append(run_id.split("_")[-2] if "_" in run_id else run_id)
            
            final_metrics = meta.get("final_metrics", {})
            final_losses.append(final_metrics.get("loss", 0))
            final_r2.append(final_metrics.get("r2", 0))
            dead_features.append(final_metrics.get("dead_features_pct", 0))
            epochs.append(meta.get("n_epochs", 0))
    
    if run_ids:
        # Final loss comparison
        ax1 = axes[0, 0]
        ax1.bar(run_ids, final_losses, color="#e74c3c")
        ax1.set_ylabel("Final Loss")
        ax1.set_title("Final Training Loss Comparison")
        ax1.grid(True, alpha=0.3, axis="y")
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # Final R² comparison
        ax2 = axes[0, 1]
        ax2.bar(run_ids, final_r2, color="#2ecc71")
        ax2.set_ylabel("Final R² Score")
        ax2.set_title("Final R² Score Comparison")
        ax2.grid(True, alpha=0.3, axis="y")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # Dead features comparison
        ax3 = axes[1, 0]
        ax3.bar(run_ids, dead_features, color="#f39c12")
        ax3.set_ylabel("Dead Features (%)")
        ax3.set_title("Dead Features Percentage Comparison")
        ax3.grid(True, alpha=0.3, axis="y")
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # Training epochs
        ax4 = axes[1, 1]
        ax4.bar(run_ids, epochs, color="#3498db")
        ax4.set_ylabel("Epochs")
        ax4.set_title("Training Epochs")
        ax4.grid(True, alpha=0.3, axis="y")
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def aggregate_metrics(
    activation_metrics: Dict[str, Any],
    training_metrics: Dict[str, Any],
    inference_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Aggregate all metrics into a comprehensive report.
    
    Args:
        activation_metrics: Metrics from activation saving
        training_metrics: Metrics from SAE training
        inference_metrics: Metrics from inference
        
    Returns:
        Aggregated metrics dictionary
    """
    return {
        "analysis_timestamp": datetime.now().isoformat(),
        "activation_saving": activation_metrics,
        "training": training_metrics,
        "inference": inference_metrics,
        "summary": {
            "total_activation_jobs": len(activation_metrics),
            "total_training_jobs": len(training_metrics),
            "total_inference_jobs": len(inference_metrics),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze performance metrics from SAE pipeline")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="experiments/slurm_sae_pipeline/logs",
        help="Directory containing SLURM logs"
    )
    parser.add_argument(
        "--store_dir",
        type=str,
        default="experiments/slurm_sae_pipeline/store",
        help="Directory containing run data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/slurm_sae_pipeline/performance_analysis",
        help="Output directory for results"
    )
    parser.add_argument(
        "--inference_job_ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific inference job IDs to analyze (default: most recent 2)"
    )
    args = parser.parse_args()
    
    # Convert to Path objects
    log_dir = Path(args.log_dir)
    store_dir = Path(args.store_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots subdirectory
    plots_dir = output_dir / f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Analyzing SAE Pipeline Performance Metrics...")
    print(f"📁 Log directory: {log_dir}")
    print(f"📁 Store directory: {store_dir}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Define runs to analyze
    activation_runs = ["activations_bielik12", "activations_bielik45"]
    sae_runs = [
        "sae_llamaforcausallm_model_layers_15_20260118_150004",
        "sae_llamaforcausallm_model_layers_20_20260118_150004",
        "sae_llamaforcausallm_model_layers_28_20260118_144434",
        "sae_llamaforcausallm_model_layers_38_20260118_144434",
    ]
    
    # Analyze each step
    print("📊 Analyzing activation saving jobs...")
    activation_metrics = analyze_activation_saving(log_dir, store_dir, activation_runs)
    
    print("📊 Analyzing SAE training jobs...")
    training_metrics = analyze_sae_training(log_dir, store_dir, sae_runs)
    
    print("📊 Analyzing inference jobs...")
    inference_metrics = analyze_inference(log_dir, store_dir, args.inference_job_ids)
    
    # Aggregate metrics
    print("📈 Aggregating metrics...")
    all_metrics = aggregate_metrics(activation_metrics, training_metrics, inference_metrics)
    
    # Generate visualizations
    print("📊 Generating visualizations...")
    plot_gpu_memory(all_metrics, plots_dir)
    plot_training_curves(all_metrics, plots_dir)
    plot_batch_times(all_metrics, plots_dir)
    plot_model_comparison(all_metrics, plots_dir)
    
    # Save JSON report
    json_file = output_dir / f"performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    print(f"💾 Saving JSON report to {json_file}...")
    with open(json_file, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    # Print summary
    print()
    print("✅ Analysis complete!")
    print(f"📄 JSON report: {json_file}")
    print(f"📊 Plots directory: {plots_dir}")
    print()
    print("Summary:")
    print(f"  - Activation jobs analyzed: {all_metrics['summary']['total_activation_jobs']}")
    print(f"  - Training jobs analyzed: {all_metrics['summary']['total_training_jobs']}")
    print(f"  - Inference jobs analyzed: {all_metrics['summary']['total_inference_jobs']}")


if __name__ == "__main__":
    main()
