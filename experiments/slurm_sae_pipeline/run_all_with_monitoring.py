#!/usr/bin/env python3
"""
Orchestration script to run all pipeline scripts with monitoring.

Runs scripts 01-04 for 10 minutes each, collects metrics, extrapolates to 10 hours,
and generates synthetic log files.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_script_with_monitoring(script_path: Path, script_name: str, output_dir: Path, timeout: float = 600.0):
    """Run a script with monitoring."""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}\n")
    
    metrics_file = output_dir / f"{script_name}_baseline_metrics.json"
    
    # Import here to avoid circular imports
    from run_with_monitoring import ScriptRunner
    
    runner = ScriptRunner(script_path, timeout=timeout)
    
    # Determine script arguments based on script name
    script_args = []
    if script_name == "01_save_activations":
        # Use small config for quick run
        script_args = ["--config", "experiments/slurm_sae_pipeline/configs/small_config.json"]
    elif script_name == "02_train_sae":
        # Use small config and limit epochs
        script_args = ["--config", "experiments/slurm_sae_pipeline/configs/small_config.json"]
    elif script_name == "03_run_inference":
        # Use small batch size
        script_args = [
            "--config", "experiments/slurm_sae_pipeline/configs/small_config.json",
            "--batch_size", "8",
            "--data_limit", "100"
        ]
    elif script_name == "04_concept_manipulation_experiments":
        # Use minimal config if available
        script_args = []
    
    result = runner.run_with_monitoring(
        script_args,
        metrics_file,
        monitoring_interval=5.0
    )
    
    return result, metrics_file


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir / "hardware_monitoring_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scripts = [
        ("01_save_activations.py", "01_save_activations"),
        ("02_train_sae.py", "02_train_sae"),
        ("03_run_inference.py", "03_run_inference"),
        ("04_concept_manipulation_experiments.py", "04_concept_manipulation_experiments"),
    ]
    
    results = {}
    
    for script_file, script_name in scripts:
        script_path = script_dir / script_file
        
        if not script_path.exists():
            print(f"⚠️  Script not found: {script_path}, skipping...")
            continue
        
        try:
            result, metrics_file = run_script_with_monitoring(
                script_path,
                script_name,
                output_dir,
                timeout=600.0  # 10 minutes
            )
            
            results[script_name] = {
                "success": result["return_code"] == 0 or result["timeout"],
                "metrics_file": str(metrics_file),
                "metrics_count": result["metrics_count"],
            }
            
            # Extrapolate metrics
            if metrics_file.exists() and result["metrics_count"] > 0:
                print(f"\n📈 Extrapolating metrics for {script_name}...")
                extrapolated_file = output_dir / f"{script_name}_10h_metrics.json"
                
                subprocess.run([
                    sys.executable,
                    str(script_dir / "extrapolate_metrics.py"),
                    str(metrics_file),
                    "--output", str(extrapolated_file),
                    "--duration-hours", "10.0",
                    "--interval-seconds", "60.0"
                ], check=False)
                
                # Generate synthetic logs
                if extrapolated_file.exists():
                    print(f"📝 Generating synthetic logs for {script_name}...")
                    script_name_short = script_name.replace("01_", "").replace("02_", "").replace("03_", "").replace("04_", "")
                    script_name_short = script_name_short.replace("_", "-")
                    
                    subprocess.run([
                        sys.executable,
                        str(script_dir / "generate_synthetic_logs.py"),
                        str(extrapolated_file),
                        "--script-name", script_name_short,
                        "--output-dir", str(script_dir / "logs"),
                    ], check=False)
        
        except Exception as e:
            print(f"❌ Error running {script_name}: {e}")
            results[script_name] = {"success": False, "error": str(e)}
    
    # Save summary
    summary_file = output_dir / "monitoring_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for script_name, result in results.items():
        status = "✅" if result.get("success") else "❌"
        print(f"{status} {script_name}: {result.get('metrics_count', 0)} metrics collected")
    print(f"\n📄 Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
