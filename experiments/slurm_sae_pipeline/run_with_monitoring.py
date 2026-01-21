#!/usr/bin/env python3
"""
Script runner that executes pipeline scripts with hardware monitoring.

Runs a script for a specified duration while monitoring hardware usage.
"""

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from monitor_hardware import monitor_hardware


class ScriptRunner:
    """Runs a script with hardware monitoring."""
    
    def __init__(self, script_path: Path, timeout: float = 600.0):
        """
        Initialize script runner.
        
        Args:
            script_path: Path to script to run
            timeout: Maximum execution time in seconds (default: 600 = 10 minutes)
        """
        self.script_path = script_path
        self.timeout = timeout
        self.process = None
        self.monitoring_stop = threading.Event()
        self.metrics = []
    
    def run_with_monitoring(
        self,
        script_args: list[str],
        metrics_output: Path,
        monitoring_interval: float = 5.0
    ) -> Dict[str, Any]:
        """
        Run script with hardware monitoring.
        
        Args:
            script_args: Arguments to pass to script (should NOT include monitoring args)
            metrics_output: Path to save metrics JSON
            monitoring_interval: Monitoring collection interval
            
        Returns:
            Dictionary with execution results and metrics
        """
        print(f"🚀 Running script: {self.script_path}")
        print(f"   Args: {' '.join(script_args)}")
        print(f"   Timeout: {self.timeout}s")
        print()
        
        # Start monitoring in background thread
        monitoring_thread = threading.Thread(
            target=self._monitor_background,
            args=(metrics_output, monitoring_interval),
            daemon=True
        )
        monitoring_thread.start()
        
        # Give monitoring a moment to start
        time.sleep(1)
        
        # Run the script (use uv run if available, otherwise python)
        start_time = time.time()
        try:
            # Try to use uv run, fallback to python
            import shutil
            if shutil.which("uv"):
                cmd = ["uv", "run", "python", str(self.script_path)]
            else:
                cmd = [sys.executable, str(self.script_path)]
            
            # Filter out monitoring-specific arguments that scripts don't understand
            # Need to filter both the flag and its value
            filtered_args = []
            skip_next = False
            for i, arg in enumerate(script_args):
                if skip_next:
                    skip_next = False
                    continue
                if arg in ["--timeout", "--metrics-output", "--monitoring-interval"]:
                    skip_next = True  # Skip the next argument (the value)
                    continue
                if arg.startswith("--metrics-output=") or arg.startswith("--timeout=") or arg.startswith("--monitoring-interval="):
                    continue
                filtered_args.append(arg)
            
            self.process = subprocess.Popen(
                cmd + filtered_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for completion or timeout
            try:
                stdout, stderr = self.process.communicate(timeout=self.timeout)
                return_code = self.process.returncode
            except subprocess.TimeoutExpired:
                print(f"⏱️  Script timeout after {self.timeout}s, terminating...")
                self.process.kill()
                stdout, stderr = self.process.communicate()
                return_code = -1
        except Exception as e:
            print(f"❌ Error running script: {e}")
            return_code = -1
            stdout = ""
            stderr = str(e)
        finally:
            elapsed_time = time.time() - start_time
            # Stop monitoring
            self.monitoring_stop.set()
            monitoring_thread.join(timeout=5)
        
        # Load collected metrics
        metrics = []
        if metrics_output.exists():
            try:
                with open(metrics_output, "r") as f:
                    metrics = json.load(f)
            except Exception as e:
                print(f"⚠️  Warning: Could not load metrics: {e}")
        
        result = {
            "script": str(self.script_path),
            "args": script_args,
            "return_code": return_code,
            "elapsed_time": elapsed_time,
            "timeout": elapsed_time >= self.timeout,
            "stdout": stdout,
            "stderr": stderr,
            "metrics_count": len(metrics),
            "metrics_file": str(metrics_output),
        }
        
        return result
    
    def _monitor_background(self, output_file: Path, interval: float):
        """Background monitoring thread."""
        self.metrics = monitor_hardware(
            output_file,
            interval=interval,
            duration=None,
            stop_event=self.monitoring_stop
        )


def main():
    parser = argparse.ArgumentParser(description="Run script with hardware monitoring")
    parser.add_argument(
        "script",
        type=str,
        help="Path to script to run"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Maximum execution time in seconds (default: 600 = 10 minutes)"
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        help="Output path for metrics JSON (default: script_name_metrics.json)"
    )
    parser.add_argument(
        "--monitoring-interval",
        type=float,
        default=5.0,
        help="Monitoring collection interval in seconds (default: 5.0)"
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to script (use -- to separate if needed)"
    )
    args = parser.parse_args()
    
    # Remove '--' separator if present (it separates monitoring args from script args)
    # Filter it out from script_args
    args.script_args = [arg for arg in args.script_args if arg != "--"]
    
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"❌ Error: Script not found: {script_path}")
        sys.exit(1)
    
    if args.metrics_output:
        metrics_output = Path(args.metrics_output)
    else:
        metrics_output = script_path.parent / f"{script_path.stem}_metrics.json"
    
    runner = ScriptRunner(script_path, timeout=args.timeout)
    result = runner.run_with_monitoring(
        args.script_args,
        metrics_output,
        args.monitoring_interval
    )
    
    # Print summary
    print()
    print("=" * 60)
    print("Execution Summary")
    print("=" * 60)
    print(f"Script: {result['script']}")
    print(f"Return code: {result['return_code']}")
    print(f"Elapsed time: {result['elapsed_time']:.1f}s")
    print(f"Timeout: {'Yes' if result['timeout'] else 'No'}")
    print(f"Metrics collected: {result['metrics_count']}")
    print(f"Metrics file: {result['metrics_file']}")
    
    if result['return_code'] != 0:
        print()
        print("STDERR:")
        print(result['stderr'][:500])
        sys.exit(result['return_code'])


if __name__ == "__main__":
    main()
