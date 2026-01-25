#!/usr/bin/env python
"""
Simple memory monitoring script that monitors a parent process.
Run this as a background process to track memory usage.
"""

import os
import sys
import time
import psutil
from pathlib import Path

def monitor_process(pid: int, interval: float = 60.0, log_file: Path | None = None):
    """Monitor memory usage of a process."""
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"Process {pid} not found")
        return
    
    baseline_mb = process.memory_info().rss / (1024 * 1024)
    start_time = time.time()
    
    print(f"Monitoring process {pid} (baseline: {baseline_mb:.1f} MB)")
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            f.write(f"Time(s),Memory(MB),Delta(MB)\n")
    
    try:
        while True:
            try:
                memory_mb = process.memory_info().rss / (1024 * 1024)
                delta_mb = memory_mb - baseline_mb
                elapsed = time.time() - start_time
                
                msg = f"[MONITOR {elapsed:.0f}s] Memory: {memory_mb:.1f} MB (Δ {delta_mb:+.1f} MB)"
                print(msg)
                
                if log_file:
                    with open(log_file, "a") as f:
                        f.write(f"{elapsed:.1f},{memory_mb:.1f},{delta_mb:.1f}\n")
                
            except psutil.NoSuchProcess:
                print(f"Process {pid} ended")
                break
            except Exception as e:
                print(f"Error monitoring: {e}")
                break
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Monitoring stopped")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <pid> [interval] [log_file]")
        sys.exit(1)
    
    pid = int(sys.argv[1])
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    log_file = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    
    monitor_process(pid, interval, log_file)
