#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=36G
#SBATCH --job-name=lpm-oom-benchmark
#SBATCH --output=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%j.err
#SBATCH --export=ALL
#SBATCH --mail-user=hubik112@gmail.com
#SBATCH --mail-type FAIL,END

set -euo pipefail

REPO_DIR=${REPO_DIR:-"$PWD"}
# Use hkowalski's store which has the prepared datasets
STORE_DIR=${STORE_DIR:-"/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store"}
LOG_DIR=${LOG_DIR:-"$REPO_DIR/slurm-logs"}

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "=========================================="
echo "LPM Baseline Benchmark Test - SLURM Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname -s)"
echo "PWD: $(pwd)"
echo "Date: $(date)"
echo "=========================================="

# Respect allocated cores for CPU backends
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Benchmark configuration - use same config as OOM version for comparison
# Test with Bielik-4.5B (the problematic model) on a small subset
MODEL="speakleash/Bielik-4.5B-v3.0-Instruct"
TRAIN_DATASET="wgmix_train"
TEST_DATASET="wgmix_test"
AGGREGATION="last_token"
METRIC="euclidean"
LAYER=59

# Limit samples for quick benchmark (same as OOM version)
MAX_SAMPLES=1000  # Use 1000 training samples
TEST_LIMIT=100   # Use 100 test samples

echo ""
echo "=========================================="
echo "Baseline Benchmark Configuration"
echo "=========================================="
echo "Model: $MODEL"
echo "Train Dataset: $TRAIN_DATASET"
echo "Test Dataset: $TEST_DATASET"
echo "Aggregation: $AGGREGATION"
echo "Metric: $METRIC"
echo "Layer: $LAYER"
echo "Max Training Samples: $MAX_SAMPLES"
echo "Test Limit: $TEST_LIMIT"
echo "Device: cpu"
echo "Script: run_lpm_experiment.py (BASELINE - no OOM fixes)"
echo "=========================================="
echo ""

echo "Starting baseline benchmark run (original script, no OOM fixes)..."
echo ""

# Start memory monitoring using psutil directly
MONITOR_LOG="$LOG_DIR/memory_monitor_${SLURM_JOB_ID}.csv"
MONITOR_INTERVAL=60  # seconds

# Create monitoring script that finds the Python process
cat > /tmp/monitor_${SLURM_JOB_ID}.py << 'MONITOR_EOF'
import os
import sys
import time
import psutil
from pathlib import Path

search_cmd = sys.argv[1]  # Command line pattern to search for
interval = float(sys.argv[2])
log_file = Path(sys.argv[3])
wait_time = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0

print(f"[MONITOR] Waiting {wait_time}s for Python process to start...", flush=True)
sys.stdout.flush()
time.sleep(wait_time)

# Find Python process by command line - be more specific
target_pid = None
candidates = []
for proc in psutil.process_iter(['pid', 'cmdline', 'memory_info']):
    try:
        cmdline = ' '.join(proc.info['cmdline'] or [])
        # Look for processes that contain the search command and are Python processes
        if search_cmd in cmdline and ('python' in cmdline.lower() or 'run_lpm_experiment' in cmdline):
            mem_mb = proc.info['memory_info'].rss / (1024 * 1024)
            candidates.append((proc.info['pid'], cmdline, mem_mb))
    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
        continue

# Prefer processes with more memory (likely the actual Python process, not uv wrapper)
if candidates:
    # Sort by memory (descending) and take the one with most memory
    candidates.sort(key=lambda x: x[2], reverse=True)
    target_pid, cmdline, mem_mb = candidates[0]
    print(f"[MONITOR] Found {len(candidates)} candidate process(es)")
    for pid, cmd, mem in candidates[:3]:  # Show top 3
        print(f"[MONITOR]   PID {pid}: {mem:.1f} MB - {cmd[:80]}")
    print(f"[MONITOR] Selected PID {target_pid} ({mem_mb:.1f} MB): {cmdline[:100]}")

if target_pid is None:
    print(f"[MONITOR] ERROR: Could not find Python process with command containing '{search_cmd}'")
    sys.exit(1)

try:
    process = psutil.Process(target_pid)
except psutil.NoSuchProcess:
    print(f"Process {target_pid} not found")
    sys.exit(1)

baseline_mb = process.memory_info().rss / (1024 * 1024)
start_time = time.time()

print(f"[MONITOR] Monitoring process {target_pid} (baseline: {baseline_mb:.1f} MB)", flush=True)
sys.stdout.flush()
if log_file:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as f:
        f.write("Time(s),Memory(MB),Delta(MB)\n")

try:
    while True:
        try:
            memory_mb = process.memory_info().rss / (1024 * 1024)
            delta_mb = memory_mb - baseline_mb
            elapsed = time.time() - start_time
            
            msg = f"[MONITOR {elapsed:.0f}s] Memory: {memory_mb:.1f} MB (Δ {delta_mb:+.1f} MB)"
            print(msg, flush=True)
            
            if log_file:
                with open(log_file, "a") as f:
                    f.write(f"{elapsed:.1f},{memory_mb:.1f},{delta_mb:.1f}\n")
            
        except psutil.NoSuchProcess:
            print(f"Process {target_pid} ended")
            break
        except Exception as e:
            print(f"Error monitoring: {e}")
            break
        
        time.sleep(interval)
except KeyboardInterrupt:
    print("Monitoring stopped")
MONITOR_EOF

# Cleanup function
cleanup() {
    kill $MONITOR_PID 2>/dev/null || true
    rm -f /tmp/monitor_${SLURM_JOB_ID}.py
}
trap cleanup EXIT

# Start monitoring in background (it will find the Python process)
echo "Starting memory monitor (will find Python process)..."
uv run python /tmp/monitor_${SLURM_JOB_ID}.py "run_lpm_experiment" $MONITOR_INTERVAL "$MONITOR_LOG" 5.0 > "$LOG_DIR/monitor_${SLURM_JOB_ID}.out" 2>&1 &
MONITOR_PID=$!

echo "Memory monitoring started (Monitor PID: $MONITOR_PID, log: $MONITOR_LOG)"
echo ""

# Run experiment with original script (no OOM fixes, but with sample limits)
uv run python -m experiments.scripts.run_lpm_experiment \
  --model "$MODEL" \
  --train-dataset "$TRAIN_DATASET" \
  --test-dataset "$TEST_DATASET" \
  --aggregation "$AGGREGATION" \
  --metric "$METRIC" \
  --layer "$LAYER" \
  --device cpu \
  --batch-size 64 \
  --max-length 512 \
  --max-samples "$MAX_SAMPLES" \
  --test-limit "$TEST_LIMIT" \
  --store "$STORE_DIR"

EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true
sleep 1
rm -f /tmp/monitor_${SLURM_JOB_ID}.py

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "✅ Baseline benchmark completed successfully"
  echo ""
  echo "Memory monitoring log: $MONITOR_LOG"
  echo "Monitor output: $LOG_DIR/monitor_${SLURM_JOB_ID}.out"
  echo "Check SLURM accounting with: sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,MaxVMSize,Elapsed"
  echo ""
  echo "To compare with OOM version, check:"
  echo "  - Baseline: $MONITOR_LOG"
  echo "  - OOM version: store/*/memory_benchmark.json"
else
  echo ""
  echo "❌ Baseline benchmark failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE