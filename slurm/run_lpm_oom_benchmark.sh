#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=36G
#SBATCH --job-name=lpm-oom-benchmark
#SBATCH --output=/mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow/slurm-logs/%x-%j.err
#SBATCH --export=ALL
#SBATCH --mail-user=adam.kaniasty@gmail.com
#SBATCH --mail-type FAIL,END

set -euo pipefail

REPO_DIR=${REPO_DIR:-"$PWD"}
# Use hkowalski's store which has the prepared datasets
STORE_DIR=${STORE_DIR:-"/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store"}
LOG_DIR=${LOG_DIR:-"$REPO_DIR/slurm-logs"}

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "=========================================="
echo "LPM OOM Benchmark Test - SLURM Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname -s)"
echo "PWD: $(pwd)"
echo "Date: $(date)"
echo "=========================================="

# Respect allocated cores for CPU backends
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Benchmark configuration - use small subset for quick test
# Test with Bielik-4.5B (the problematic model) on a small subset
MODEL="speakleash/Bielik-4.5B-v3.0-Instruct"
TRAIN_DATASET="wgmix_train"
TEST_DATASET="wgmix_test"
AGGREGATION="last_token"
METRIC="euclidean"
LAYER=59

# Limit samples for quick benchmark
MAX_SAMPLES=1000  # Use 1000 training samples
TEST_LIMIT=100   # Use 100 test samples

echo ""
echo "=========================================="
echo "Benchmark Configuration"
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
echo "Benchmark: enabled"
echo "=========================================="
echo ""

# Verify psutil is available
echo "Checking dependencies..."
uv run python -c "import psutil; print(f'✅ psutil {psutil.__version__} available')" || {
    echo "❌ psutil not available - installing..."
    exit 1
}

echo ""
echo "Starting benchmark run..."
echo ""

# Run experiment with memory benchmarking
uv run python -m experiments.scripts.run_lpm_experiment_oom \
  --model "$MODEL" \
  --train-dataset "$TRAIN_DATASET" \
  --test-dataset "$TEST_DATASET" \
  --aggregation "$AGGREGATION" \
  --metric "$METRIC" \
  --layer "$LAYER" \
  --device cpu \
  --batch-size 32 \
  --max-length 512 \
  --max-samples "$MAX_SAMPLES" \
  --test-limit "$TEST_LIMIT" \
  --benchmark \
  --store "$STORE_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "✅ Benchmark completed successfully"
  echo ""
  echo "Check the memory_benchmark.json file in the results directory for memory usage report"
else
  echo ""
  echo "❌ Benchmark failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE