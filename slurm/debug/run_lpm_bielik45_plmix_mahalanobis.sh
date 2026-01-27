#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --mem=40G
#SBATCH --job-name=lpm-bielik45-plmix-mahal
#SBATCH --output=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%j.err
#SBATCH --export=ALL
#SBATCH --mail-user hubik112@gmail.com
#SBATCH --mail-type FAIL,END

set -euo pipefail

REPO_DIR=${REPO_DIR:-"$PWD"}
STORE_DIR=${STORE_DIR:-"/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store"}
LOG_DIR=${LOG_DIR:-"$REPO_DIR/slurm-logs"}

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "=========================================="
echo "SLURM Job: LPM Bielik-4.5B PLMix Mahalanobis (Fix)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname -s)"
echo "PWD: $(pwd)"
echo "Date: $(date)"
echo "=========================================="

# Configuration - the failing experiment
MODEL="speakleash/Bielik-4.5B-v3.0-Instruct"
TRAIN_DATASET="plmix_train"
TEST_DATASET="plmix_test"
AGGREGATION="last_token_prefix"
METRIC="mahalanobis"
LAYER=59

echo ""
echo "=========================================="
echo "Experiment Configuration"
echo "=========================================="
echo "Model:       $MODEL"
echo "Train Set:   $TRAIN_DATASET"
echo "Test Set:    $TEST_DATASET"
echo "Aggregation: $AGGREGATION"
echo "Metric:      $METRIC"
echo "Layer:       $LAYER"
echo "Batch Size:  32"
echo "Device:      cpu"
echo "Benchmark:   enabled"
echo "=========================================="
echo ""

# Respect allocated cores for CPU backends
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "Starting experiment (with bfloat16 fix)..."
echo ""

# Run experiment using the OOM-optimized script
uv run python -m experiments.scripts.run_lpm_experiment_oom \
  --model "$MODEL" \
  --train-dataset "$TRAIN_DATASET" \
  --test-dataset "$TEST_DATASET" \
  --aggregation "$AGGREGATION" \
  --metric "$METRIC" \
  --layer "$LAYER" \
  --device cpu \
  --batch-size 32 \
  --benchmark \
  --store "$STORE_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "✅ Experiment completed successfully"
  echo "   Results should now include:"
  echo "   - LPM model with Mahalanobis precision matrix"
  echo "   - Predictions on test set"
  echo "   - Evaluation metrics"
  echo "   - Memory benchmark report"
else
  echo ""
  echo "❌ Experiment failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
