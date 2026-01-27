#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short,long
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=36G
#SBATCH --job-name=probe-retry
#SBATCH --output=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%A-%a.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%A-%a.err
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user hubik112@gmail.com
#SBATCH --array=0-1

# Linear Probe Experiments - Retry Failed Runs
#
# Runs only the 2 failed experiments:
# 1. Bielik-4.5B + plmix_train + plmix_test + last_token_prefix
# 2. Bielik-4.5B + plmix_train + wgmix_test + last_token_prefix
#
# These failed due to BFloat16/Float32 dtype mismatch (now fixed)

set -euo pipefail

REPO_DIR=${REPO_DIR:-"$PWD"}
STORE_DIR=${STORE_DIR:-"$REPO_DIR/store"}
LOG_DIR=${LOG_DIR:-"$REPO_DIR/slurm-logs"}

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "=========================================="
echo "SLURM Array Job: Probe Retry"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname -s)"
echo "PWD: $(pwd)"
echo "=========================================="

# Fixed configuration for all runs
MODEL="speakleash/Bielik-4.5B-v3.0-Instruct"
LAYER=59
TRAIN_DATASET="plmix_train"
AGGREGATION="last_token_prefix"

# Hyperparameters
LR=1e-3
WEIGHT_DECAY=1e-4
BATCH_SIZE=32
MAX_EPOCHS=50
PATIENCE=5

# Determine test dataset based on array index
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    TEST_DATASET="plmix_test"
    TEST_LIMIT=""
else
    TEST_DATASET="wgmix_test"
    TEST_LIMIT="--test-limit 500"
fi

echo "=========================================="
echo "Experiment Configuration"
echo "=========================================="
echo "Model: $MODEL"
echo "Layer: $LAYER"
echo "Train Dataset: $TRAIN_DATASET"
echo "Test Dataset: $TEST_DATASET"
echo "Aggregation: $AGGREGATION"
echo "Learning Rate: $LR"
echo "Weight Decay: $WEIGHT_DECAY"
echo "Batch Size: $BATCH_SIZE"
echo "Max Epochs: $MAX_EPOCHS"
echo "Patience: $PATIENCE"
echo "Test Limit: ${TEST_LIMIT:-None}"
echo "=========================================="
echo ""

# Run experiment
echo "Running probe experiment (retry)..."
uv run python -m experiments.scripts.run_probe_experiment_oom \
    --model "$MODEL" \
    --train-dataset "$TRAIN_DATASET" \
    --test-dataset "$TEST_DATASET" \
    --aggregation "$AGGREGATION" \
    --layer "$LAYER" \
    --learning-rate "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --batch-size "$BATCH_SIZE" \
    --inference-batch-size 32 \
    --max-epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    $TEST_LIMIT \
    --seed 42 \
    --benchmark

echo ""
echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID completed successfully"
echo "=========================================="
