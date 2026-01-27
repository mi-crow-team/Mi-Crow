#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=36G
#SBATCH --job-name=probe-experiments
#SBATCH --output=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%A-%a.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%A-%a.err
#SBATCH --export=ALL
#SBATCH --mail-user hubik112@gmail.com
#SBATCH --mail-type FAIL,END
#SBATCH --array=0-35

# Linear Probe Experiments - Full Matrix
#
# Matrix: 3 models × 2 train datasets × 3 aggregations × 2 test datasets = 36 experiments
#
# Models:
#   - speakleash/Bielik-1.5B-v3.0-Instruct (layer 31)
#   - speakleash/Bielik-4.5B-v3.0-Instruct (layer 59)
#   - meta-llama/Llama-3.2-3B-Instruct (layer 27)
#
# Train Datasets:
#   - wgmix_train (English)
#   - plmix_train (Polish)
#
# Aggregations:
#   - mean
#   - last_token
#   - last_token_prefix
#
# Test Datasets:
#   - Same as train (wgmix_test or plmix_test)
#   - Cross-lingual (plmix_test when train=wgmix, wgmix_test when train=plmix)

set -euo pipefail

REPO_DIR=${REPO_DIR:-"$PWD"}
STORE_DIR=${STORE_DIR:-"$REPO_DIR/store"}
LOG_DIR=${LOG_DIR:-"$REPO_DIR/slurm-logs"}

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "=========================================="
echo "SLURM Array Job: Probe Experiments"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname -s)"
echo "PWD: $(pwd)"
echo "=========================================="

# Configuration arrays
MODELS=(
    "speakleash/Bielik-1.5B-v3.0-Instruct"
    "speakleash/Bielik-4.5B-v3.0-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
)

LAYERS=(31 59 27)

TRAIN_DATASETS=(
    "wgmix_train"
    "plmix_train"
)

AGGREGATIONS=(
    "mean"
    "last_token"
    "last_token_prefix"
)

# Hyperparameters (from user requirements)
LR=1e-3
WEIGHT_DECAY=1e-4
BATCH_SIZE=32
MAX_EPOCHS=50
PATIENCE=5

# Calculate indices
# Total: 3 models × 2 train_datasets × 3 aggregations × 2 test_datasets = 36
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 12))  # 0-2
REMAINDER=$((SLURM_ARRAY_TASK_ID % 12))
TRAIN_DATASET_IDX=$((REMAINDER / 6))  # 0-1
REMAINDER=$((REMAINDER % 6))
AGGREGATION_IDX=$((REMAINDER / 2))  # 0-2
TEST_VARIANT=$((REMAINDER % 2))  # 0=same-language, 1=cross-lingual

MODEL=${MODELS[$MODEL_IDX]}
LAYER=${LAYERS[$MODEL_IDX]}
TRAIN_DATASET=${TRAIN_DATASETS[$TRAIN_DATASET_IDX]}
AGGREGATION=${AGGREGATIONS[$AGGREGATION_IDX]}

# Determine test dataset
if [ $TEST_VARIANT -eq 0 ]; then
    # Same language test
    if [ "$TRAIN_DATASET" == "wgmix_train" ]; then
        TEST_DATASET="wgmix_test"
    else
        TEST_DATASET="plmix_test"
    fi
else
    # Cross-lingual test
    if [ "$TRAIN_DATASET" == "wgmix_train" ]; then
        TEST_DATASET="plmix_test"
    else
        TEST_DATASET="wgmix_test"
    fi
fi

# Apply test limit for wgmix test sets (larger datasets)
TEST_LIMIT=""
if [[ "$TEST_DATASET" == "wgmix_test" ]]; then
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
echo "Running probe experiment..."
uv run python -m experiments.scripts.run_probe_experiment_oom \
    --model "$MODEL" \
    --train-dataset "$TRAIN_DATASET" \
    --test-dataset "$TEST_DATASET" \
    --aggregation "$AGGREGATION" \
    --layer "$LAYER" \
    --learning-rate "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --batch-size "$BATCH_SIZE" \
    --max-epochs "$MAX_EPOCHS" \
    --patience "$PATIENCE" \
    $TEST_LIMIT \
    --seed 42

echo ""
echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID completed successfully"
echo "=========================================="
