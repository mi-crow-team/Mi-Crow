#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short,long
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=36G
#SBATCH --job-name=lpm-experiments
#SBATCH --output=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%A-%a.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%A-%a.err
#SBATCH --export=ALL
#SBATCH --mail-user hubik112@gmail.com
#SBATCH --mail-type FAIL,END
#SBATCH --array=0-35

set -euo pipefail

REPO_DIR=${REPO_DIR:-"$PWD"}
STORE_DIR=${STORE_DIR:-"$REPO_DIR/store"}
LOG_DIR=${LOG_DIR:-"$REPO_DIR/slurm-logs"}

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "=========================================="
echo "SLURM Array Job: LPM Experiments"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname -s)"
echo "PWD: $(pwd)"
echo "=========================================="

# Define experiment grid
# Models: 3 (Llama-3B, Bielik-1.5B, Bielik-4.5B)
# Datasets: 2 pairs (wgmix, plmix)
# Aggregations: 3 (mean, last_token, last_token_prefix)
# Metrics: 2 (euclidean, mahalanobis)
# Total: 3 * 2 * 3 * 2 = 36 experiments

# Model configurations
declare -a MODELS=(
  "meta-llama/Llama-3.2-3B-Instruct"
  "speakleash/Bielik-1.5B-v3.0-Instruct"
  "speakleash/Bielik-4.5B-v3.0-Instruct"
)

declare -a LAYERS=(27 31 59)  # Corresponding layers for each model

# Dataset pairs (train, test)
declare -a TRAIN_DATASETS=("wgmix_train" "plmix_train")
declare -a TEST_DATASETS=("wgmix_test" "plmix_test")

# Aggregation methods
declare -a AGGREGATIONS=("mean" "last_token" "last_token_prefix")

# Distance metrics
declare -a METRICS=("euclidean" "mahalanobis")

# Calculate experiment from array task ID
# Grid: model x dataset_pair x aggregation x metric
# 3 models * 2 dataset_pairs * 3 aggregations * 2 metrics = 36

TASK_ID=$SLURM_ARRAY_TASK_ID

# Decode array index into experiment parameters
NUM_MODELS=3
NUM_DATASET_PAIRS=2
NUM_AGGREGATIONS=3
NUM_METRICS=2

# Calculate indices
METRIC_IDX=$((TASK_ID % NUM_METRICS))
REMAINING=$((TASK_ID / NUM_METRICS))

AGG_IDX=$((REMAINING % NUM_AGGREGATIONS))
REMAINING=$((REMAINING / NUM_AGGREGATIONS))

DATASET_PAIR_IDX=$((REMAINING % NUM_DATASET_PAIRS))
REMAINING=$((REMAINING / NUM_DATASET_PAIRS))

MODEL_IDX=$((REMAINING % NUM_MODELS))

# Get experiment parameters
MODEL="${MODELS[$MODEL_IDX]}"
LAYER="${LAYERS[$MODEL_IDX]}"
TRAIN_DATASET="${TRAIN_DATASETS[$DATASET_PAIR_IDX]}"
TEST_DATASET="${TEST_DATASETS[$DATASET_PAIR_IDX]}"
AGGREGATION="${AGGREGATIONS[$AGG_IDX]}"
METRIC="${METRICS[$METRIC_IDX]}"


echo ""
echo "=========================================="
echo "Experiment Configuration"
echo "=========================================="
echo "Model: $MODEL"
echo "Layer: $LAYER"
echo "Train Dataset: $TRAIN_DATASET"
echo "Test Dataset: $TEST_DATASET"
echo "Aggregation: $AGGREGATION"
echo "Metric: $METRIC"
echo "=========================================="
echo ""

# Run experiment
uv run python -m experiments.scripts.run_lpm_experiment \
  --model "$MODEL" \
  --train-dataset "$TRAIN_DATASET" \
  --test-dataset "$TEST_DATASET" \
  --aggregation "$AGGREGATION" \
  --metric "$METRIC" \
  --layer "$LAYER" \
  --device cpu \
  --batch-size 64 \
  --store "$STORE_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "✅ Experiment completed successfully"
else
  echo ""
  echo "❌ Experiment failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
