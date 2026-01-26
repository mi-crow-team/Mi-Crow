#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --job-name=lpm-opt-experiments
#SBATCH --output=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%A-%a.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%A-%a.err
#SBATCH --export=ALL
#SBATCH --mail-user hubik112@gmail.com
#SBATCH --mail-type FAIL,END
#SBATCH --array=0-35

set -euo pipefail

REPO_DIR=${REPO_DIR:-"$PWD"}
# Using the store path provided in your benchmark scripts
STORE_DIR=${STORE_DIR:-"/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store"}
LOG_DIR=${LOG_DIR:-"$REPO_DIR/slurm-logs"}

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "=========================================="
echo "SLURM Array Job: LPM Optimized Experiments"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname -s)"
echo "PWD: $(pwd)"
echo "=========================================="

# Define experiment grid
# Models: 3
declare -a MODELS=(
  "meta-llama/Llama-3.2-3B-Instruct"
  "speakleash/Bielik-1.5B-v3.0-Instruct"
  "speakleash/Bielik-4.5B-v3.0-Instruct"
)
declare -a LAYERS=(27 31 59)

# Dataset pairs: 2
declare -a TRAIN_DATASETS=("wgmix_train" "plmix_train")
declare -a TEST_DATASETS=("wgmix_test" "plmix_test")

# Aggregation methods: 3
declare -a AGGREGATIONS=("mean" "last_token" "last_token_prefix")

# Distance metrics: 2
declare -a METRICS=("euclidean" "mahalanobis")

# Logic to decode SLURM_ARRAY_TASK_ID
TASK_ID=$SLURM_ARRAY_TASK_ID
NUM_MODELS=3
NUM_DATASET_PAIRS=2
NUM_AGGREGATIONS=3
NUM_METRICS=2

METRIC_IDX=$((TASK_ID % NUM_METRICS))
REMAINING=$((TASK_ID / NUM_METRICS))

AGG_IDX=$((REMAINING % NUM_AGGREGATIONS))
REMAINING=$((REMAINING / NUM_AGGREGATIONS))

DATASET_PAIR_IDX=$((REMAINING % NUM_DATASET_PAIRS))
REMAINING=$((REMAINING / NUM_DATASET_PAIRS))

MODEL_IDX=$((REMAINING % NUM_MODELS))

# Extract parameters
MODEL="${MODELS[$MODEL_IDX]}"
LAYER="${LAYERS[$MODEL_IDX]}"
TRAIN_DATASET="${TRAIN_DATASETS[$DATASET_PAIR_IDX]}"
TEST_DATASET="${TEST_DATASETS[$DATASET_PAIR_IDX]}"
AGGREGATION="${AGGREGATIONS[$AGG_IDX]}"
METRIC="${METRICS[$METRIC_IDX]}"

# Setup command arguments
ARGS=(
  --model "$MODEL"
  --train-dataset "$TRAIN_DATASET"
  --test-dataset "$TEST_DATASET"
  --aggregation "$AGGREGATION"
  --metric "$METRIC"
  --layer "$LAYER"
  --device cpu
  --batch-size 32
  --benchmark
  --store "$STORE_DIR"
)

# Apply test limit for wgmix datasets
if [[ "$TRAIN_DATASET" == *"wgmix"* ]]; then
  ARGS+=(--test-limit 500)
  echo ">>> WGMIX detected: Setting test-limit to 500"
fi

echo ""
echo "=========================================="
echo "Experiment Configuration"
echo "=========================================="
echo "Model:       $MODEL"
echo "Layer:       $LAYER"
echo "Train Set:   $TRAIN_DATASET"
echo "Test Set:    $TEST_DATASET"
echo "Aggregation: $AGGREGATION"
echo "Metric:      $METRIC"
echo "Batch Size:  32"
echo "=========================================="
echo ""

# Respect allocated cores for CPU backends
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Run experiment using the new OOM-optimized script
uv run python -m experiments.scripts.run_lpm_experiment_oom "${ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "✅ Experiment completed successfully"
else
  echo ""
  echo "❌ Experiment failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
