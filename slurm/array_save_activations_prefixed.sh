#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short
#SBATCH -t 3:00:00
#SBATCH -N 1
#SBATCH -c 3
#SBATCH --mem=42G
#SBATCH --job-name=array-save-activations-prefixed
#SBATCH --array=0-5
#SBATCH --output=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%A_%a.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%A_%a.err
#SBATCH --export=ALL
#SBATCH --mail-user hubik112@gmail.com
#SBATCH --mail-type FAIL,END

set -euo pipefail

REPO_DIR=${REPO_DIR:-"$PWD"}
STORE_DIR=${STORE_DIR:-"$REPO_DIR/store"}
LOG_DIR=${LOG_DIR:-"$REPO_DIR/slurm-logs"}

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname -s)"
echo "PWD:  $(pwd)"

# Configuration
BATCH_SIZE=${BATCH_SIZE:-32}
DEVICE="cpu"

# Define all model-dataset combinations with prefix
# 3 models × 2 datasets (train only) = 6 tasks (indices 0-5)
MODELS=(
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "speakleash/Bielik-1.5B-v3.0-Instruct"
  "speakleash/Bielik-1.5B-v3.0-Instruct"
  "speakleash/Bielik-4.5B-v3.0-Instruct"
  "speakleash/Bielik-4.5B-v3.0-Instruct"
)

DATASETS=(
  "wgmix_train"
  "plmix_train"
  "wgmix_train"
  "plmix_train"
  "wgmix_train"
  "plmix_train"
)

LAYERS=(
  27
  27
  31
  31
  59
  59
)

# Get configuration for this task
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
LAYER_NUM=${LAYERS[$SLURM_ARRAY_TASK_ID]}

echo ""
echo "========================================================================"
echo "Task $SLURM_ARRAY_TASK_ID: Processing $MODEL on $DATASET (layer $LAYER_NUM) WITH PREFIX"
echo "========================================================================"
echo ""

uv run python -m experiments.scripts.save_activations \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --layer-num "$LAYER_NUM" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --store "$STORE_DIR" \
  --use-prefix

echo ""
echo "✅ Task $SLURM_ARRAY_TASK_ID completed: $MODEL on $DATASET (prefixed)"
echo ""
