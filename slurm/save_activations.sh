#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short,long
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --job-name=save-activations
#SBATCH --output=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%j.err
#SBATCH --export=ALL
#SBATCH --mail-user hubik112@gmail.com
#SBATCH --mail-type FAIL,END

set -euo pipefail

REPO_DIR=${REPO_DIR:-"$PWD"}
STORE_DIR=${STORE_DIR:-"$REPO_DIR/store"}
LOG_DIR=${LOG_DIR:-"$REPO_DIR/slurm-logs"}

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "Node: $(hostname -s)"
echo "PWD:  $(pwd)"
echo "GPU:  $(nvidia-smi -L)"

# Configuration
BATCH_SIZE=${BATCH_SIZE:-64}
DEVICE="cuda"

# Model configurations with their default last layers
declare -A MODEL_LAYERS
MODEL_LAYERS["meta-llama/Llama-3.2-3B-Instruct"]=27
MODEL_LAYERS["speakleash/Bielik-1.5B-v3.0-Instruct"]=31
MODEL_LAYERS["speakleash/Bielik-4.5B-v3.0-Instruct"]=59

# Datasets to process
DATASETS=("wgmix_train" "plmix_train")

# Process all combinations of models and datasets
for MODEL in "${!MODEL_LAYERS[@]}"; do
  LAYER_NUM=${MODEL_LAYERS[$MODEL]}
  
  echo ""
  echo "========================================================================"
  echo "Processing model: $MODEL (layer $LAYER_NUM)"
  echo "========================================================================"
  
  for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo ">>> Dataset: $DATASET"
    echo ""
    
    uv run python -m experiments.scripts.save_activations \
      --model "$MODEL" \
      --dataset "$DATASET" \
      --layer-num "$LAYER_NUM" \
      --batch-size "$BATCH_SIZE" \
      --device "$DEVICE" \
      --store "$STORE_DIR"
    
    echo ""
    echo "✅ Completed: $MODEL on $DATASET"
    echo ""
  done
done

echo ""
echo "========================================================================"
echo "All activations saved successfully!"
echo "========================================================================"
