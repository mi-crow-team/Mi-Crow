#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=18G
#SBATCH --job-name=debug-save-activations
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

echo "========================================================================"
echo "DEBUG TEST: Save Activations"
echo "========================================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname -s)"
echo "PWD:       $(pwd)"
echo "Timestamp: $(date)"
echo "========================================================================"
echo ""

# Test configuration
MODEL="speakleash/Bielik-1.5B-v3.0-Instruct"
DATASET="plmix_train"
LAYER_NUM=31
BATCH_SIZE=16  # Small batch size for quick test
DEVICE="cpu"

echo "Test Configuration:"
echo "  Model:      $MODEL"
echo "  Dataset:    $DATASET"
echo "  Layer:      $LAYER_NUM"
echo "  Batch size: $BATCH_SIZE"
echo "  Device:     $DEVICE"
echo ""
echo "========================================================================"
echo ""

uv run python -m experiments.scripts.save_activations \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --layer-num "$LAYER_NUM" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --store "$STORE_DIR"

echo ""
echo "========================================================================"
echo "✅ DEBUG TEST COMPLETED SUCCESSFULLY"
echo "Timestamp: $(date)"
echo "========================================================================"
