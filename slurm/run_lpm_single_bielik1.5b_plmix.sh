#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short,long
#SBATCH -t 05:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=36G
#SBATCH --job-name=lpm-single-bielik1.5b-plmix
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

echo "=========================================="
echo "SLURM Job: LPM Single Experiment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname -s)"
echo "PWD: $(pwd)"
echo "=========================================="

MODEL="speakleash/Bielik-1.5B-v3.0-Instruct"
LAYER=31
TRAIN_DATASET="plmix_train"
TEST_DATASET="plmix_test"
AGGREGATION="mean"
METRIC="euclidean"
DEVICE="cpu"
BATCH_SIZE=64

# Print experiment configuration
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
echo "Device: $DEVICE"
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
  --device "$DEVICE" \
  --batch-size $BATCH_SIZE \
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
