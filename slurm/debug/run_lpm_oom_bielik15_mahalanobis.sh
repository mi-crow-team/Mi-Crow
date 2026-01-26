#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --mem=36G
#SBATCH --job-name=lpm-bielik15-mahalanobis
#SBATCH --output=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%j.err
#SBATCH --export=ALL
#SBATCH --mail-user=hubik112@gmail.com
#SBATCH --mail-type FAIL,END

set -euo pipefail

REPO_DIR=${REPO_DIR:-"$PWD"}
STORE_DIR=${STORE_DIR:-"/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store"}
LOG_DIR=${LOG_DIR:-"$REPO_DIR/slurm-logs"}

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

# Configuration
MODEL="speakleash/Bielik-1.5B-v3.0-Instruct"
TRAIN_DATASET="wgmix_train"
TEST_DATASET="wgmix_test"
AGGREGATION="last_token"
METRIC="mahalanobis"
LAYER=31
TEST_LIMIT=400

# Respect allocated cores for CPU backends
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}

echo "Starting LPM run for Bielik-1.5B (Mahalanobis, limit 400)..."

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
  --test-limit "$TEST_LIMIT" \
  --benchmark \
  --store "$STORE_DIR"
