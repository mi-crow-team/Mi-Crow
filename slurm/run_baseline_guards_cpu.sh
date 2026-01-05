#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --job-name=baseline-guards-cpu
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

# Using pre-cached PLMix dataset from store/datasets/plmix_test
# No HuggingFace authentication needed since we load from disk

BATCH_SIZE=${BATCH_SIZE:-16}

# Respect allocated cores for common CPU backends.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}


# Run all combinations: (wgmix_test, plmix_test) x (BielikGuard, LlamaGuard)

for DATASET in wgmix_test plmix_test; do
  # BielikGuard
  uv run python -m experiments.scripts.run_baseline_guards \
    --store "$STORE_DIR" \
    --dataset-name "$DATASET" \
    --run-bielik \
    --device cpu \
    --batch-size "$BATCH_SIZE"

  # LlamaGuard
  uv run python -m experiments.scripts.run_baseline_guards \
    --store "$STORE_DIR" \
    --dataset-name "$DATASET" \
    --run-llama \
    --llama-model "meta-llama/Llama-Guard-3-1B" \
    --device cpu \
    --batch-size "$BATCH_SIZE"
done