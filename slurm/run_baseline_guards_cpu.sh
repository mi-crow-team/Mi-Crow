#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --job-name=baseline-guards-cpu
#SBATCH --output=/mnt/evafs/groups/mi2lab/hkowalski/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/hkowalski/slurm-logs/%x-%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-"$PWD"}
STORE_DIR=${STORE_DIR:-"$REPO_DIR/store"}
LOG_DIR=${LOG_DIR:-"$REPO_DIR/slurm-logs"}

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "Node: $(hostname -s)"
echo "PWD:  $(pwd)"

# Optional: reduce accidental huge runs. Override by exporting LIMIT=... before sbatch.
BATCH_SIZE=${BATCH_SIZE:-16}

# Respect allocated cores for common CPU backends.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

uv run python -m experiments.scripts.run_baseline_guards \
  --store "$STORE_DIR" \
  --run-bielik \
  --run-llama \
  --device cpu \
  --batch-size "$BATCH_SIZE"
