#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --job-name=test-save-activations
#SBATCH --output=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%j.err
#SBATCH --export=ALL

# This is a test in feat/mem-optim-activations to test if still works
set -euo pipefail

REPO_DIR=${REPO_DIR:-"$PWD"}
mkdir -p "$REPO_DIR/slurm-logs"
cd "$REPO_DIR"

echo "Node: $(hostname -s)"
echo "PWD:  $(pwd)"
echo "GPU:  $(nvidia-smi -L)"

# Run the debug script
# Using uv run python -m to ensure the package structure is respected
uv run python -m experiments.scripts.debug.test_activation_saving

echo "✅ Debug test script finished!"
