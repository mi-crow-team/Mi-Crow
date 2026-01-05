#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short,long
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --mem=40G
#SBATCH --job-name=direct-prompting
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

echo "============================================"
echo "Direct Prompting Experiments - SLURM Job"
echo "============================================"
echo "Node: $(hostname -s)"
echo "PWD:  $(pwd)"
echo "Date: $(date)"
echo "============================================"

# Using pre-cached datasets from store/datasets/{wgmix_test, plmix_test}
# No HuggingFace authentication needed since we load from disk

BATCH_SIZE=${BATCH_SIZE:-128}

# Respect allocated cores for common CPU backends
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
# Models to test
LLAMA_MODEL="meta-llama/Llama-3.2-3B-Instruct"
BIELIK_MODEL="speakleash/Bielik-4.5B-v3.0-Instruct"

# Run all combinations: 2 datasets x 2 models = 4 experiments
# Each experiment runs all 4 prompts sequentially

echo ""
echo "============================================"
echo "Experiment 1/4: Llama on WGMix (English)"
echo "============================================"
uv run python -m experiments.scripts.run_direct_prompting \
  --store "$STORE_DIR" \
  --dataset-name wgmix_test \
  --model "$LLAMA_MODEL" \
  --device cpu \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens 10 \
  --temperature 0.0

echo ""
echo "============================================"
echo "Experiment 2/4: Llama on PLMix (Polish)"
echo "============================================"
uv run python -m experiments.scripts.run_direct_prompting \
  --store "$STORE_DIR" \
  --dataset-name plmix_test \
  --model "$LLAMA_MODEL" \
  --device cpu \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens 10 \
  --temperature 0.0

echo ""
echo "============================================"
echo "Experiment 3/4: Bielik on WGMix (English)"
echo "============================================"
uv run python -m experiments.scripts.run_direct_prompting \
  --store "$STORE_DIR" \
  --dataset-name wgmix_test \
  --model "$BIELIK_MODEL" \
  --device cpu \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens 10 \
  --temperature 0.0

echo ""
echo "============================================"
echo "Experiment 4/4: Bielik on PLMix (Polish)"
echo "============================================"
uv run python -m experiments.scripts.run_direct_prompting \
  --store "$STORE_DIR" \
  --dataset-name plmix_test \
  --model "$BIELIK_MODEL" \
  --device cpu \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens 10 \
  --temperature 0.0

echo ""
echo "============================================"
echo "All experiments complete!"
echo "Date: $(date)"
echo "============================================"
