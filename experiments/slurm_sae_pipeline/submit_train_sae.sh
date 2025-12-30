#!/bin/bash
#SBATCH --job-name=sae_train
#SBATCH --output=sae_train_%j.out
#SBATCH --error=sae_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Example SLURM script for training SAE
# Adjust module loading and paths for your cluster

# Load modules (adjust for your cluster)
# module load python/3.10
# module load cuda/11.8

# Activate virtual environment (adjust path)
# source /path/to/venv/bin/activate

# Set environment variables
export STORE_DIR=${SCRATCH:-./store}/sae_store
export MODEL_ID="speakleash/Bielik-1.5B-v3.0-Instruct"
export EPOCHS=20
export BATCH_SIZE_TRAIN=64
export N_LATENTS_MULTIPLIER=8
export TOP_K=8
export LR=1e-3
export L1_LAMBDA=1e-4
export DEVICE=cuda

# Parse --run_id flag if provided
RUN_ID_ARG=""
if [ "$1" = "--run_id" ] && [ -n "$2" ]; then
    RUN_ID_ARG="--run_id $2"
fi

# Run script
cd "$(dirname "$0")"
uv run python 02_train_sae.py $RUN_ID_ARG

