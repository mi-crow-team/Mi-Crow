#!/bin/bash
#SBATCH --job-name=sae_save_activations
#SBATCH --output=sae_save_activations_%j.out
#SBATCH --error=sae_save_activations_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Example SLURM script for saving activations
# Adjust module loading and paths for your cluster

# Load modules (adjust for your cluster)
# module load python/3.10
# module load cuda/11.8

# Activate virtual environment (adjust path)
# source /path/to/venv/bin/activate

# Set environment variables
export STORE_DIR=${SCRATCH:-./store}/sae_store
export MODEL_ID="speakleash/Bielik-1.5B-v3.0-Instruct"
export DATA_LIMIT=100000
export BATCH_SIZE_SAVE=32
export LAYER_NUM=16
export DEVICE=cuda
export HF_DATASET="roneneldan/TinyStories"
export DATA_SPLIT="train"
export TEXT_FIELD="text"
export MAX_LENGTH=128

# Run script
cd "$(dirname "$0")"
python 01_save_activations.py

