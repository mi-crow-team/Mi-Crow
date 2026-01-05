#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH --job-name=prepare_datasets
#SBATCH --output=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --export=ALL
#SBATCH --mail-user hubik112@gmail.com
#SBATCH --mail-type FAIL,END

# Prepare and cache all datasets for experiments
# This downloads WildGuardMix and loads PL Mix datasets from CSV

echo "=================================================="
echo "Dataset Preparation Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Load required modules (adjust for your cluster)
# module load python/3.11
# module load cuda/12.1  # If needed for HF datasets backend

# Activate virtual environment if needed
# source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs/slurm

# Run the dataset preparation script
echo ""
echo "Running dataset preparation..."
echo ""

python -m experiments.scripts.prepare_datasets --seed 42

exit_code=$?

echo ""
echo "=================================================="
echo "Job finished at: $(date)"
echo "Exit code: $exit_code"
echo "=================================================="

exit $exit_code
