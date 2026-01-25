#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH --output=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%j.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/slurm-logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --job-name=new_prep_datasets
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=short

# Prepare NEW datasets for debugging/comparison
# This creates new_wgmix_train, new_wgmix_test, new_plmix_train, new_plmix_test
# to compare with the original datasets

echo "========================================="
echo "Preparing NEW datasets (for debugging)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

cd /mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow

# Run the new prepare datasets script
uv run python -m experiments.scripts.debug.new_prepare_datasets --seed 42

echo ""
echo "========================================="
echo "NEW Dataset preparation complete!"
echo "End time: $(date)"
echo "========================================="
