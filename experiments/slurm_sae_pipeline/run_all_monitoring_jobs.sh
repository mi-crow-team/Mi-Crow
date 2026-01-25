#!/bin/bash
# Submit all monitoring jobs to SLURM

REPO_DIR="/mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow"
cd "$REPO_DIR"

echo "Submitting hardware monitoring jobs for all pipeline scripts..."
echo ""

# Submit jobs for each script
for script in "01_save_activations" "02_train_sae" "03_run_inference" "04_concept_manipulation_experiments"; do
    echo "Submitting job for $script..."
    JOB_ID=$(sbatch \
        --export=ALL,SCRIPT_NAME="$script",TIMEOUT=600,MONITORING_INTERVAL=5 \
        experiments/slurm_sae_pipeline/run_with_monitoring_slurm.sh \
        | grep -oP '\d+')
    
    if [[ -n "$JOB_ID" ]]; then
        echo "  ✅ Job submitted: $JOB_ID"
    else
        echo "  ❌ Failed to submit job"
    fi
done

echo ""
echo "All jobs submitted. Check status with: squeue -u \$USER"
echo "Results will be in: experiments/slurm_sae_pipeline/hardware_monitoring_output/"
