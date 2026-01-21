#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short,long,debug
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=monitor-pipeline
#SBATCH --output=/mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow/slurm-logs/monitor_pipeline-%j.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow/slurm-logs/monitor_pipeline-%j.err
#SBATCH --export=ALL

set -euo pipefail

REPO_DIR="/mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow"
SCRIPT_NAME="${SCRIPT_NAME:-01_save_activations}"
TIMEOUT="${TIMEOUT:-600}"
MONITORING_INTERVAL="${MONITORING_INTERVAL:-5}"

cd "$REPO_DIR"

echo "=== Hardware Monitoring Job ==="
echo "Node: $(hostname -s)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'No GPU available')"
echo "Script: $SCRIPT_NAME"
echo "Timeout: ${TIMEOUT}s"
echo ""

# Setup uv
UV_BIN="$REPO_DIR/.uv-bin/uv"
if [[ ! -f "$UV_BIN" ]]; then
    echo "Installing uv..."
    mkdir -p "$REPO_DIR/.uv-bin"
    if [[ -f "$HOME/.local/bin/uv" ]]; then
        cp "$HOME/.local/bin/uv" "$UV_BIN"
        chmod +x "$UV_BIN"
    fi
fi

export PATH="$REPO_DIR/.uv-bin:$PATH"
export UV_CACHE_DIR="$REPO_DIR/.uv-cache"
mkdir -p "$UV_CACHE_DIR"

# Output directory for metrics
OUTPUT_DIR="$REPO_DIR/experiments/slurm_sae_pipeline/hardware_monitoring_output"
mkdir -p "$OUTPUT_DIR"

# Determine script path and arguments
SCRIPT_PATH="$REPO_DIR/experiments/slurm_sae_pipeline/${SCRIPT_NAME}.py"
METRICS_FILE="$OUTPUT_DIR/${SCRIPT_NAME}_baseline_metrics.json"

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "Error: Script not found: $SCRIPT_PATH"
    exit 1
fi

# Determine script arguments
SCRIPT_ARGS=()
if [[ "$SCRIPT_NAME" == "01_save_activations" ]]; then
    SCRIPT_ARGS=("--config" "experiments/slurm_sae_pipeline/configs/small_config.json")
elif [[ "$SCRIPT_NAME" == "02_train_sae" ]]; then
    SCRIPT_ARGS=("--config" "experiments/slurm_sae_pipeline/configs/small_config.json")
elif [[ "$SCRIPT_NAME" == "03_run_inference" ]]; then
    # Find existing SAE models for testing
    # Use only 1 SAE path since small_config.json has layer_num: 0 (single layer)
    SAE_PATHS_ARRAY=($(find "$REPO_DIR/experiments/slurm_sae_pipeline/store/runs" -name "model.pt" -path "*/sae_*" 2>/dev/null | head -1))
    if [[ ${#SAE_PATHS_ARRAY[@]} -eq 0 ]]; then
        echo "Warning: No SAE models found. Using dummy path for monitoring."
        SAE_PATHS_ARRAY=("dummy_path1.pt")
    fi
    SCRIPT_ARGS=(
        "--sae_paths" "${SAE_PATHS_ARRAY[@]}"
        "--config" "experiments/slurm_sae_pipeline/configs/small_config.json"
        "--batch_size" "8"
        "--data_limit" "10"
    )
elif [[ "$SCRIPT_NAME" == "04_concept_manipulation_experiments" ]]; then
    SCRIPT_ARGS=("--test")
fi

echo "Starting monitoring and script execution..."
echo "Script: $SCRIPT_PATH"
echo "Args: ${SCRIPT_ARGS[*]}"
echo "Metrics output: $METRICS_FILE"
echo ""

# Run with monitoring using Python
# Use -- separator to ensure script args are not confused with monitoring args
uv run python "$REPO_DIR/experiments/slurm_sae_pipeline/run_with_monitoring.py" \
    "$SCRIPT_PATH" \
    --timeout "$TIMEOUT" \
    --metrics-output "$METRICS_FILE" \
    --monitoring-interval "$MONITORING_INTERVAL" \
    -- \
    "${SCRIPT_ARGS[@]}"

EXIT_CODE=$?

# Check if metrics were saved to default location instead (run_with_monitoring.py may save to script dir)
DEFAULT_METRICS_FILE="$REPO_DIR/experiments/slurm_sae_pipeline/${SCRIPT_NAME}_metrics.json"
if [[ ! -f "$METRICS_FILE" ]] && [[ -f "$DEFAULT_METRICS_FILE" ]]; then
    echo "⚠️ Metrics saved to default location, copying to expected location..."
    mkdir -p "$OUTPUT_DIR"
    cp "$DEFAULT_METRICS_FILE" "$METRICS_FILE"
    echo "✅ Copied metrics to: $METRICS_FILE"
fi

if [[ $EXIT_CODE -eq 0 ]] || [[ -f "$METRICS_FILE" ]]; then
    echo ""
    echo "=== Extrapolating metrics to 10 hours ==="
    
    EXTRAPOLATED_FILE="$OUTPUT_DIR/${SCRIPT_NAME}_10h_metrics.json"
    
    uv run python "$REPO_DIR/experiments/slurm_sae_pipeline/extrapolate_metrics.py" \
        "$METRICS_FILE" \
        --output "$EXTRAPOLATED_FILE" \
        --duration-hours 10.0 \
        --interval-seconds 60.0
    
    if [[ -f "$EXTRAPOLATED_FILE" ]]; then
        echo ""
        echo "=== Generating synthetic log files ==="
        
        SCRIPT_NAME_SHORT="${SCRIPT_NAME#*_}"
        SCRIPT_NAME_SHORT="${SCRIPT_NAME_SHORT//_/-}"
        
        uv run python "$REPO_DIR/experiments/slurm_sae_pipeline/generate_synthetic_logs.py" \
            "$EXTRAPOLATED_FILE" \
            --script-name "$SCRIPT_NAME_SHORT" \
            --output-dir "$REPO_DIR/experiments/slurm_sae_pipeline/logs" \
            --job-id "$SLURM_JOB_ID"
        
        echo ""
        echo "✅ Monitoring complete!"
        echo "   Baseline metrics: $METRICS_FILE"
        echo "   Extrapolated metrics: $EXTRAPOLATED_FILE"
    fi
fi

exit $EXIT_CODE
