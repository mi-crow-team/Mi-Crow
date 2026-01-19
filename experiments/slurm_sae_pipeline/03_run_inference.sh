#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short,long,debug
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=hopper-2,dgx-2,dgx-3,sr-1,sr-2
#SBATCH --job-name=sae-run-inference
#SBATCH --output=/mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow/slurm-logs/sae_run_inference-%j.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow/slurm-logs/sae_run_inference-%j.err
#SBATCH --export=ALL
#SBATCH --mail-user=adam.master111@gmail.com
#SBATCH --mail-type FAIL,END

set -euo pipefail

REPO_DIR="/mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow"
STORE_DIR="${STORE_DIR:-$REPO_DIR/experiments/slurm_sae_pipeline/store}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/slurm-logs}"
CONFIG_FILE="${CONFIG_FILE:-$REPO_DIR/experiments/slurm_sae_pipeline/configs/config_bielik12_polemo2.json}"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"
NVIDIA_SMI_LOG="$LOG_DIR/nvidia-smi-%j.log"

echo "=== Job Information ==="
echo "Node: $(hostname -s)"
echo "PWD: $(pwd)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'No GPU available')"
echo ""

echo "Logging GPU usage to $NVIDIA_SMI_LOG"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 60 > "$NVIDIA_SMI_LOG" &
SMI_PID=$!
trap "kill $SMI_PID 2>/dev/null || true" EXIT

UV_BIN="$REPO_DIR/.uv-bin/uv"
if [[ ! -f "$UV_BIN" ]]; then
    echo "Installing uv to project directory..."
    mkdir -p "$REPO_DIR/.uv-bin"
    
    if [[ -f "$HOME/.local/bin/uv" ]]; then
        cp "$HOME/.local/bin/uv" "$UV_BIN"
        chmod +x "$UV_BIN"
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
        if [[ -f "$HOME/.cargo/bin/uv" ]]; then
            cp "$HOME/.cargo/bin/uv" "$UV_BIN"
            chmod +x "$UV_BIN"
        elif [[ -f "$HOME/.local/bin/uv" ]]; then
            cp "$HOME/.local/bin/uv" "$UV_BIN"
            chmod +x "$UV_BIN"
        fi
    fi
fi

export UV_CACHE_DIR="$REPO_DIR/.uv-cache"
mkdir -p "$UV_CACHE_DIR"

export PATH="$REPO_DIR/.uv-bin:$PATH"
export UV_HTTP_TIMEOUT=300

echo "Using uv: $(command -v uv)"
echo "uv version: $(uv --version)"
echo ""

echo "Setting up Python environment..."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Syncing dependencies with uv..."
uv sync --frozen
echo ""

if [[ -z "${HF_TOKEN:-}" ]] && [[ ! -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]]; then
    echo "⚠️  Warning: HF_TOKEN not set and no HuggingFace token file found."
    echo "   The job may fail if the model requires authentication."
    echo "   Set HF_TOKEN environment variable or run 'huggingface-cli login'"
    echo ""
fi

if [[ -z "${SAE_PATHS:-}" ]]; then
    echo "❌ Error: SAE_PATHS environment variable is required"
    echo "   Example: SAE_PATHS='path1.pt path2.pt' sbatch 03_run_inference.sh"
    exit 1
fi

echo "SAE paths: $SAE_PATHS"
echo ""

uv run python "$REPO_DIR/experiments/slurm_sae_pipeline/03_run_inference.py" \
    --config "$CONFIG_FILE" \
    --sae_paths $SAE_PATHS \
    ${TRACK_BOTH:+--track_both} \
    ${TOP_K:+--top_k $TOP_K} \
    ${BATCH_SIZE:+--batch_size $BATCH_SIZE} \
    ${DUMP_EVERY_N_BATCHES:+--dump_every_n_batches $DUMP_EVERY_N_BATCHES} \
    ${DATA_LIMIT:+--data_limit $DATA_LIMIT}

echo "✅ SAE inference completed!"
