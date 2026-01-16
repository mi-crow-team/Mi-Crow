#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p experimental
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --job-name=sae_save_activations
#SBATCH --output=/mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow/slurm-logs/sae_save_activations-%j.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow/slurm-logs/sae_save_activations-%j.err
#SBATCH --export=ALL
#SBATCH --mail-user=adam.master111@gmail.com
#SBATCH --mail-type=FAIL,END

set -euo pipefail

REPO_DIR=${REPO_DIR:-"/mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow"}
STORE_DIR=${STORE_DIR:-"$REPO_DIR/experiments/slurm_sae_pipeline/store"}
LOG_DIR=${LOG_DIR:-"$REPO_DIR/slurm-logs"}
CONFIG_FILE=${CONFIG_FILE:-"$REPO_DIR/experiments/slurm_sae_pipeline/configs/config_bielik12_polemo2.json"}

mkdir -p "$LOG_DIR"
mkdir -p "$STORE_DIR"
cd "$REPO_DIR"

echo "Node: $(hostname -s)"
echo "PWD:  $(pwd)"
echo "Config: $CONFIG_FILE"
echo "GPU:  $(nvidia-smi -L)"


# Respect allocated cores for common CPU backends
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Set cache directory to repo-local location to enable hardlinks on NFS
export UV_CACHE_DIR="$REPO_DIR/.uv-cache"
mkdir -p "$UV_CACHE_DIR"

# Use project-local uv installation
UV_BIN="$REPO_DIR/.uv-bin/uv"
if [ ! -f "$UV_BIN" ]; then
    echo "Installing uv locally..."
    mkdir -p "$REPO_DIR/.uv-bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Copy to project directory
    if [ -f "$HOME/.local/bin/uv" ]; then
        cp "$HOME/.local/bin/uv" "$UV_BIN"
        chmod +x "$UV_BIN"
        # Clean up home installation
        rm -f "$HOME/.local/bin/uv" "$HOME/.local/bin/uvx" 2>/dev/null || true
    else
        echo "ERROR: uv installation failed"
        exit 1
    fi
    if [ ! -f "$UV_BIN" ]; then
        echo "ERROR: Failed to install uv to project directory"
        exit 1
    fi
fi

echo "Using uv: $UV_BIN"
echo "uv version: $($UV_BIN --version)"

# Check for HuggingFace authentication (required for gated models)
if [[ -z "${HF_TOKEN:-}" ]] && [[ ! -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]]; then
  echo "ERROR: HuggingFace auth missing (Bielik model is gated)." >&2
  echo "Run once:  $UV_BIN run huggingface-cli login" >&2
  echo "Or submit with: HF_TOKEN=... sbatch 01_save_activations.sh" >&2
  exit 2
fi

# Run activation saving script
cd "$REPO_DIR/experiments/slurm_sae_pipeline"
$UV_BIN run python 01_save_activations.py --config "$CONFIG_FILE" --run_id "bielik12_polemo2_layer12_3epochs"

echo "✅ Activation saving completed!"
