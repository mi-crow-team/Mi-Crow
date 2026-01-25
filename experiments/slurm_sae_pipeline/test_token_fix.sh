#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p short,debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=hopper-2,dgx-2,dgx-3,sr-1,sr-2
#SBATCH --job-name=test-token-fix
#SBATCH --output=/mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow/slurm-logs/test_token_fix-%j.out
#SBATCH --error=/mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow/slurm-logs/test_token_fix-%j.err
#SBATCH --export=ALL

set -euo pipefail

REPO_DIR="/mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow"
LOG_DIR="${LOG_DIR:-$REPO_DIR/slurm-logs}"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "=== Token ID Fix Test Job ==="
echo "Node: $(hostname -s)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'No GPU available')"
echo ""

UV_BIN="$REPO_DIR/.uv-bin/uv"
if [[ ! -f "$UV_BIN" ]]; then
    echo "Installing uv..."
    mkdir -p "$REPO_DIR/.uv-bin"
    if [[ -f "$HOME/.local/bin/uv" ]]; then
        cp "$HOME/.local/bin/uv" "$UV_BIN"
        chmod +x "$UV_BIN"
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
        if [[ -f "$HOME/.cargo/bin/uv" ]]; then
            cp "$HOME/.cargo/bin/uv" "$UV_BIN"
            chmod +x "$UV_BIN"
        fi
    fi
fi

export UV_CACHE_DIR="$REPO_DIR/.uv-cache"
mkdir -p "$UV_CACHE_DIR"
export PATH="$REPO_DIR/.uv-bin:$PATH"
export UV_HTTP_TIMEOUT=300

echo "Syncing dependencies..."
uv sync --frozen
echo ""

export PYTHONUNBUFFERED=1
uv run python "$REPO_DIR/experiments/slurm_sae_pipeline/test_token_fix.py"

echo ""
echo "✅ Test completed!"
