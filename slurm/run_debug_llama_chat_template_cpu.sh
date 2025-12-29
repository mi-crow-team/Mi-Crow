#!/bin/bash
#SBATCH -A mi2lab-normal
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --job-name=debug-llama-chat-template-cpu
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

echo "Node: $(hostname -s)"
echo "PWD:  $(pwd)"

# WildGuardMix is gated on HF Hub; authenticate via `huggingface-cli login` on the login node
# or pass a token via environment variable: `HF_TOKEN=... sbatch ...`.
if [[ -z "${HF_TOKEN:-}" ]] && [[ ! -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]]; then
  echo "ERROR: HuggingFace auth missing (wildguardmix is gated)." >&2
  echo "Run once:  uv run huggingface-cli login" >&2
  echo "Or submit with: HF_TOKEN=... sbatch slurm/run_debug_llama_chat_template_cpu.sh" >&2
  exit 2
fi

LIMIT=${LIMIT:-6}
BATCH_SIZE=${BATCH_SIZE:-3}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}

# Respect allocated cores for common CPU backends.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

uv run python -m experiments.scripts.debug_llama_chat_template \
  --store "$STORE_DIR" \
  --device cpu \
  --limit "$LIMIT" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS"
