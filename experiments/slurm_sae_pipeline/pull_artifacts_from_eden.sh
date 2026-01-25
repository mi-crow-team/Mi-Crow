#!/bin/bash
set -euo pipefail

REMOTE_HOST="${EDEN_HOST:-eden}"
REMOTE_USER="${EDEN_USER:-akaniasty}"
REMOTE_BASE_PATH="${EDEN_BASE_PATH:-/mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow}"
REMOTE_RUNS_DIR="${REMOTE_BASE_PATH}/experiments/slurm_sae_pipeline/store/runs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCAL_RUNS_DIR="${PROJECT_ROOT}/experiments/slurm_sae_pipeline/store/runs"

DRY_RUN="${DRY_RUN:-0}"
VERBOSE="${VERBOSE:-1}"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Pull SAE training and inference runs from Eden server to local machine.
Excludes activation runs (activations_*).

Options:
    -h, --help              Show this help message
    -n, --dry-run           Show what would be synced without actually syncing
    -q, --quiet             Suppress progress output
    -H, --host HOST         Remote hostname (default: eden)
    -u, --user USER         Remote username (default: akaniasty)
    -p, --path PATH         Remote base path (default: /mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow)
    -l, --local PATH        Local runs directory (default: ./experiments/slurm_sae_pipeline/store/runs)

Environment variables:
    EDEN_HOST               Remote hostname
    EDEN_USER               Remote username
    EDEN_BASE_PATH          Remote base path
    DRY_RUN                 Set to 1 for dry run
    VERBOSE                 Set to 0 to suppress output

Examples:
    $0
    $0 --dry-run
    $0 --host eden --user akaniasty
    DRY_RUN=1 $0
EOF
    exit 0
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                ;;
            -n|--dry-run)
                DRY_RUN=1
                shift
                ;;
            -q|--quiet)
                VERBOSE=0
                shift
                ;;
            -H|--host)
                REMOTE_HOST="$2"
                shift 2
                ;;
            -u|--user)
                REMOTE_USER="$2"
                shift 2
                ;;
            -p|--path)
                REMOTE_BASE_PATH="$2"
                REMOTE_RUNS_DIR="${REMOTE_BASE_PATH}/experiments/slurm_sae_pipeline/store/runs"
                shift 2
                ;;
            -l|--local)
                LOCAL_RUNS_DIR="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1" >&2
                usage
                exit 1
                ;;
        esac
    done
}

parse_args "$@"

REMOTE="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_RUNS_DIR}"

if [[ $VERBOSE -eq 1 ]]; then
    echo "=== Pulling SAE Artifacts from Eden ==="
    echo "Remote: ${REMOTE}"
    echo "Local:  ${LOCAL_RUNS_DIR}"
    echo "Excluding: activations_* directories"
    echo ""
fi

mkdir -p "$LOCAL_RUNS_DIR"

RSYNC_OPTS=(
    --archive
    --compress
    --partial
    --progress
    --human-readable
    --exclude='activations_*'
    --exclude='*.tmp'
    --exclude='*.lock'
)

if [[ $DRY_RUN -eq 1 ]]; then
    RSYNC_OPTS+=(--dry-run --verbose)
    if [[ $VERBOSE -eq 1 ]]; then
        echo "🔍 DRY RUN MODE - No files will be transferred"
        echo ""
    fi
else
    if [[ $VERBOSE -eq 0 ]]; then
        RSYNC_OPTS+=(--quiet)
    fi
fi

if [[ $VERBOSE -eq 1 ]]; then
    echo "Syncing runs (excluding activations)..."
    echo ""
fi

if rsync "${RSYNC_OPTS[@]}" "$REMOTE/" "$LOCAL_RUNS_DIR/"; then
    if [[ $VERBOSE -eq 1 ]]; then
        echo ""
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "✅ Dry run completed successfully"
        else
            echo "✅ Successfully pulled artifacts from Eden"
            echo ""
            echo "Synced runs:"
            find "$LOCAL_RUNS_DIR" -maxdepth 1 -type d ! -path "$LOCAL_RUNS_DIR" | sort | while read -r dir; do
                echo "  - $(basename "$dir")"
            done
        fi
    fi
    exit 0
else
    echo "❌ Error: rsync failed" >&2
    exit 1
fi
