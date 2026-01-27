#!/bin/bash
#
# Test script for Linear Probe implementation
# Runs a small test with Bielik 1.5B on wgmix dataset with limited samples

set -e

echo "=================================="
echo "Testing Linear Probe Implementation"
echo "=================================="

# Test with small sample size for quick validation
uv run python -m experiments.scripts.run_probe_experiment_oom \
    --model speakleash/Bielik-1.5B-v3.0-Instruct \
    --train-dataset wgmix_train \
    --test-dataset wgmix_test \
    --aggregation last_token \
    --layer 31 \
    --learning-rate 1e-3 \
    --weight-decay 1e-4 \
    --batch-size 32 \
    --max-epochs 20 \
    --patience 5 \
    --max-train-samples 200 \
    --test-limit 100 \
    --seed 42

echo "=================================="
echo "Test completed successfully!"
echo "=================================="
