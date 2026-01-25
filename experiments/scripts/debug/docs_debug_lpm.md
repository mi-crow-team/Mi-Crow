# LPM Experiment Debug Scripts

This directory contains debug scripts for diagnosing issues with LPM (Latent Prototype Model) experiments.

## Overview

These scripts were created to diagnose and fix errors from the cluster LPM experiment runs (Job ID: 1541517). They help identify issues with:

- Dataset determinism and ordering
- Activation-dataset alignment
- Tokenization consistency across steps
- Classification collapse (single-class predictions)

## Prerequisites

All scripts require access to:
- Dataset files in `store/datasets/`
- Saved activation runs in `store/runs/`
- LPM experiment results in `store/lpm_*/`

For cluster usage, ensure paths point to `/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store/`.

## Scripts

### 1. verify_dataset_determinism.py

**Purpose:** Verify that datasets load in the same order every time.

**Usage:**
```bash
python verify_dataset_determinism.py --dataset wgmix_train --num_loads 10
```

**What it checks:**
- Sample count consistency across multiple loads
- Sample order (using text hashes)
- Label consistency
- First/last text matching

**Expected output:**
```
✅ DATASET IS DETERMINISTIC
   - Sample order is identical across all loads
   - Text content matches character-by-character
   - Labels are consistent
```

**When to use:**
- Before running experiments to ensure reproducibility
- If suspecting dataset ordering issues
- After dataset modifications

---

### 2. verify_activation_alignment.py

**Purpose:** Verify that saved activations align with the dataset samples.

**Usage:**
```bash
python verify_activation_alignment.py \
    --dataset wgmix_train \
    --activation_run activations_maxlen_512_llama_3_2_3b_instruct_wgmix_train_prefixed_layer27_20260117_233725 \
    --store_path /mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store
```

**What it checks:**
- Sample count matches between dataset and activations
- Label alignment (if labels stored in activations)
- Batch structure consistency

**Expected output:**
```
✅ ACTIVATIONS ARE ALIGNED WITH DATASET
   - Sample counts match (5000)
   - Labels match
   - Batch structure is consistent
```

**When to use:**
- Before training LPM to ensure activations match dataset
- If getting unexpected training results
- If suspecting dataset/activation mismatch

**Common issues detected:**
- `SAMPLE_COUNT_MISMATCH`: Different number of samples
- `LABEL_ALIGNMENT_ERROR`: Labels don't match between dataset and activations
- `UNEXPECTED_BATCH_SIZE`: Inconsistent batching

---

### 3. verify_tokenization.py

**Purpose:** Verify tokenization consistency between attention mask saving (Step 3) and inference (Step 4).

**Usage:**
```bash
python verify_tokenization.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset wgmix_test \
    --aggregation last_token_prefix \
    --sample_size 100
```

**What it checks:**
- Token IDs match between Step 3 and Step 4
- Attention mask shapes are consistent
- Token counts are identical
- Padding is applied correctly
- Special tokens handled consistently

**Expected output:**
```
✅ TOKENIZATION IS CONSISTENT
   - Step 3 and Step 4 produce identical tokenization
   - All sequences properly padded/truncated
   - No critical issues detected
```

**When to use:**
- Before running inference to ensure consistency
- If getting shape mismatch errors during inference
- If suspecting tokenization parameter differences
- When using prefix templates (`last_token_prefix` aggregation)

**Common issues detected:**
- `TOKEN_ID_MISMATCH`: Different tokens produced
- `ATTENTION_MASK_MISMATCH`: Mask shapes don't match
- `TOKEN_COUNT_MISMATCH`: Different sequence lengths

---

### 4. debug_single_class_predictions.py

**Purpose:** Investigate why all predictions go to a single class (F1=0.0).

**Usage:**
```bash
python debug_single_class_predictions.py \
    --experiment_dir store/lpm_llama_3b_wgmix_train_wgmix_test_last_token_prefix_layer27_euclidean \
    --dataset wgmix_test \
    --num_samples 50
```

**What it checks:**
- **Prototype Analysis:**
  - Prototype statistics (mean, std, norm)
  - Prototype separation (L2 distance)
  - Anomalies (zero variance, extreme values)
  
- **Prediction Analysis:**
  - Prediction distribution
  - Distance to each prototype
  - Distance variance
  - Sample-by-sample breakdown

**Expected output (healthy model):**
```
PROTOTYPE ANALYSIS:
  Prototype 'unharmful': norm=15.234
  Prototype 'harmful': norm=15.189
  Prototype Separation: L2 Distance=2.456

PREDICTION ANALYSIS:
  Prediction Distribution:
    unharmful: 28 (56.0%)
    harmful: 22 (44.0%)
  Accuracy: 0.6400

✅ No obvious issues detected
```

**Expected output (broken model):**
```
⚠️ PROTOTYPES ARE EXTREMELY CLOSE - classification will be poor!
⚠️ ALL PREDICTIONS ARE 'unharmful'!
⚠️ ALL samples are closer to 'unharmful'!

❌ CRITICAL ISSUES:
  • PROTOTYPES_TOO_CLOSE: relative distance=0.0001
  • SINGLE_CLASS_PREDICTION: ALL predictions are 'unharmful'
  • ALL_CLOSER_TO_UNHARMFUL: All distance differences are negative

RECOMMENDATIONS:
  1. Prototypes are nearly identical:
     → Check if activations were loaded correctly during training
     → Verify aggregation method is working properly
  2. All predictions go to single class:
     → Verify dataset/activation alignment
     → Run verify_activation_alignment.py script
```

**When to use:**
- When F1 score is 0.0 despite code completing
- When all predictions go to same class
- When accuracy matches class distribution (~0.55)
- To understand why model has no discriminative power

**Common issues detected:**
- `PROTOTYPES_TOO_CLOSE`: Prototypes are nearly identical
- `SINGLE_CLASS_PREDICTION`: All predictions are same class
- `ALL_CLOSER_TO_UNHARMFUL`: Systematic bias towards one class
- `LOW_DISTANCE_VARIANCE`: Model can't distinguish between classes
- `ZERO_VARIANCE`: Prototype has no variation (critical error)

---

## Typical Diagnostic Workflow

### Problem: Shape mismatch during inference

```bash
# 1. Verify tokenization consistency
python verify_tokenization.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset wgmix_test \
    --aggregation last_token_prefix

# 2. If mismatch found, check if prefix is applied consistently
# Look for differences in padding_side, max_length, or template application
```

### Problem: All predictions go to single class

```bash
# 1. Debug the model itself
python debug_single_class_predictions.py \
    --experiment_dir store/lpm_llama_3b_wgmix_train_wgmix_test_mean_layer27_euclidean \
    --dataset wgmix_test

# 2. If prototypes are too close, check activation alignment
python verify_activation_alignment.py \
    --dataset wgmix_train \
    --activation_run activations_maxlen_512_llama_3_2_3b_instruct_wgmix_train_layer27_...

# 3. If alignment issues found, verify dataset determinism
python verify_dataset_determinism.py --dataset wgmix_train
```

### Problem: Non-reproducible results

```bash
# 1. Check dataset loading
python verify_dataset_determinism.py --dataset wgmix_train

# 2. Check activation alignment
python verify_activation_alignment.py \
    --dataset wgmix_train \
    --activation_run <your_activation_run_id>

# 3. Check tokenization
python verify_tokenization.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset wgmix_test
```

## Error Reference

### Critical Severity
These issues will cause experiment failure or completely invalid results:
- `SAMPLE_COUNT_MISMATCH`: Dataset and activations have different sample counts
- `PROTOTYPES_TOO_CLOSE`: Classification is impossible due to identical prototypes
- `SINGLE_CLASS_PREDICTION`: Model predicts only one class
- `TOKEN_ID_MISMATCH`: Tokenization produces different results

### High Severity  
These issues will cause poor performance or silent failures:
- `LABEL_ALIGNMENT_ERROR`: Labels don't match between dataset and activations
- `TOKEN_COUNT_MISMATCH`: Sequence lengths differ between steps
- `ALL_CLOSER_TO_UNHARMFUL/HARMFUL`: Systematic bias in distances
- `LOW_DISTANCE_VARIANCE`: Model has poor discriminative power

### Medium Severity
These issues may affect performance but won't cause immediate failure:
- `UNEXPECTED_BATCH_SIZE`: Batching is inconsistent
- `PROTOTYPES_CLOSE`: Prototypes are close but not identical

## Output Files

All scripts write results to stdout. To save results:

```bash
python verify_dataset_determinism.py --dataset wgmix_train 2>&1 | tee determinism_check.log
```

## Notes

- All scripts use logging instead of print statements for better integration with experiment pipelines
- Scripts return exit code 0 on success, 1 on failure (useful for CI/CD)
- For cluster usage, update `DATASET_CONFIGS` paths if datasets are in different location
- Scripts are read-only and won't modify any data

## Troubleshooting

### "Dataset not found"
- Check that dataset parquet files exist in `store/datasets/`
- Verify paths in `DATASET_CONFIGS` dict match your setup
- For local testing, update paths to point to your local store

### "Activation run not found"
- Verify activation run ID is correct
- Check that activations exist in `store/runs/<activation_run_id>/`
- Ensure `--store_path` points to correct location

### "Model not found"
- Verify experiment directory path is correct
- Check that LPM model was saved in `<experiment_dir>/models/lpm_*.pt`
- Some failed experiments may not have saved models

## See Also

- [LPM Cluster Error Analysis](../../.llm_context/docs/lpm_cluster_error_analysis.md) - Comprehensive analysis of all errors from cluster runs
- [Run LPM Experiment README](../README_lpm_experiments.md) - Guide for running LPM experiments
- `run_lpm_experiment.py` - Main experiment script
- `slurm/run_lpm_experiments.sh` - SLURM batch script for cluster
