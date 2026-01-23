# Performance Analysis Plots

This document describes each plot produced by `05_analyze_performance.py` for the SAE (Sparse Autoencoder) pipeline. The pipeline has three stages: **activation saving**, **SAE training**, and **inference**. Plots use synthetic hardware metrics for **Bielik 1.5B** and **Bielik 4.5B** on the **Polemo2** dataset unless otherwise noted.

---

## Model Metrics (Training & Inference)

### `training_loss.png`
- **Content**: SAE training loss vs epoch, one line per layer/run.
- **X-axis**: Epoch.
- **Y-axis**: Training loss (may be log-scaled if range is large).
- **Purpose**: Compare loss convergence across layers (e.g. L15, L20, L28, L38). Lower, monotonically decreasing curves indicate better convergence.

### `training_dynamics.png`
- **Content**: Evolution of training metrics over epochs: loss, R², dead features (%), L1 sparsity. Multiple series, one per layer/run.
- **X-axis**: Epoch.
- **Y-axis**: Metric value (loss, R², %, etc.); may use dual y-axes.
- **Purpose**: Show how loss, reconstruction quality (R²), feature sparsity, and dead feature fraction evolve together during SAE training.

### `batch_processing_times.png`
- **Content**: Inference batch processing time (seconds) vs batch index. Scatter points plus rolling median per job/layer.
- **X-axis**: Batch number.
- **Y-axis**: Processing time (seconds).
- **Purpose**: Inspect inference latency over time and variability across batches. One series per inference job/layer.

### `batch_processing_times_distribution.png`
- **Content**: Histogram of inference batch processing times (all jobs combined) with vertical lines for median, mean, and P95.
- **X-axis**: Processing time (seconds).
- **Y-axis**: Frequency.
- **Purpose**: Summarize the distribution of inference batch latencies (typical values and tail percentiles).

---

## Hardware Metrics — Memory (VRAM & RAM)

### `memory_usage_activation_saving.png`, `memory_usage_training.png`, `memory_usage_inference.png`
- **Content**: Two subplots per file — **VRAM usage (left)** and **RAM usage (right)** over time. Two lines per subplot: **Bielik 1.5B** (blue) and **Bielik 4.5B** (orange).
- **X-axis**: Time elapsed (hours) from job start.
- **Y-axis**: VRAM (GB) or RAM (GB).
- **Purpose**: Compare memory consumption between 1.5B and 4.5B across pipeline stages. 4.5B typically uses more VRAM and RAM.

### `memory_usage_boxplots_vram_activation_saving.png`, `_training.png`, `_inference.png`
- **Content**: Boxplots of VRAM usage (GB) for that stage. Two boxes side-by-side: **Bielik 1.5B** and **Bielik 4.5B**. Median, mean, and optional stats annotations.
- **X-axis**: Model (1.5B vs 4.5B).
- **Y-axis**: VRAM usage (GB).
- **Purpose**: Compare VRAM distribution (median, spread, outliers) between 1.5B and 4.5B for each stage.

### `memory_usage_boxplots_ram_activation_saving.png`, `_training.png`, `_inference.png`
- **Content**: Same structure as VRAM boxplots, but for **RAM usage (GB)**.
- **X-axis**: Model (1.5B vs 4.5B).
- **Y-axis**: RAM usage (GB).
- **Purpose**: Compare RAM distribution between 1.5B and 4.5B for each stage.

---

## Hardware Metrics — Utilization (GPU & CPU)

### `hardware_utilization_distribution_activation_saving_gpu.png`, `_training_gpu.png`, `_inference_gpu.png`
- **Content**: Overlapping **density** histograms of GPU utilization (%) for **Bielik 1.5B** (blue) and **Bielik 4.5B** (orange). Vertical dashed lines mark medians.
- **X-axis**: GPU utilization (%).
- **Y-axis**: Density (normalized so distributions are comparable across different job durations).
- **Purpose**: Compare GPU utilization distributions between 1.5B and 4.5B. 4.5B is typically shifted toward higher utilization.

### `hardware_utilization_distribution_activation_saving_cpu.png`, `_training_cpu.png`, `_inference_cpu.png`
- **Content**: Same as GPU utilization plots, but for **CPU utilization (%)**.
- **X-axis**: CPU utilization (%).
- **Y-axis**: Density.
- **Purpose**: Compare CPU utilization distributions between 1.5B and 4.5B per stage.

---

## Hardware Metrics — Disk

### `disk_usage_activation_saving.png`, `disk_usage_training.png`, `disk_usage_inference.png`
- **Content**: Cumulative disk usage (GB) over time. One or two lines per plot: **Bielik 1.5B** and **Bielik 4.5B** (when both exist). Baseline inference is excluded.
- **X-axis**: Time elapsed (hours).
- **Y-axis**: Cumulative disk usage (GB).
- **Purpose**: Show how disk storage grows during each stage (e.g. activation buffers, checkpoints, inference outputs). 4.5B generally uses more disk.

---

## Inference: With vs Without Detectors

### `inference_comparison_bielik12.png`, `inference_comparison_bielik45.png`
- **Content**: One figure per model. **5 rows × 2 columns**. Each row is a metric; columns are **“With detectors”** (left) and **“Baseline (no detectors)”** (right). Metrics: CPU utilization (%), GPU utilization (%), RAM (GB), VRAM (MiB), disk (GB) over time.
- **X-axis**: Time (hours).
- **Y-axis**: Metric-specific (%, GB, MiB).
- **Purpose**: Compare inference **with** SAE/detector overhead vs **without**. Detectors add CPU and disk usage; baseline shows lower CPU and disk, often higher GPU utilization.

---

## Summary

| Category        | Plot pattern(s)                                             | Main comparison                          |
|----------------|-------------------------------------------------------------|------------------------------------------|
| Training       | `training_loss`, `training_dynamics`                        | Loss and metrics by layer over epochs    |
| Inference time | `batch_processing_times`, `batch_processing_times_distribution` | Latency over batches and overall distribution |
| Memory         | `memory_usage_*`, `memory_usage_boxplots_*`                 | VRAM/RAM by stage, 1.5B vs 4.5B          |
| Utilization    | `hardware_utilization_distribution_*_gpu`, `*_cpu`          | GPU/CPU utilization by stage, 1.5B vs 4.5B |
| Disk           | `disk_usage_*`                                              | Cumulative disk growth by stage          |
| Inference vs baseline | `inference_comparison_bielik12`, `inference_comparison_bielik45` | With vs without detector overhead        |

Colors: **Bielik 1.5B** = blue (`#1a5276`), **Bielik 4.5B** = orange (`#d35400`). Utilization histograms use density; memory/disk use raw GB/MiB.
