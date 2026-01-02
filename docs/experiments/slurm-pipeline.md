# SLURM SAE Pipeline

This experiment demonstrates distributed training setup for large-scale SAE training on cluster environments using SLURM.

## Overview

This pipeline shows how to:
- Configure SLURM jobs for activation saving
- Set up distributed SAE training on clusters
- Manage resources and job dependencies
- Handle large-scale datasets efficiently

## Prerequisites

- Access to SLURM cluster
- Understanding of cluster computing
- Large-scale dataset
- Sufficient cluster resources

## Experiment Structure

```
slurm_sae_pipeline/
├── 01_save_activations.py      # Activation saving script
├── 02_train_sae.py              # SAE training script
├── submit_save_activations.sh   # SLURM submission script for activations
├── submit_train_sae.sh         # SLURM submission script for training
└── README.md                    # Pipeline documentation
```

## Configuration

### SLURM Job Configuration

The submission scripts configure:
- **Resources**: GPU allocation, memory, time limits
- **Dependencies**: Job ordering (train after save)
- **Environment**: Python environment setup
- **Output**: Logging and error handling

### mi-crow Configuration

The Python scripts use standard mi-crow APIs:
- `lm.activations.save()` for distributed activation saving
- `SaeTrainer.train()` for SAE training
- Store configuration for cluster filesystems

## Workflow

1. **Submit activation saving job**: Uses `submit_save_activations.sh`
2. **Wait for completion**: Activation saving must complete first
3. **Submit training job**: Uses `submit_train_sae.sh` (depends on step 1)
4. **Monitor jobs**: Track progress through SLURM

## Key Features

- **Distributed processing**: Handles large datasets across cluster nodes
- **Resource management**: Proper GPU and memory allocation
- **Job dependencies**: Ensures correct execution order
- **Error handling**: Robust failure recovery

## Customization

Adapt the pipeline for your cluster:
- Modify resource requests in submission scripts
- Adjust batch sizes for available memory
- Configure store paths for cluster filesystem
- Set appropriate time limits

## Related Documentation

- **[Training SAE Models](../guide/workflows/training-sae.md)** - SAE training guide
- **[Saving Activations](../guide/workflows/saving-activations.md)** - Activation saving guide
- **[Best Practices](../guide/best-practices.md)** - Performance optimization

