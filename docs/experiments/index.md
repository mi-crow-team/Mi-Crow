# Experiments

This section provides detailed walkthroughs of sample experiments demonstrating real-world usage of mi-crow.

## Overview

Experiments showcase complete workflows from data collection through analysis, using real models and datasets. They demonstrate best practices and provide templates for your own research.

## Available Experiments

### [Verify SAE Training](verify-sae-training.md)

Complete workflow for training and validating SAE models on the Bielik model using TinyStories dataset.

**What it covers**:
- Saving activations from a production model
- Training SAEs with proper hyperparameters
- Validating training success
- Concept discovery and naming
- Analysis and visualization

**Time required**: Several hours (depending on hardware)

**Prerequisites**:
- Access to Bielik model or similar
- Sufficient GPU memory
- Understanding of basic SAE concepts

### [SLURM SAE Pipeline](slurm-pipeline.md)

Distributed training setup for large-scale SAE training on cluster environments.

**What it covers**:
- SLURM job configuration
- Distributed activation saving
- Large-scale SAE training
- Resource management

**Time required**: Days (cluster-dependent)

**Prerequisites**:
- Access to SLURM cluster
- Understanding of cluster computing
- Large-scale dataset

## Experiment Structure

Each experiment typically includes:

1. **Setup**: Environment and dependencies
2. **Data Collection**: Saving activations
3. **Training**: SAE model training
4. **Validation**: Verifying results
5. **Analysis**: Understanding outcomes
6. **Documentation**: Recording findings

## Running Experiments

### Prerequisites

```bash
# Install dependencies
pip install -e .

# Or with uv
uv sync
```

### Basic Workflow

```bash
# 1. Navigate to experiment directory
cd experiments/verify_sae_training

# 2. Review README
cat README.md

# 3. Run scripts in order
python 01_save_activations.py
python 02_train_sae.py

# 4. Open analysis notebooks
jupyter notebook 03_analyze_training.ipynb
```

### Customization

Experiments are designed to be customizable:

- Modify model names
- Adjust hyperparameters
- Change dataset sources
- Adapt to your hardware

## Experiment Outputs

Experiments produce:

- **Saved activations**: Organized in store
- **Trained models**: SAE checkpoints
- **Analysis results**: Visualizations and metrics
- **Documentation**: Findings and observations

## Best Practices

1. **Start small**: Test with limited data first
2. **Monitor resources**: Watch memory and compute usage
3. **Document changes**: Record any modifications
4. **Save checkpoints**: Don't lose progress
5. **Validate results**: Verify outputs make sense

## Contributing Experiments

If you create a new experiment:

1. Create directory in `experiments/`
2. Include README with description
3. Provide runnable scripts/notebooks
4. Document setup and requirements
5. Share findings and observations

## Next Steps

- **[Verify SAE Training](verify-sae-training.md)** - Start with this experiment
- **[User Guide](../guide/index.md)** - Learn fundamentals first
- **[Examples](../guide/examples.md)** - Try examples before experiments

