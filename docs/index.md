# mi_crow

mi_crow is a Python package for explaining and steering LLM behavior using Sparse Autoencoders (SAE) and concepts.

## Quick Links

- **[User Guide](guide/index.md)** - Comprehensive guide to using mi-crow
- **[Experiments](experiments/index.md)** - Sample experiments and walkthroughs
- **[API Reference](api/index.md)** - Complete API documentation

## What is mi-crow?

mi-crow provides a complete toolkit for mechanistic interpretability research:

- **Activation Analysis**: Save and analyze model activations from any layer
- **SAE Training**: Train sparse autoencoders to discover interpretable features
- **Concept Discovery**: Identify and name concepts learned by SAE neurons
- **Model Steering**: Manipulate model behavior through concept-based interventions
- **Hook System**: Flexible framework for intercepting and modifying activations

## Getting Started

1. **[Installation](guide/installation.md)** - Set up mi-crow
2. **[Quick Start](guide/quickstart.md)** - Run your first example
3. **[Core Concepts](guide/core-concepts.md)** - Understand the fundamentals
4. **[Hooks System](guide/hooks/index.md)** - Learn about the powerful hooks framework

## Documentation Structure

### User Guide

The user guide provides comprehensive documentation:

- **[Installation & Setup](guide/installation.md)** - Environment configuration
- **[Quick Start](guide/quickstart.md)** - Get up and running quickly
- **[Core Concepts](guide/core-concepts.md)** - Fundamental concepts and architecture
- **[Hooks System](guide/hooks/index.md)** - Complete hooks documentation
- **[Workflows](guide/workflows/index.md)** - Step-by-step guides for common tasks
- **[Best Practices](guide/best-practices.md)** - Tips for effective research
- **[Troubleshooting](guide/troubleshooting.md)** - Common issues and solutions
- **[Examples](guide/examples.md)** - Example notebooks overview

### Experiments

Real-world experiments demonstrating mi-crow usage:

- **[Experiments Overview](experiments/index.md)** - Available experiments
- **[Verify SAE Training](experiments/verify-sae-training.md)** - Complete SAE training workflow
- **[SLURM Pipeline](experiments/slurm-pipeline.md)** - Distributed training setup

### API Reference

Complete API documentation:

- **[API Overview](api/index.md)** - API structure
- **[Language Model](api/language_model.md)** - Model loading and inference
- **[SAE](api/sae.md)** - Sparse autoencoder APIs
- **[Datasets](api/datasets.md)** - Dataset loading
- **[Store](api/store.md)** - Persistence layer
- **[Hooks](api/hooks.md)** - Hook system APIs

## Key Features

### Hooks System

The hooks system is the foundation of mi-crow's capabilities:

- **Detectors**: Observe activations without modification
- **Controllers**: Modify activations to change behavior
- **Flexible**: Register on any layer, compose multiple hooks
- **Non-invasive**: No model code changes required

See the [Hooks System Guide](guide/hooks/index.md) for details.

### Sparse Autoencoders

Train SAEs to discover interpretable features:

- **TopK SAE**: Efficient sparse autoencoder implementation
- **Concept Discovery**: Find what each neuron represents
- **Model Control**: Manipulate behavior through concepts

See [Training SAE Models](guide/workflows/training-sae.md) for details.

### Workflows

Complete workflows for common tasks:

- [Saving Activations](guide/workflows/saving-activations.md)
- [Training SAE Models](guide/workflows/training-sae.md)
- [Concept Discovery](guide/workflows/concept-discovery.md)
- [Concept Manipulation](guide/workflows/concept-manipulation.md)
- [Activation Control](guide/workflows/activation-control.md)

## Repository

- **GitHub**: [AdamKaniasty/Inzynierka](https://github.com/AdamKaniasty/Inzynierka)
- **Documentation**: This site

## Next Steps

- Start with the [User Guide](guide/index.md)
- Try the [Quick Start](guide/quickstart.md) tutorial
- Explore [Examples](guide/examples.md)
- Check out [Experiments](experiments/index.md)
