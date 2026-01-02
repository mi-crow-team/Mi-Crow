# User Guide

Welcome to the mi-crow user guide! This comprehensive guide will help you understand and use the mi-crow library for mechanistic interpretability research.

## What is mi-crow?

mi-crow is a Python package for explaining and steering LLM behavior using Sparse Autoencoders (SAE) and concepts. It provides a complete toolkit for:

- **Activation Analysis**: Save and analyze model activations from any layer
- **SAE Training**: Train sparse autoencoders to discover interpretable features
- **Concept Discovery**: Identify and name concepts learned by SAE neurons
- **Model Steering**: Manipulate model behavior through concept-based interventions
- **Hook System**: Flexible system for intercepting and modifying activations

## What is Mechanistic Interpretability?

Mechanistic interpretability is the study of understanding how neural networks work by reverse-engineering their internal computations. In the context of language models, this means:

- Understanding what features the model learns at different layers
- Identifying how these features combine to produce outputs
- Discovering interpretable concepts that correspond to human-understandable ideas
- Using this understanding to control and improve model behavior

## Library Capabilities

mi-crow provides a modular architecture for mechanistic interpretability research:

- **Language Model Wrapper**: Easy loading and inference with HuggingFace models
- **Sparse Autoencoders**: Train and use SAEs to discover interpretable features
- **Hooks System**: Powerful framework for observing and modifying activations
- **Store**: Hierarchical storage for activations, models, and metadata
- **Datasets**: Flexible data loading from HuggingFace or local files

## Getting Started

1. **[Installation](installation.md)** - Set up mi-crow and its dependencies
2. **[Quick Start](quickstart.md)** - Run your first example in minutes
3. **[Core Concepts](core-concepts.md)** - Understand the fundamental ideas
4. **[Hooks System](hooks/index.md)** - Learn about the powerful hooks framework
5. **[Workflows](workflows/index.md)** - Step-by-step guides for common tasks

## Documentation Structure

### Core Documentation

- **[Installation & Setup](installation.md)** - Installation and environment configuration
- **[Quick Start](quickstart.md)** - Get up and running quickly
- **[Core Concepts](core-concepts.md)** - Fundamental concepts and architecture

### Hooks System

The hooks system is the foundation of mi-crow's interpretability capabilities:

- **[Hooks Overview](hooks/index.md)** - Introduction to the hooks system
- **[Hooks Fundamentals](hooks/fundamentals.md)** - Core concepts and lifecycle
- **[Detector Hooks](hooks/detectors.md)** - Observing activations without modification
- **[Controller Hooks](hooks/controllers.md)** - Modifying activations during inference
- **[Hook Registration](hooks/registration.md)** - Managing hooks on layers
- **[Advanced Hooks](hooks/advanced.md)** - Advanced patterns and best practices

### Workflows

Step-by-step guides for common tasks:

- **[Workflows Overview](workflows/index.md)** - When to use each workflow
- **[Saving Activations](workflows/saving-activations.md)** - Collect activation data
- **[Training SAE Models](workflows/training-sae.md)** - Train sparse autoencoders
- **[Concept Discovery](workflows/concept-discovery.md)** - Find interpretable concepts
- **[Concept Manipulation](workflows/concept-manipulation.md)** - Control model behavior
- **[Activation Control](workflows/activation-control.md)** - Direct activation manipulation

### Additional Resources

- **[Best Practices](best-practices.md)** - Tips for effective research
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Examples](examples.md)** - Example notebooks and learning path
- **[Experiments](../experiments/index.md)** - Detailed experiment walkthroughs

## Next Steps

If you're new to mi-crow, we recommend following this path:

1. Start with [Installation](installation.md) to set up your environment
2. Run through the [Quick Start](quickstart.md) tutorial
3. Read [Core Concepts](core-concepts.md) to understand the fundamentals
4. Explore the [Hooks System](hooks/index.md) - it's central to everything
5. Try a [Workflow](workflows/index.md) that matches your research goals
6. Check out [Examples](examples.md) for more detailed code

For API reference, see the [API Documentation](../api/index.md).

