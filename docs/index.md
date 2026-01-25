<div align="center">
  <img src="logo.svg" alt="Mi-Crow Logo" width="300">
  
  # Mi-Crow
  
  **Python library for mechanistic interpretability research on Large Language Models**
  
  [![PyPI](https://img.shields.io/pypi/v/mi-crow)](https://pypi.org/project/mi-crow/)
  [![CI](https://github.com/AdamKaniasty/Inzynierka/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/AdamKaniasty/Inzynierka/actions)
  [![Docs](https://img.shields.io/badge/docs-gh--pages-blue)](https://adamkaniasty.github.io/Inzynierka/)
</div>

---

## What is Mi-Crow?

**Mi-Crow** is a Python library designed for researchers working on mechanistic interpretability of Large Language Models (LLMs). It provides a unified interface for analyzing and controlling model behavior through mechanistic interpretability methods, making it easy to understand what's happening inside neural networks.

### Key Capabilities

<div class="grid cards" markdown>

-   :material-robot:{ .lg .middle } __Activation Analysis__
    
    Save and analyze model activations from any layer with minimal performance overhead

-   :material-brain:{ .lg .middle } __SAE Training__
    
    Train sparse autoencoders to discover interpretable features and concepts

-   :material-lightbulb:{ .lg .middle } __Concept Discovery__
    
    Identify and name concepts learned by SAE neurons through automated analysis

-   :material-gamepad:{ .lg .middle } __Model Steering__
    
    Manipulate model behavior through concept-based interventions and activation control

-   :material-hook:{ .lg .middle } __Hook System__
    
    Flexible framework for intercepting and modifying activations at any layer

-   :material-database:{ .lg .middle } __Data Persistence__
    
    Efficient hierarchical storage for managing large-scale experiment data

</div>

## Quick Start

### Installation

```bash
pip install mi-crow
```

### Basic Usage

```python
from mi_crow.language_model import LanguageModel

# Initialize a language model
lm = LanguageModel(model_id="bielik")

# Run inference
outputs = lm.forwards(["Hello, world!"])

# Access activations and outputs
print(outputs.logits)
```

### Training an SAE

```python
from mi_crow.language_model import LanguageModel
from mi_crow.mechanistic.sae import SaeTrainer
from mi_crow.mechanistic.sae.modules import TopKSae

# Load model and collect activations
lm = LanguageModel(model_id="bielik")
activations = lm.save_activations(
    dataset=["Your text data here"],
    layers=["transformer_h_0_attn_c_attn"]
)

# Train SAE
trainer = SaeTrainer(
    model=lm,
    layer="transformer_h_0_attn_c_attn",
    sae_class=TopKSae,
    hyperparams={"epochs": 10, "batch_size": 256}
)
sae = trainer.train(activations)
```

---

## Documentation Structure

### 🚀 Getting Started

New to Mi-Crow? Start here:

- **[Installation Guide](guide/installation.md)** - Set up your environment
- **[Quick Start Tutorial](guide/quickstart.md)** - Run your first example in minutes
- **[Core Concepts](guide/core-concepts.md)** - Understand the fundamentals

### 📚 User Guide

Comprehensive guides for all features:

- **[Hooks System](guide/hooks/index.md)** - Complete guide to the powerful hooks framework
  - [Fundamentals](guide/hooks/fundamentals.md) - Core concepts
  - [Detectors](guide/hooks/detectors.md) - Observing activations
  - [Controllers](guide/hooks/controllers.md) - Modifying behavior
  - [Registration](guide/hooks/registration.md) - Hook management
  - [Advanced Usage](guide/hooks/advanced.md) - Advanced patterns

- **[Workflows](guide/workflows/index.md)** - Step-by-step guides for common tasks
  - [Saving Activations](guide/workflows/saving-activations.md)
  - [Training SAE Models](guide/workflows/training-sae.md)
  - [Concept Discovery](guide/workflows/concept-discovery.md)
  - [Concept Manipulation](guide/workflows/concept-manipulation.md)
  - [Activation Control](guide/workflows/activation-control.md)

- **[Best Practices](guide/best-practices.md)** - Tips for effective research
- **[Troubleshooting](guide/troubleshooting.md)** - Common issues and solutions
- **[Examples](guide/examples.md)** - Example notebooks overview

### 🧪 Experiments

Real-world experiments demonstrating Mi-Crow usage:

- **[Experiments Overview](experiments/index.md)** - Available experiments
- **[Verify SAE Training](experiments/verify-sae-training.md)** - Complete SAE training workflow
- **[SLURM Pipeline](experiments/slurm-pipeline.md)** - Distributed training setup

### 📖 API Reference

Complete API documentation:

- **[API Overview](api/index.md)** - API structure and organization
- **[Language Model](api/language_model.md)** - Model loading and inference
- **[SAE](api/sae.md)** - Sparse autoencoder APIs
- **[Datasets](api/datasets.md)** - Dataset loading and processing
- **[Store](api/store.md)** - Persistence layer
- **[Hooks](api/hooks.md)** - Hook system APIs

---

## Features

### Unified Model Interface

Work with any HuggingFace language model through a consistent API. No need to handle model-specific initialization details.

### Research-Focused Design

Built specifically for interpretability research workflows:

- **Comprehensive Testing**: 85%+ code coverage requirement
- **Type Safety**: Extensive use of Python type annotations
- **Documentation**: Complete API reference and user guides
- **CI/CD**: Automated testing and deployment
- **Minimal Overhead**: Hook system introduces negligible latency during inference

### Flexible Architecture

Five core modules that work independently or together:

1. **Language Model** - Unified interface for any HuggingFace model
2. **Hooks** - Flexible activation interception system
3. **Mechanistic** - SAE training and concept manipulation
4. **Store** - Hierarchical data persistence
5. **Datasets** - Dataset loading and processing

---

## Repository & Links

- **GitHub**: [AdamKaniasty/Inzynierka](https://github.com/AdamKaniasty/Inzynierka)
- **PyPI**: [mi-crow](https://pypi.org/project/mi-crow/)
- **Documentation**: This site

---

## Citation

If you use Mi-Crow in your research, please cite:

```bibtex
@thesis{kaniasty2025microw,
  title={Mechanistic Interpretability for Large Language Models: A Production-Ready Framework},
  author={Kaniasty, Adam and Kowalski, Hubert},
  year={2025},
  school={Warsaw University of Technology},
  note={Engineering Thesis}
}
```

---

## Next Steps

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __[Quick Start](guide/quickstart.md)__
    
    Get up and running in minutes

-   :material-book-open-variant:{ .lg .middle } __[User Guide](guide/index.md)__
    
    Comprehensive documentation

-   :material-code-tags:{ .lg .middle } __[Examples](guide/examples.md)__
    
    Explore example notebooks

-   :material-flask-outline:{ .lg .middle } __[Experiments](experiments/index.md)__
    
    Real-world use cases

</div>
