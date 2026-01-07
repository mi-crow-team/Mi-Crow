# Core Concepts

This guide explains the fundamental concepts and architecture of mi-crow. Understanding these concepts will help you use the library effectively.

## Language Models in mi-crow

mi-crow provides a unified interface for working with language models through the `LanguageModel` class. It wraps PyTorch models (typically from HuggingFace) and provides:

- **Inference**: Run forward passes with `forwards()` and generation with `generate()`
- **Layer Access**: Inspect and manipulate individual layers
- **Activation Saving**: Collect activations from any layer
- **Hook Integration**: Attach hooks for observation and control

### Model Loading

```python
from mi_crow.language_model import LanguageModel
from mi_crow.store import LocalStore
import torch

store = LocalStore(base_path="./store")

# Use GPU when available
device = "cuda" if torch.cuda.is_available() else "cpu"

lm = LanguageModel.from_huggingface(
    "gpt2",  # Or any HuggingFace model
    store=store,
    device=device,
)
```

The model is automatically tokenized and ready for inference. You can access layers through `lm.layers` and run inference with `lm.inference.execute_inference()`.

## Sparse Autoencoders (SAE)

Sparse Autoencoders are the core interpretability tool in mi-crow. They learn to represent model activations using a sparse set of interpretable features.

### What are SAEs?

An SAE is a neural network that:
1. Takes dense activations from a model layer as input
2. Encodes them into a sparse latent representation
3. Decodes back to reconstruct the original activations

The sparsity constraint encourages the SAE to learn discrete, interpretable features (neurons) that correspond to meaningful concepts.

### Why Use SAEs?

- **Interpretability**: Each SAE neuron often corresponds to a human-understandable concept
- **Feature Discovery**: Automatically discover what features the model uses
- **Control**: Manipulate model behavior by amplifying or suppressing specific neurons
- **Analysis**: Understand which features are important for different tasks

### SAE Architecture

mi-crow supports TopK SAEs, which enforce sparsity by keeping only the top-K most active neurons:

```python
from mi_crow.mechanistic.sae import TopKSae

sae = TopKSae(
    n_latents=4096,  # Number of SAE neurons (overcomplete)
    n_inputs=768,    # Size of input activations
    k=32,            # Top-K sparsity (only 32 neurons active at once)
    device="cuda"
)
```

The overcomplete ratio (`n_latents / n_inputs`) determines how many features the SAE can learn. Higher ratios allow more fine-grained feature discovery.

## Concepts and Interpretability

Concepts are human-interpretable meanings associated with SAE neurons. The concept discovery process involves:

1. **Training an SAE** on model activations
2. **Collecting top texts** that activate each neuron
3. **Manual curation** to name concepts based on patterns
4. **Using concepts** to understand and control model behavior

### Concept Discovery

After training an SAE, you can discover concepts by:

```python
# Enable text tracking during inference
sae.concepts.enable_text_tracking(top_k=10)

# Run inference on a dataset
outputs, encodings = lm.inference.execute_inference(dataset_texts)

# Get top activating texts for each neuron
top_texts = sae.concepts.get_top_texts()
```

Each neuron's top texts reveal what patterns it detects, allowing you to assign meaningful names like "family relationships" or "scientific terminology".

### Concept Manipulation

Once concepts are identified, you can manipulate model behavior:

```python
# Amplify a concept (neuron 42)
sae.concepts.manipulate_concept(neuron_idx=42, scale=1.5)

# Suppress a concept
sae.concepts.manipulate_concept(neuron_idx=42, scale=0.5)

# Run inference with modified behavior
outputs, encodings = lm.inference.execute_inference(["Your prompt here"])
```

## Hooks System Overview

The hooks system is the foundation of mi-crow's interpretability capabilities. It allows you to intercept and process activations during model inference.

### What are Hooks?

Hooks are callbacks that execute at specific points during a model's forward pass. They can:

- **Observe** activations without modification (Detectors)
- **Modify** activations to change model behavior (Controllers)

### Hook Types

- **Detectors**: Collect data, save activations, track statistics
- **Controllers**: Modify inputs or outputs to steer model behavior

### Why Hooks Matter

Hooks enable:
- **Non-invasive inspection**: Observe model internals without changing code
- **Flexible control**: Modify behavior at any layer
- **Composable interventions**: Combine multiple hooks for complex experiments
- **SAE integration**: SAEs work as both detectors and controllers

For detailed information about hooks, see the [Hooks System Guide](hooks/index.md).

## Store Architecture

The Store provides a hierarchical persistence layer for:

- **Activations**: Saved layer activations organized by run
- **Models**: Trained SAE models and checkpoints
- **Metadata**: Training history, configurations, and run information

### Store Structure

```
store/
├── activations/
│   └── <run_id>/
│       ├── batch_0/
│       │   └── <layer_name>/
│       │       └── activations.safetensors
│       └── meta.json
├── runs/
│   └── <run_id>/
│       └── training_history.json
└── sae_models/
    └── <sae_id>/
        └── model.pt
```

### LocalStore

The default implementation uses the local filesystem:

```python
from mi_crow.store import LocalStore

store = LocalStore(base_path="./store")
```

All operations automatically organize data in this hierarchical structure, making it easy to manage large-scale experiments.

## Datasets and Data Loading

mi-crow provides flexible dataset loading for:

- **HuggingFace datasets**: Direct integration with HF datasets
- **Local files**: Load from text files or custom formats
- **In-memory data**: Use Python lists directly

### TextDataset

For simple text data:

```python
from mi_crow.datasets import TextDataset

dataset = TextDataset(texts=["Text 1", "Text 2", "Text 3"])
```

### HuggingFace Integration

```python
from mi_crow.datasets import HuggingFaceDataset

dataset = HuggingFaceDataset(
    name="wikitext",
    split="train",
    text_field="text"
)
```

Datasets are automatically batched and tokenized when used with `lm.activations.save()` or during inference.

## Putting It All Together

The typical mi-crow workflow combines these concepts:

1. **Load a model** and create a store
2. **Save activations** from a layer using hooks (detectors)
3. **Train an SAE** on the activations
4. **Discover concepts** by analyzing neuron activations
5. **Manipulate concepts** to control model behavior (controllers)
6. **Analyze results** using the store's organized data

Each component is designed to work seamlessly with the others, providing a complete toolkit for mechanistic interpretability research.

## Next Steps

- **[Hooks System](hooks/index.md)** - Deep dive into the hooks framework
- **[Saving Activations](workflows/saving-activations.md)** - Detailed activation collection guide
- **[Training SAE Models](workflows/training-sae.md)** - SAE training best practices
- **[Concept Discovery](workflows/concept-discovery.md)** - Finding interpretable concepts

