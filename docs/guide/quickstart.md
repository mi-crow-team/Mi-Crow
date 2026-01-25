# Quick Start

Get up and running with mi-crow in minutes! This tutorial will walk you through a minimal example that demonstrates the core workflow.

## Minimal Example

Let's start with the simplest possible example: loading a model and running inference.

```python
from mi_crow.language_model import LanguageModel
from mi_crow.store import LocalStore
import torch

# Create a store for saving data
store = LocalStore(base_path="./store")

# Choose device: use GPU when available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a small model for testing on the chosen device
lm = LanguageModel.from_huggingface(
    "sshleifer/tiny-gpt2",
    store=store,
    device=device,
)

# Run inference
texts = ["Hello, world!", "How are you?"]
outputs, encodings = lm.inference.execute_inference(texts)

print(outputs.logits.shape)  # (batch_size, seq_len, vocab_size)
```

## Basic SAE Workflow

The core mi-crow workflow consists of three main steps:

### Step 1: Save Activations

First, we need to collect activations from a model layer:

```python
from mi_crow.datasets import TextDataset

# Create a simple dataset
dataset = TextDataset(texts=["The cat sat on the mat."] * 100)

# Save activations from a layer
run_id = lm.activations.save(
    layer_signature="transformer.h.0.attn.c_attn",  # Layer name
    dataset=dataset,
    sample_limit=100,
    batch_size=4
)

print(f"Saved activations with run_id: {run_id}")
```

### Step 2: Train an SAE

Train a sparse autoencoder on the saved activations:

```python
from mi_crow.mechanistic.sae import TopKSae, SaeTrainer
from mi_crow.mechanistic.sae.train import SaeTrainingConfig

# Create SAE model
sae = TopKSae(
    n_latents=512,  # Number of SAE neurons
    n_inputs=768,   # Must match layer activation size
    k=8,            # Top-K sparsity
    device="cpu"
)

# Configure training
config = SaeTrainingConfig(
    epochs=10,
    batch_size=256,
    lr=1e-3
)

# Train the SAE
trainer = SaeTrainer(sae)
history = trainer.train(
    store=store,
    run_id=run_id,
    layer_signature="transformer.h.0.attn.c_attn",
    config=config
)

print("Training complete!")
```

### Step 3: Use Concepts

Attach the SAE to the model and use it for concept discovery:

```python
# Attach SAE to model
lm.attach_sae(sae, layer_signature="transformer.h.0.attn.c_attn")

# Enable text tracking to see what activates each neuron
sae.concepts.enable_text_tracking(top_k=5)

# Run inference with SAE attached
outputs, encodings = lm.inference.execute_inference(["The cat sat on the mat."])

# Get top texts for each neuron
top_texts = sae.concepts.get_top_texts()
print(f"Found {len(top_texts)} neurons with tracked texts")
```

## Complete Example

Here's a complete, runnable example:

```python
from mi_crow.language_model import LanguageModel
from mi_crow.store import LocalStore
from mi_crow.datasets import TextDataset
from mi_crow.mechanistic.sae import TopKSae
from mi_crow.mechanistic.sae.train import SaeTrainer, SaeTrainingConfig

# Setup
store = LocalStore(base_path="./store")

# Choose device: GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

lm = LanguageModel.from_huggingface("sshleifer/tiny-gpt2", store=store, device=device)

# Step 1: Save activations
dataset = TextDataset(texts=["The cat sat on the mat."] * 50)
run_id = lm.activations.save(
    layer_signature="transformer.h.0.attn.c_attn",
    dataset=dataset,
    sample_limit=50,
    batch_size=4
)

# Step 2: Train SAE
sae = TopKSae(n_latents=256, n_inputs=768, k=4, device="cpu")
trainer = SaeTrainer(sae)
config = SaeTrainingConfig(epochs=5, batch_size=64, lr=1e-3)
history = trainer.train(store, run_id, "transformer.h.0.attn.c_attn", config)

# Step 3: Use SAE
lm.attach_sae(sae, "transformer.h.0.attn.c_attn")
sae.concepts.enable_text_tracking(top_k=3)
outputs, encodings = lm.inference.execute_inference(["The cat sat on the mat."])
top_texts = sae.concepts.get_top_texts()

print("Quick start complete!")
```

## What's Next?

Now that you've run the basic workflow, explore more:

- **[Core Concepts](core-concepts.md)** - Understand SAEs, concepts, and hooks
- **[Hooks System](hooks/index.md)** - Learn about the powerful hooks framework
- **[Saving Activations](workflows/saving-activations.md)** - Detailed guide for activation collection
- **[Training SAE Models](workflows/training-sae.md)** - Advanced SAE training techniques
- **[Examples](examples.md)** - More detailed example notebooks

## Tips

- Start with small models like `sshleifer/tiny-gpt2` for quick experimentation
- Use `sample_limit` to control dataset size during development
- Check layer names with `lm.layers.list_layers()` before saving activations
- Monitor training with `history` metrics or enable wandb logging

