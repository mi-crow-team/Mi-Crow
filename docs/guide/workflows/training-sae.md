# Training SAE Models

This guide covers training sparse autoencoders (SAEs) on saved activations to discover interpretable features.

## Overview

Training an SAE involves:
1. Loading saved activations
2. Creating an SAE model
3. Configuring training parameters
4. Training the SAE
5. Saving the trained model

## Prerequisites

Before training, you need:
- Saved activations (see [Saving Activations](saving-activations.md))
- A run_id from the activation saving step
- Knowledge of the activation dimensions

## Step 1: Load Saved Activations

```python
from mi_crow.store import LocalStore

store = LocalStore(base_path="./store")

# Use the run_id from saving activations
run_id = "your-run-id-here"

# Verify activations exist
from pathlib import Path
run_path = Path(f"store/activations/{run_id}")
assert run_path.exists(), f"Run {run_id} not found"
```

## Step 2: Determine Activation Dimensions

You need to know the activation size for your layer:

```python
# Check metadata
import json
with open(f"store/activations/{run_id}/meta.json") as f:
    metadata = json.load(f)

# Activation size is typically in metadata
# Or inspect a saved activation file
```

Common activation sizes:
- GPT-2 small: 768
- GPT-2 medium: 1024
- GPT-2 large: 1280
- BERT base: 768

## Step 3: Create SAE Model

```python
from mi_crow.mechanistic.sae import TopKSae

# Create SAE
sae = TopKSae(
    n_latents=4096,  # Number of SAE neurons (overcomplete)
    n_inputs=768,    # Must match activation size
    k=32,            # Top-K sparsity (only 32 neurons active)
    device="cuda"    # or "cpu" or "mps"
)
```

### SAE Architecture Choices

**Overcompleteness Ratio**: `n_latents / n_inputs`
- 2x: Fewer features, faster training
- 4x: Balanced (common choice)
- 8x+: More features, slower training

**Top-K Sparsity**: `k`
- Smaller k: More sparse, fewer active neurons
- Larger k: Less sparse, more active neurons
- Typical: 8-32

## Step 4: Configure Training

```python
from mi_crow.mechanistic.sae.train import SaeTrainingConfig

config = SaeTrainingConfig(
    epochs=100,           # Number of training epochs
    batch_size=256,       # Training batch size
    lr=1e-3,             # Learning rate
    l1_lambda=1e-4,       # L1 regularization strength
    use_wandb=False,      # Enable Weights & Biases logging
    wandb_project="sae"   # W&B project name
)
```

### Hyperparameter Selection

**Learning Rate**:
- Start with `1e-3`
- Reduce if training is unstable
- Increase if convergence is slow

**L1 Lambda**:
- Controls sparsity
- Higher = more sparse
- Typical: `1e-4` to `1e-3`

**Batch Size**:
- Larger = faster training
- Limited by GPU memory
- Typical: 128-512

**Epochs**:
- Depends on dataset size
- Monitor loss to determine convergence
- Typical: 50-200

## Step 5: Train the SAE

```python
from mi_crow.mechanistic.sae.train import SaeTrainer

# Create trainer
trainer = SaeTrainer(sae)

# Train
history = trainer.train(
    store=store,
    run_id=run_id,
    layer_signature="transformer.h.0.attn.c_attn",
    config=config
)

print("Training complete!")
```

### Training Output

The `history` object contains:
- `loss`: Reconstruction loss over time
- `r2`: R² score (reconstruction quality)
- `l0`: L0 norm (sparsity)
- `dead_features`: Number of dead neurons

## Step 6: Monitor Training

### Check Training Progress

```python
# Access training history
print(f"Final loss: {history['loss'][-1]}")
print(f"Final R²: {history['r2'][-1]}")
print(f"Final L0: {history['l0'][-1]}")
```

### Visualize Training

```python
import matplotlib.pyplot as plt

# Plot loss
plt.plot(history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Plot R²
plt.plot(history['r2'])
plt.xlabel('Epoch')
plt.ylabel('R²')
plt.title('Reconstruction Quality')
plt.show()
```

### Weights & Biases Integration

```python
config = SaeTrainingConfig(
    epochs=100,
    batch_size=256,
    lr=1e-3,
    use_wandb=True,        # Enable W&B
    wandb_project="sae",   # Project name
    wandb_run_name="gpt2-layer0"  # Run name
)
```

## Step 7: Save Trained Model

The trainer automatically saves the model, but you can also save manually:

```python
# Model is saved automatically during training
# Check store/runs/<training_run_id>/

# Or save manually
import torch
torch.save(sae.state_dict(), "sae_model.pt")
```

## Loading a Trained SAE

```python
from mi_crow.mechanistic.sae import TopKSae

# Create SAE with same architecture
sae = TopKSae(n_latents=4096, n_inputs=768, k=32, device="cuda")

# Load weights
sae.load_state_dict(torch.load("sae_model.pt"))
sae.eval()  # Set to evaluation mode
```

## Training Tips

### Start Small

```python
# Start with small SAE for testing
sae = TopKSae(
    n_latents=512,   # Small overcompleteness
    n_inputs=768,
    k=8,             # Small sparsity
    device="cpu"      # CPU for testing
)

config = SaeTrainingConfig(
    epochs=10,        # Few epochs for testing
    batch_size=64
)
```

### Monitor Dead Features

Dead features (neurons that never activate) indicate:
- Too much sparsity (increase k or reduce l1_lambda)
- Learning rate too high
- Not enough training data

```python
# Check dead features
dead_count = history['dead_features'][-1]
total_features = sae.n_latents
dead_ratio = dead_count / total_features

if dead_ratio > 0.1:  # More than 10% dead
    print("Warning: Many dead features!")
```

### Verify Learning

```python
# Check that weights are learning (not uniform)
weight_variance = sae.encoder.weight.var().item()
print(f"Weight variance: {weight_variance}")

# Should be > 0.01 for learned features
if weight_variance < 0.01:
    print("Warning: Weights may not be learning!")
```

## Common Issues

### Out of Memory

```python
# Solution: Reduce batch size
config = SaeTrainingConfig(
    epochs=100,
    batch_size=64,  # Smaller batch
    lr=1e-3
)
```

### Training Instability

```python
# Solution: Reduce learning rate
config = SaeTrainingConfig(
    epochs=100,
    batch_size=256,
    lr=1e-4,  # Lower learning rate
    l1_lambda=1e-5  # Lower regularization
)
```

### Poor Reconstruction

```python
# Solution: Increase model capacity
sae = TopKSae(
    n_latents=8192,  # More neurons
    n_inputs=768,
    k=64,            # More active neurons
    device="cuda"
)
```

## Next Steps

After training an SAE:

- **[Concept Discovery](concept-discovery.md)** - Find what each neuron represents
- **[Concept Manipulation](concept-manipulation.md)** - Use SAE to control model
- **[Hooks: Advanced](../hooks/advanced.md)** - SAE as detector and controller

## Related Examples

- `examples/01_train_sae_model.ipynb` - Complete SAE training example
- `experiments/verify_sae_training/` - Detailed training experiment

