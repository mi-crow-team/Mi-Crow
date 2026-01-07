# Best Practices

This guide covers best practices for using mi-crow effectively in your research.

## Model Selection for Experimentation

### Start Small

- Use small models (e.g., `sshleifer/tiny-gpt2`) for initial experiments
- Faster iteration and lower memory requirements
- Easier to understand and debug

### Scale Gradually

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Start with tiny model
lm = LanguageModel.from_huggingface("sshleifer/tiny-gpt2", store=store, device=device)

# Then move to small
lm = LanguageModel.from_huggingface("gpt2", store=store, device=device)

# Finally use larger models
lm = LanguageModel.from_huggingface("gpt2-large", store=store, device=device)
```

### Consider Your Goals

- **Quick prototyping**: Use tiny/small models
- **Production research**: Use appropriately sized models
- **Memory constraints**: Start small, scale if needed

## Memory Management

### Activation Saving

```python
# Use appropriate batch sizes
run_id = lm.activations.save(
    layer_signature="layer_0",
    dataset=dataset,
    batch_size=4,  # Start small
    sample_limit=100
)
```

### SAE Training

```python
# Monitor memory during training
config = SaeTrainingConfig(
    batch_size=128,  # Adjust based on available memory
    epochs=100
)
```

### Hook Management

```python
# Always unregister hooks
try:
    hook_id = lm.layers.register_hook("layer_0", detector)
    # ... use hook ...
finally:
    lm.layers.unregister_hook(hook_id)  # Critical!
```

### Move to CPU

```python
# Move large tensors to CPU
activations = detector.get_captured()
activations_cpu = activations.detach().cpu()  # Saves GPU memory
```

## Performance Optimization

### Batch Processing

```python
# Process in larger batches when possible
run_id = lm.activations.save(
    layer_signature="layer_0",
    dataset=dataset,
    batch_size=32,  # Larger = faster (if memory allows)
    sample_limit=1000
)
```

### Device Selection

```python
# Use GPU when available
device = "cuda" if torch.cuda.is_available() else "cpu"
sae = TopKSae(n_latents=4096, n_inputs=768, k=32, device=device)
```

### Efficient Data Loading

```python
# Use appropriate dataset sizes
dataset = TextDataset(texts=texts[:1000])  # Limit for testing
# Then scale up for full experiments
```

## Experiment Organization

### Naming Conventions

```python
# Use descriptive names
run_id = f"gpt2_layer0_attn_{timestamp}"
sae_id = f"sae_gpt2_layer0_4x_k32"
```

### Store Organization

```
store/
├── activations/
│   └── <model>_<layer>_<date>/
├── runs/
│   └── <sae_training>_<date>/
└── sae_models/
    └── <sae_id>/
```

### Version Control

- Track configurations in code
- Save metadata with experiments
- Document hyperparameters

### Logging

```python
# Use wandb for tracking
config = SaeTrainingConfig(
    use_wandb=True,
    wandb_project="sae-experiments",
    wandb_run_name="gpt2-layer0-4x"
)
```

## Debugging Tips

### Verify Layer Names

```python
# Always check available layers
layers = lm.layers.list_layers()
print(f"Available layers: {layers}")

# Use exact names
layer_name = layers[0]  # Don't guess!
```

### Check Activations

```python
# Verify activations were saved
from pathlib import Path
run_path = Path(f"store/activations/{run_id}")
assert run_path.exists(), f"Run {run_id} not found"

# Check metadata
import json
with open(run_path / "meta.json") as f:
    meta = json.load(f)
    print(f"Samples: {meta['sample_count']}")
```

### Validate SAE Training

```python
# Check training metrics
print(f"Final loss: {history['loss'][-1]}")
print(f"Final R²: {history['r2'][-1]}")
print(f"Dead features: {history['dead_features'][-1]}")

# Verify weights learned
weight_var = sae.encoder.weight.var().item()
assert weight_var > 0.01, "Weights may not have learned!"
```

### Test Hooks

```python
# Verify hook is registered
hook_id = lm.layers.register_hook("layer_0", detector)
assert hook_id in lm.layers.context._hook_id_map

# Check hook executes
outputs, encodings = lm.inference.execute_inference(["test"])
activations = detector.get_captured()
assert activations is not None, "Hook didn't execute!"
```

## Common Pitfalls

### Forgetting to Cleanup

```python
# ❌ Wrong - hook never unregistered
hook_id = lm.layers.register_hook("layer_0", detector)
# ... use hook ...
# Forgot to unregister!

# ✅ Correct - always cleanup
try:
    hook_id = lm.layers.register_hook("layer_0", detector)
    # ... use hook ...
finally:
    lm.layers.unregister_hook(hook_id)
```

### Wrong Layer Names

```python
# ❌ Wrong - guessing layer name
hook_id = lm.layers.register_hook("layer_0", detector)  # May not exist!

# ✅ Correct - check first
layers = lm.layers.list_layers()
layer_name = layers[0]  # Use actual name
hook_id = lm.layers.register_hook(layer_name, detector)
```

### Memory Leaks

```python
# ❌ Wrong - accumulating without clearing
def process_activations(self, module, input, output):
    self.all_activations.append(output)  # Never cleared!

# ✅ Correct - clear periodically
def process_activations(self, module, input, output):
    if len(self.all_activations) > 100:
        self.all_activations.clear()
    self.all_activations.append(output.detach().cpu())
```

### In-place Modification

```python
# ❌ Wrong - modifies in place
def modify_fn(x):
    x *= 2.0  # In-place modification
    return x

# ✅ Correct - return new tensor
def modify_fn(x):
    return x * 2.0  # Creates new tensor
```

## Code Organization

### Reusable Functions

```python
def create_sae(n_latents, n_inputs, k, device):
    """Create and return SAE."""
    return TopKSae(n_latents=n_latents, n_inputs=n_inputs, k=k, device=device)

def train_sae(sae, store, run_id, layer_signature, config):
    """Train SAE and return history."""
    trainer = SaeTrainer(sae)
    return trainer.train(store, run_id, layer_signature, config)
```

### Configuration Management

```python
# Use dataclasses or config files
@dataclass
class ExperimentConfig:
    model_name: str
    layer_name: str
    n_latents: int
    k: int
    epochs: int
    batch_size: int

config = ExperimentConfig(
    model_name="gpt2",
    layer_name="transformer.h.0.attn.c_attn",
    n_latents=4096,
    k=32,
    epochs=100,
    batch_size=256
)
```

### Error Handling

```python
try:
    run_id = lm.activations.save(
        layer_signature=layer_name,
        dataset=dataset,
        sample_limit=1000
    )
except ValueError as e:
    print(f"Layer not found: {e}")
    # Handle error
except RuntimeError as e:
    print(f"Memory error: {e}")
    # Reduce batch size and retry
```

## Documentation

### Document Experiments

```python
# Save experiment metadata
experiment_meta = {
    "model": "gpt2",
    "layer": "transformer.h.0.attn.c_attn",
    "sae_config": {
        "n_latents": 4096,
        "n_inputs": 768,
        "k": 32
    },
    "training_config": {
        "epochs": 100,
        "batch_size": 256,
        "lr": 1e-3
    },
    "results": {
        "final_loss": history['loss'][-1],
        "final_r2": history['r2'][-1]
    }
}

import json
with open("experiment_meta.json", "w") as f:
    json.dump(experiment_meta, f, indent=2)
```

### Comment Code

```python
# Save activations from attention layer
# Using batch size 4 to fit in GPU memory
run_id = lm.activations.save(
    layer_signature="transformer.h.0.attn.c_attn",
    dataset=dataset,
    batch_size=4,  # Limited by GPU memory
    sample_limit=1000
)
```

## Testing

### Start with Small Examples

```python
# Test with minimal data first
test_texts = ["Hello, world!"] * 10
test_dataset = TextDataset(texts=test_texts)

# Verify workflow works
run_id = lm.activations.save(
    layer_signature="layer_0",
    dataset=test_dataset,
    sample_limit=10
)
```

### Validate Each Step

```python
# Verify activations saved
assert Path(f"store/activations/{run_id}").exists()

# Verify SAE trains
history = trainer.train(...)
assert history['loss'][-1] < history['loss'][0]  # Loss decreased

# Verify concepts discovered
top_texts = sae.concepts.get_top_texts()
assert len(top_texts) > 0  # Found some concepts
```

## Next Steps

- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Examples](examples.md)** - Example code patterns
- **[Hooks: Advanced](hooks/advanced.md)** - Advanced patterns

