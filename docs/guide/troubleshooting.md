# Troubleshooting

This guide covers common issues and their solutions when using mi-crow.

## Common Errors and Solutions

### Layer Signature Not Found

**Error**: `ValueError: Layer signature not found`

**Causes**:
- Incorrect layer name
- Layer doesn't exist in model
- Typo in layer name

**Solutions**:

```python
# 1. List available layers
layers = lm.layers.list_layers()
print("Available layers:", layers)

# 2. Use exact name from list
layer_name = layers[0]  # Don't guess!

# 3. Verify layer exists before using
if layer_name in layers:
    hook_id = lm.layers.register_hook(layer_name, detector)
else:
    print(f"Layer {layer_name} not found!")
```

### Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Causes**:
- Batch size too large
- Model too large for GPU
- Accumulating tensors in memory

**Solutions**:

```python
# 1. Reduce batch size
run_id = lm.activations.save(
    layer_signature="layer_0",
    dataset=dataset,
    batch_size=1,  # Minimal batch size
    sample_limit=100
)

# 2. Use CPU instead of GPU
sae = TopKSae(n_latents=4096, n_inputs=768, k=32, device="cpu")

# 3. Clear detector data periodically
detector.clear_captured()

# 4. Move tensors to CPU
activations = detector.get_captured()
activations_cpu = activations.detach().cpu()
```

### Hook Not Executing

**Symptoms**: Hook doesn't seem to run, no data collected

**Causes**:
- Hook not registered
- Hook disabled
- Wrong layer signature
- Hook unregistered too early

**Solutions**:

```python
# 1. Verify hook is registered
hook_id = lm.layers.register_hook("layer_0", detector)
assert hook_id in lm.layers.context._hook_id_map

# 2. Check hook is enabled
assert detector.is_enabled(), "Hook is disabled!"

# 3. Verify layer name is correct
layers = lm.layers.list_layers()
assert "layer_0" in layers, "Layer doesn't exist!"

# 4. Ensure hook stays registered during inference
outputs, encodings = lm.inference.execute_inference(["test"])
activations = detector.get_captured()  # Check before unregistering
```

### SAE Training Instability

**Symptoms**: Loss doesn't decrease, weights not learning

**Causes**:
- Learning rate too high
- Too much regularization
- Not enough training data
- Dead features

**Solutions**:

```python
# 1. Reduce learning rate
config = SaeTrainingConfig(
    epochs=100,
    batch_size=256,
    lr=1e-4,  # Lower learning rate
    l1_lambda=1e-5  # Lower regularization
)

# 2. Check for dead features
dead_count = history['dead_features'][-1]
if dead_count > sae.n_latents * 0.1:  # More than 10% dead
    # Reduce sparsity
    sae = TopKSae(n_latents=4096, n_inputs=768, k=64, device="cuda")  # Increase k

# 3. Verify weights are learning
weight_var = sae.encoder.weight.var().item()
if weight_var < 0.01:
    print("Warning: Weights may not be learning!")
    # Try different learning rate or more epochs
```

### Poor SAE Reconstruction

**Symptoms**: Low R² score, high reconstruction error

**Causes**:
- Model capacity too small
- Not enough training
- Wrong hyperparameters

**Solutions**:

```python
# 1. Increase model capacity
sae = TopKSae(
    n_latents=8192,  # More neurons
    n_inputs=768,
    k=64,            # More active neurons
    device="cuda"
)

# 2. Train longer
config = SaeTrainingConfig(
    epochs=200,  # More epochs
    batch_size=256,
    lr=1e-3
)

# 3. Adjust hyperparameters
config = SaeTrainingConfig(
    epochs=100,
    batch_size=256,
    lr=1e-3,
    l1_lambda=1e-5  # Less regularization
)
```

## Layer Signature Issues

### Finding Correct Layer Names

```python
# List all layers
all_layers = lm.layers.list_layers()

# Filter by pattern
attention_layers = [l for l in all_layers if "attn" in l]
mlp_layers = [l for l in all_layers if "mlp" in l]

# Print for inspection
for i, layer in enumerate(all_layers):
    print(f"{i}: {layer}")
```

### Layer Name Variations

Different models use different naming conventions:

```python
# GPT-2 style
"transformer.h.0.attn.c_attn"

# BERT style
"bert.encoder.layer.0.attention.self.query"

# Custom models
"model.layers.0.self_attn.q_proj"

# Always check your specific model
layers = lm.layers.list_layers()
```

## Memory Problems

### GPU Memory

```python
# Check GPU memory
import torch
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Clear cache if needed
torch.cuda.empty_cache()
```

### Memory Leaks

```python
# Check for unregistered hooks
hook_count = len(lm.layers.context._hook_id_map)
print(f"Registered hooks: {hook_count}")

# Unregister all if needed
for hook_id in list(lm.layers.context._hook_id_map.keys()):
    lm.layers.unregister_hook(hook_id)
```

### Accumulating Data

```python
# Clear detector data
detector.clear_captured()
detector.tensor_metadata.clear()

# Move to CPU
activations = detector.get_captured()
if activations is not None:
    activations_cpu = activations.detach().cpu()
    # Use CPU version, original will be garbage collected
```

## Training Instability

### Loss Not Decreasing

```python
# Check training history
print(f"Initial loss: {history['loss'][0]}")
print(f"Final loss: {history['loss'][-1]}")

if history['loss'][-1] >= history['loss'][0]:
    print("Loss didn't decrease!")
    # Try:
    # - Lower learning rate
    # - More epochs
    # - Different initialization
```

### Exploding Gradients

```python
# Reduce learning rate
config = SaeTrainingConfig(
    epochs=100,
    batch_size=256,
    lr=1e-4,  # Much lower learning rate
    l1_lambda=1e-5
)

# Or use gradient clipping (if supported)
```

### Dead Features

```python
# Check dead features
dead_ratio = history['dead_features'][-1] / sae.n_latents

if dead_ratio > 0.1:
    print(f"Too many dead features: {dead_ratio:.2%}")
    # Solutions:
    # - Increase k (sparsity)
    # - Reduce l1_lambda
    # - Increase learning rate slightly
```

## Device Compatibility

### CUDA Issues

```python
import torch
from mi_crow.language_model import LanguageModel
from mi_crow.store import LocalStore

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Always choose device based on availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

store = LocalStore(base_path="./store")

# This will raise a clear ValueError if you force device=\"cuda\" but CUDA is not available
lm = LanguageModel.from_huggingface(
    "sshleifer/tiny-gpt2",
    store=store,
    device=device,
)
```

### MPS (Apple Silicon)

```python
# Check MPS availability
if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon GPU")
else:
    device = "cpu"
    print("Using CPU")
```

### Device Mismatch

```python
# Ensure model and data on same device
model = model.to(device)
data = data.to(device)

# Check device
print(f"Model device: {next(model.parameters()).device}")
print(f"Data device: {data.device}")
```

## Import Errors

### Module Not Found

```python
# Verify installation
import mi_crow
print(mi_crow.__version__)

# Check imports
from mi_crow.language_model import LanguageModel
from mi_crow.hooks import Detector
```

### Version Mismatches

```python
# Check versions
import torch
import transformers
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")

# Update if needed
# pip install --upgrade torch transformers
```

## Getting Help

### Check Documentation

- User guide sections
- API reference
- Example notebooks
- Experiment walkthroughs

### Debug Information

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check system info
import sys
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
```

### Reproduce Issues

```python
# Create minimal reproduction
# 1. Start with simplest possible case
# 2. Add complexity until issue appears
# 3. Document exact steps

import torch
from mi_crow.language_model import LanguageModel
from mi_crow.store import LocalStore

store = LocalStore(base_path="./store")
device = "cuda" if torch.cuda.is_available() else "cpu"

lm = LanguageModel.from_huggingface("sshleifer/tiny-gpt2", store=store, device=device)
layers = lm.layers.list_layers()
print(layers)
```

## Next Steps

- **[Best Practices](best-practices.md)** - Prevent issues before they occur
- **[Examples](examples.md)** - See working code
- **[API Reference](../api/index.md)** - Detailed API documentation

