# Using Detector Hooks

Detector hooks are used to observe and collect data from model activations without modifying them. This guide covers built-in detectors, creating custom detectors, and common use cases.

## What are Detectors?

Detectors are hooks that:
- **Observe** activations during inference
- **Collect** data (tensors, metadata, statistics)
- **Never modify** activations - they're purely observational
- **Can save** data to the Store for persistence

Detectors are perfect for:
- Saving activations for analysis
- Tracking statistics across batches
- Collecting examples for concept discovery
- Debugging model behavior

## Built-in Detector Implementations

### LayerActivationDetector

Captures activations from a specific layer.

```python
from mi_crow.hooks import LayerActivationDetector

# Create detector
detector = LayerActivationDetector(
    layer_signature="transformer.h.0.attn.c_attn"
)

# Register on model
hook_id = lm.layers.register_hook("transformer.h.0.attn.c_attn", detector)

# Run inference
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])

# Access captured activations
activations = detector.get_captured()
print(f"Activations shape: {activations.shape}")

# Clear for next batch
detector.clear_captured()
```

**Key Methods**:
- `get_captured()`: Get the current batch's activations
- `clear_captured()`: Clear stored activations
- `tensor_metadata['activations']`: Direct access to tensor
- `metadata['activations_shape']`: Shape information

**Use Cases**:
- Saving activations for SAE training
- Analyzing activation patterns
- Debugging layer outputs

### ModelInputDetector

Captures model inputs (tokenized text).

```python
from mi_crow.hooks import ModelInputDetector

# Create detector
detector = ModelInputDetector()

# Register on model (attaches to input layer)
hook_id = lm.layers.register_hook("input", detector)

# Run inference
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])

# Access inputs
inputs = detector.tensor_metadata.get("inputs")
```

**Use Cases**:
- Tracking input tokens
- Saving input-output pairs
- Attention mask handling

### ModelOutputDetector

Captures final model outputs.

```python
from mi_crow.hooks import ModelOutputDetector

# Create detector
detector = ModelOutputDetector()

# Register on output layer
hook_id = lm.layers.register_hook("output", detector)

# Run inference
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])

# Access outputs
model_outputs = detector.tensor_metadata.get("outputs")
```

**Use Cases**:
- Saving model predictions
- Analyzing output distributions
- Collecting generation results

## Creating Custom Detectors

To create a custom detector, inherit from `Detector` and implement `process_activations`:

```python
from mi_crow.hooks import Detector
from mi_crow.hooks.hook import HookType
import torch

class StatisticsDetector(Detector):
    """Detector that tracks activation statistics."""
    
    def __init__(self, layer_signature: str | int):
        super().__init__(
            hook_type=HookType.FORWARD,
            layer_signature=layer_signature
        )
        # Initialize statistics
        self.metadata['mean'] = 0.0
        self.metadata['std'] = 0.0
        self.metadata['count'] = 0
    
    def process_activations(self, module, input, output):
        """Process and accumulate statistics."""
        # Extract tensor from output
        tensor = output
        if isinstance(output, (tuple, list)):
            tensor = output[0]
        
        if tensor is not None and isinstance(tensor, torch.Tensor):
            # Update running statistics
            batch_mean = tensor.mean().item()
            batch_std = tensor.std().item()
            count = self.metadata['count']
            
            # Running average
            total = count + 1
            self.metadata['mean'] = (
                (self.metadata['mean'] * count + batch_mean) / total
            )
            self.metadata['std'] = (
                (self.metadata['std'] * count + batch_std) / total
            )
            self.metadata['count'] = total
```

**Key Points**:
- Inherit from `Detector`
- Implement `process_activations(module, input, output)`
- Use `self.metadata` for scalar data
- Use `self.tensor_metadata` for tensors
- Don't return anything (detectors don't modify)

## Accumulating Metadata Across Batches

Detectors can accumulate data across multiple forward passes:

```python
class BatchAccumulator(Detector):
    """Accumulates activations across batches."""
    
    def __init__(self, layer_signature: str | int):
        super().__init__(hook_type=HookType.FORWARD, layer_signature=layer_signature)
        self.all_activations = []
    
    def process_activations(self, module, input, output):
        tensor = output
        if isinstance(output, (tuple, list)):
            tensor = output[0]
        
        if tensor is not None:
            # Store each batch (move to CPU to save memory)
            self.all_activations.append(tensor.detach().cpu())
    
    def get_all_activations(self):
        """Get all accumulated activations."""
        if self.all_activations:
            return torch.cat(self.all_activations, dim=0)
        return None
```

**Memory Considerations**:
- Move tensors to CPU: `tensor.detach().cpu()`
- Consider batching large accumulations
- Clear old data periodically

## Saving Detector Data to Store

Detectors can save data to the Store:

```python
from mi_crow.store import LocalStore
from mi_crow.hooks import LayerActivationDetector

store = LocalStore(base_path="./store")

# Create detector with store
detector = LayerActivationDetector(
    layer_signature="transformer.h.0.attn.c_attn"
)
detector.store = store  # Attach store

# Register and use
hook_id = lm.layers.register_hook("transformer.h.0.attn.c_attn", detector)

# After inference, save data
activations = detector.get_captured()
if activations is not None:
    # Save to store (example - actual API may vary)
    store.save_tensor("activations", activations)
```

## Use Cases

### Activation Analysis

```python
detector = LayerActivationDetector("transformer.h.0.attn.c_attn")
hook_id = lm.layers.register_hook("transformer.h.0.attn.c_attn", detector)

# Analyze multiple examples
for text in dataset:
    lm.inference.execute_inference([text])
    activations = detector.get_captured()
    print(f"Mean activation: {activations.mean().item()}")
    detector.clear_captured()
```

### Debugging Model Behavior

```python
# Track activations at multiple layers
detectors = {}
for layer_name in ["layer_0", "layer_5", "layer_10"]:
    det = LayerActivationDetector(layer_name)
    detectors[layer_name] = det
    lm.layers.register_hook(layer_name, det)

# Run inference and inspect
outputs, encodings = lm.inference.execute_inference(["Debug this"])

for name, det in detectors.items():
    acts = det.get_captured()
    print(f"{name}: {acts.shape}, mean={acts.mean().item()}")
```

### Data Collection for Training

```python
# Collect activations for SAE training
detector = LayerActivationDetector("transformer.h.0.attn.c_attn")
hook_id = lm.layers.register_hook("transformer.h.0.attn.c_attn", detector)

all_activations = []
for batch in dataset:
    lm.inference.execute_inference(batch)
    acts = detector.get_captured()
    all_activations.append(acts.detach().cpu())
    detector.clear_captured()

# Concatenate for training
training_data = torch.cat(all_activations, dim=0)
```

## Best Practices

1. **Clear between batches**: Use `clear_captured()` to avoid memory leaks
2. **Move to CPU**: Detach and move to CPU for large accumulations
3. **Use metadata**: Store scalar statistics in `metadata`, not `tensor_metadata`
4. **Handle None**: Check for None before accessing tensors
5. **Error handling**: Wrap operations in try/except blocks

## Integration with Other Features

Detectors integrate with:

- **Activation Saving**: `lm.activations.save()` uses detectors internally
- **SAE Training**: Collect activations for training data
- **Concept Discovery**: Track top activating texts
- **Store**: Save detector data persistently

## Next Steps

- **[Using Controllers](controllers.md)** - Learn about modification hooks
- **[Hook Registration](registration.md)** - Managing detectors on layers
- **[Advanced Patterns](advanced.md)** - Complex detector patterns
- **[Saving Activations](../workflows/saving-activations.md)** - Workflow guide

