# Activation Control with Hooks

This guide covers directly manipulating activations using hooks for fine-grained model control.

## Overview

Activation control provides:
- Direct manipulation of layer activations
- Fine-grained control without SAEs
- Custom intervention patterns
- Multi-layer coordination

## When to Use Activation Control

Use activation control when:
- You need direct control over activations
- SAE-based manipulation is insufficient
- You want custom intervention patterns
- You're experimenting with new control methods

## Basic Activation Control

### Using Detector Hooks for Inspection

First, inspect activations to understand what you're working with:

```python
from mi_crow.hooks import LayerActivationDetector

# Create detector
detector = LayerActivationDetector("transformer.h.0.attn.c_attn")

# Register hook
hook_id = lm.layers.register_hook("transformer.h.0.attn.c_attn", detector)

# Run inference
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])

# Inspect activations
activations = detector.get_captured()
print(f"Activations shape: {activations.shape}")
print(f"Mean: {activations.mean().item()}")
print(f"Std: {activations.std().item()}")

# Cleanup
lm.layers.unregister_hook(hook_id)
```

### Using Controller Hooks for Modification

Modify activations directly:

```python
from mi_crow.hooks import FunctionController

# Create controller that scales activations
controller = FunctionController(
    layer_signature="transformer.h.0.attn.c_attn",
    function=lambda x: x * 1.5  # Scale by 1.5
)

# Register hook
hook_id = lm.layers.register_hook("transformer.h.0.attn.c_attn", controller)

# Run inference with modification
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])

# Cleanup
lm.layers.unregister_hook(hook_id)
```

## Custom Controller Implementation

Create custom controllers for specific needs:

### Scaling Controller

```python
from mi_crow.hooks import Controller
from mi_crow.hooks.hook import HookType
import torch

class ScalingController(Controller):
    """Scales activations by a factor."""
    
    def __init__(self, layer_signature: str | int, scale_factor: float):
        super().__init__(hook_type=HookType.FORWARD, layer_signature=layer_signature)
        self.scale_factor = scale_factor
    
    def modify_activations(self, module, inputs, output):
        if output is None:
            return output
        return output * self.scale_factor

# Use
controller = ScalingController("transformer.h.0.attn.c_attn", scale_factor=1.5)
hook_id = lm.layers.register_hook("transformer.h.0.attn.c_attn", controller)
```

### Selective Neuron Controller

```python
class SelectiveController(Controller):
    """Modifies only specific neurons."""
    
    def __init__(self, layer_signature: str | int, neuron_indices: list[int], scale: float):
        super().__init__(hook_type=HookType.FORWARD, layer_signature=layer_signature)
        self.neuron_indices = set(neuron_indices)
        self.scale = scale
    
    def modify_activations(self, module, inputs, output):
        if output is None or not isinstance(output, torch.Tensor):
            return output
        
        modified = output.clone()
        for idx in self.neuron_indices:
            if idx < modified.shape[-1]:
                modified[..., idx] *= self.scale
        return modified

# Use
controller = SelectiveController(
    "transformer.h.0.attn.c_attn",
    neuron_indices=[42, 100, 200],
    scale=2.0
)
hook_id = lm.layers.register_hook("transformer.h.0.attn.c_attn", controller)
```

## Multi-Layer Interventions

Coordinate interventions across multiple layers:

### Sequential Modifications

```python
# Modify early layer
early_controller = FunctionController("transformer.h.0.attn.c_attn", lambda x: x * 1.2)
hook1 = lm.layers.register_hook("transformer.h.0.attn.c_attn", early_controller)

# Modify late layer
late_controller = FunctionController("transformer.h.10.attn.c_attn", lambda x: x * 0.8)
hook2 = lm.layers.register_hook("transformer.h.10.attn.c_attn", late_controller)

# Both apply during forward pass
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])

# Cleanup
lm.layers.unregister_hook(hook1)
lm.layers.unregister_hook(hook2)
```

### Cross-Layer Communication

```python
class CoordinatedController(Controller):
    """Uses information from another layer."""
    
    def __init__(self, layer_signature, reference_detector):
        super().__init__(hook_type=HookType.FORWARD, layer_signature=layer_signature)
        self.reference_detector = reference_detector
    
    def modify_activations(self, module, inputs, output):
        # Get activations from reference layer
        ref_activations = self.reference_detector.get_captured()
        
        if ref_activations is not None and output is not None:
            # Scale based on reference layer
            scale = 1.0 + 0.1 * ref_activations.mean().item()
            return output * scale
        
        return output

# Setup
ref_detector = LayerActivationDetector("transformer.h.0.attn.c_attn")
lm.layers.register_hook("transformer.h.0.attn.c_attn", ref_detector)

coordinated = CoordinatedController("transformer.h.5.attn.c_attn", ref_detector)
hook_id = lm.layers.register_hook("transformer.h.5.attn.c_attn", coordinated)
```

## A/B Testing with Hooks

Compare behavior with and without interventions:

### Baseline

```python
# Get baseline
baseline_outputs, _ = lm.inference.execute_inference(
    ["Your prompt"],
    with_controllers=False  # Disable all controllers
)
```

### With Intervention

```python
# Apply intervention
controller = FunctionController("layer_0", lambda x: x * 1.5)
hook_id = lm.layers.register_hook("layer_0", controller)

# Get modified output
intervention_outputs, _ = lm.inference.execute_inference(
    ["Your prompt"],
    with_controllers=True  # Enable controllers
)

# Compare
difference = intervention_outputs.logits - baseline_outputs.logits

# Cleanup
lm.layers.unregister_hook(hook_id)
```

## Advanced Patterns

### Conditional Control

```python
class ConditionalController(Controller):
    """Applies modification conditionally."""
    
    def __init__(self, layer_signature, condition_fn, modification_fn):
        super().__init__(hook_type=HookType.FORWARD, layer_signature=layer_signature)
        self.condition_fn = condition_fn
        self.modification_fn = modification_fn
    
    def modify_activations(self, module, inputs, output):
        if output is None:
            return output
        
        if self.condition_fn(output):
            return self.modification_fn(output)
        
        return output

# Use: only modify if activation magnitude is high
controller = ConditionalController(
    "layer_0",
    condition_fn=lambda x: x.abs().mean() > 1.0,
    modification_fn=lambda x: x * 1.5
)
```

### Pipeline of Transformations

```python
class PipelineController(Controller):
    """Applies multiple transformations."""
    
    def __init__(self, layer_signature, transformations):
        super().__init__(hook_type=HookType.FORWARD, layer_signature=layer_signature)
        self.transformations = transformations
    
    def modify_activations(self, module, inputs, output):
        result = output
        for transform in self.transformations:
            result = transform(result)
        return result

# Use
pipeline = PipelineController(
    "layer_0",
    transformations=[
        lambda x: x * 1.2,                    # Scale
        lambda x: torch.clamp(x, -2, 2),      # Clamp
        lambda x: (x - x.mean()) / (x.std() + 1e-8)  # Normalize
    ]
)
```

## Best Practices

1. **Always cleanup**: Unregister hooks when done
2. **Use context managers**: For automatic cleanup
3. **Test incrementally**: Start with simple modifications
4. **Monitor effects**: Compare before/after
5. **Document interventions**: Record what each does

## Common Patterns

### Context Manager Pattern

```python
class HookContext:
    """Context manager for hook lifecycle."""
    
    def __init__(self, layers, hook, layer_signature):
        self.layers = layers
        self.hook = hook
        self.layer_signature = layer_signature
        self.hook_id = None
    
    def __enter__(self):
        self.hook_id = self.layers.register_hook(self.layer_signature, self.hook)
        return self.hook
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_id:
            self.layers.unregister_hook(self.hook_id)

# Usage
with HookContext(lm.layers, controller, "layer_0") as hook:
    outputs, encodings = lm.inference.execute_inference(["Hello, world!"])
# Hook automatically unregistered
```

### Try/Finally Pattern

```python
hook_id = None
try:
    controller = FunctionController("layer_0", lambda x: x * 1.5)
    hook_id = lm.layers.register_hook("layer_0", controller)
    outputs, encodings = lm.inference.execute_inference(["Hello, world!"])
finally:
    if hook_id:
        lm.layers.unregister_hook(hook_id)
```

## Next Steps

After learning activation control:

- **[Hooks: Controllers](../hooks/controllers.md)** - Detailed controller guide
- **[Hooks: Advanced](../hooks/advanced.md)** - Advanced hook patterns
- **[Concept Manipulation](concept-manipulation.md)** - SAE-based control
- **[Examples](../examples.md)** - See example notebooks

## Related Examples

- `examples/08_inference_with_hooks.ipynb` - Complete hooks example
- `examples/03_load_concepts.ipynb` - Concept manipulation with hooks

