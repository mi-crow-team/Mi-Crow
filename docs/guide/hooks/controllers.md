# Using Controller Hooks

Controller hooks modify activations during inference to change model behavior. This guide covers built-in controllers, creating custom controllers, and intervention patterns.

## What are Controllers?

Controllers are hooks that:
- **Modify** activations during the forward pass
- **Return** modified values that replace originals
- **Change** model behavior in real-time
- **Enable** intervention experiments

Controllers are used for:
- Concept manipulation through SAE neurons
- Activation scaling and masking
- Intervention studies
- Model steering

## Built-in Controller Implementations

### FunctionController

Applies a user-provided function to activations.

```python
from mi_crow.hooks import FunctionController
import torch

# Scale activations by a factor
controller = FunctionController(
    layer_signature="transformer.h.0.attn.c_attn",
    function=lambda x: x * 2.0  # Double all activations
)

# Register and use
hook_id = lm.layers.register_hook("transformer.h.0.attn.c_attn", controller)
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])
```

**Function Requirements**:
- Takes a `torch.Tensor` as input
- Returns a `torch.Tensor` of the same shape
- Must be deterministic (no random operations)

**Common Functions**:

```python
# Scale by constant
scale_controller = FunctionController(
    layer_signature="layer_0",
    function=lambda x: x * 1.5
)

# Clamp values
clamp_controller = FunctionController(
    layer_signature="layer_0",
    function=lambda x: torch.clamp(x, min=-1.0, max=1.0)
)

# Add noise (for experimentation)
noise_controller = FunctionController(
    layer_signature="layer_0",
    function=lambda x: x + torch.randn_like(x) * 0.1
)
```

## Creating Custom Controllers

To create a custom controller, inherit from `Controller` and implement `modify_activations`:

### Simple Scaling Controller

```python
from mi_crow.hooks import Controller
from mi_crow.hooks.hook import HookType
import torch

class ScalingController(Controller):
    """Controller that scales activations by a factor."""
    
    def __init__(self, layer_signature: str | int, scale_factor: float):
        super().__init__(
            hook_type=HookType.FORWARD,
            layer_signature=layer_signature
        )
        self.scale_factor = scale_factor
    
    def modify_activations(self, module, inputs, output):
        """Scale the output activations."""
        if output is None:
            return output
        
        if isinstance(output, torch.Tensor):
            return output * self.scale_factor
        elif isinstance(output, (tuple, list)):
            # Handle tuple outputs (e.g., (hidden_states, attention_weights))
            return tuple(x * self.scale_factor if isinstance(x, torch.Tensor) else x 
                        for x in output)
        
        return output
```

### Selective Activation Controller

```python
class SelectiveController(Controller):
    """Controller that modifies only specific neurons."""
    
    def __init__(self, layer_signature: str | int, neuron_indices: list[int], scale: float):
        super().__init__(hook_type=HookType.FORWARD, layer_signature=layer_signature)
        self.neuron_indices = set(neuron_indices)
        self.scale = scale
    
    def modify_activations(self, module, inputs, output):
        """Scale only specified neurons."""
        if output is None or not isinstance(output, torch.Tensor):
            return output
        
        # Create modified output
        modified = output.clone()
        
        # Scale only specified neurons (assuming last dimension is features)
        for idx in self.neuron_indices:
            if idx < modified.shape[-1]:
                modified[..., idx] *= self.scale
        
        return modified
```

### Conditional Controller

```python
class ConditionalController(Controller):
    """Controller that applies modifications conditionally."""
    
    def __init__(self, layer_signature: str | int, condition_fn, modification_fn):
        super().__init__(hook_type=HookType.FORWARD, layer_signature=layer_signature)
        self.condition_fn = condition_fn
        self.modification_fn = modification_fn
    
    def modify_activations(self, module, inputs, output):
        """Apply modification if condition is met."""
        if output is None:
            return output
        
        # Check condition (e.g., based on activation statistics)
        if self.condition_fn(output):
            return self.modification_fn(output)
        
        return output
```

## Modifying Inputs vs Outputs

### FORWARD Hooks (Modify Outputs)

Most common - modify layer outputs:

```python
class OutputController(Controller):
    def modify_activations(self, module, inputs, output):
        # output is the layer's output
        # Modify and return
        return modified_output
```

**When to use**:
- Modifying activations after processing
- SAE-based interventions
- Concept manipulation
- Most intervention experiments

### PRE_FORWARD Hooks (Modify Inputs)

Modify inputs before layer processes them:

```python
class InputController(Controller):
    def __init__(self, layer_signature: str | int):
        super().__init__(
            hook_type=HookType.PRE_FORWARD,
            layer_signature=layer_signature
        )
    
    def modify_activations(self, module, inputs, output):
        # inputs is a tuple of input tensors
        # Modify and return new input tuple
        modified_inputs = tuple(x * 2.0 if isinstance(x, torch.Tensor) else x 
                               for x in inputs)
        return modified_inputs
```

**When to use**:
- Early intervention in forward pass
- Input preprocessing
- Modifying residual connections

## Activation Scaling, Masking, and Transformation

### Scaling Patterns

```python
# Uniform scaling
uniform_scale = FunctionController(
    layer_signature="layer_0",
    function=lambda x: x * 1.5
)

# Per-neuron scaling (requires custom controller)
# See SelectiveController example above

# Adaptive scaling based on magnitude
adaptive_scale = FunctionController(
    layer_signature="layer_0",
    function=lambda x: x * (1.0 + 0.1 * torch.sigmoid(x.abs().mean()))
)
```

### Masking Patterns

```python
# Zero out small activations
masking_controller = FunctionController(
    layer_signature="layer_0",
    function=lambda x: x * (x.abs() > 0.1).float()
)

# Top-K masking
topk_masking = FunctionController(
    layer_signature="layer_0",
    function=lambda x: torch.where(
        x.abs() >= torch.topk(x.abs(), k=10, dim=-1)[0][..., -1:],
        x, torch.zeros_like(x)
    )
)
```

### Transformation Patterns

```python
# Normalization
normalize_controller = FunctionController(
    layer_signature="layer_0",
    function=lambda x: (x - x.mean()) / (x.std() + 1e-8)
)

# Clipping
clip_controller = FunctionController(
    layer_signature="layer_0",
    function=lambda x: torch.clamp(x, min=-2.0, max=2.0)
)

# Non-linear transformation
tanh_controller = FunctionController(
    layer_signature="layer_0",
    function=lambda x: torch.tanh(x * 2.0)
)
```

## Use Cases

### Concept Manipulation

```python
# Amplify a specific SAE concept (neuron)
def amplify_concept(neuron_idx: int, scale: float):
    def modify_fn(x):
        modified = x.clone()
        modified[..., neuron_idx] *= scale
        return modified
    
    return FunctionController(
        layer_signature="layer_0",
        function=modify_fn
    )

controller = amplify_concept(neuron_idx=42, scale=2.0)
hook_id = lm.layers.register_hook("layer_0", controller)
```

### Intervention Experiments

```python
# A/B testing: with and without intervention
baseline_outputs, _ = lm.inference.execute_inference(["prompt"], with_controllers=False)

# Apply intervention
controller = FunctionController("layer_0", lambda x: x * 1.5)
hook_id = lm.layers.register_hook("layer_0", controller)
intervention_outputs, _ = lm.inference.execute_inference(["prompt"], with_controllers=True)

# Compare
difference = intervention_outputs.logits - baseline_outputs.logits
```

### Model Steering

```python
# Steer model toward certain behaviors
def steer_toward_concept(concept_vector: torch.Tensor, strength: float):
    def modify_fn(x):
        # Add concept vector scaled by strength
        return x + strength * concept_vector
    
    return FunctionController("layer_0", modify_fn)

# Use learned concept vector
concept_vec = torch.randn(768)  # Example concept vector
controller = steer_toward_concept(concept_vec, strength=0.1)
hook_id = lm.layers.register_hook("layer_0", controller)
```

## Best Practices

1. **Preserve shapes**: Return tensors with the same shape as input
2. **Handle None**: Check for None before modifying
3. **Clone when needed**: Use `.clone()` to avoid in-place modifications
4. **Test functions**: Verify your function works on sample tensors
5. **Document effects**: Clearly document what your controller does
6. **Clean up**: Always unregister controllers when done

## Common Pitfalls

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

### Shape Mismatches

```python
# ❌ Wrong - may cause shape errors
def modify_fn(x):
    return x.reshape(-1)  # Changes shape

# ✅ Correct - preserve shape
def modify_fn(x):
    return x * 2.0  # Same shape
```

### Non-deterministic Functions

```python
# ❌ Wrong - random operations
def modify_fn(x):
    return x + torch.randn_like(x)  # Non-deterministic

# ✅ Correct - deterministic (or document randomness)
def modify_fn(x):
    return x * 2.0  # Deterministic
```

## Integration with SAEs

SAEs work as controllers when attached to models:

```python
# SAE automatically works as a controller
lm.attach_sae(sae, layer_signature="layer_0")

# SAE decodes activations and can modify them
# Concept manipulation uses SAE as controller
sae.concepts.manipulate_concept(neuron_idx=42, scale=1.5)
```

## Next Steps

- **[Hook Registration](registration.md)** - Managing controllers on layers
- **[Advanced Patterns](advanced.md)** - Complex controller patterns
- **[Concept Manipulation](../workflows/concept-manipulation.md)** - Using controllers for concepts
- **[Activation Control](../workflows/activation-control.md)** - Workflow guide

