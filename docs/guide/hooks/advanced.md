# Advanced Hooks Patterns

This guide covers advanced hook patterns, performance considerations, and complex use cases.

## SAE as Both Detector and Controller

SAEs in mi-crow implement both `Detector` and `Controller` interfaces, allowing them to:

- **Detect**: Decode activations to sparse latents
- **Control**: Modify activations based on concept manipulation

### How SAEs Work as Hooks

```python
from mi_crow.mechanistic.sae import TopKSae

# Create SAE
sae = TopKSae(n_latents=512, n_inputs=768, k=8)

# Attach to model (registers as hook internally)
lm.attach_sae(sae, layer_signature="layer_0")

# SAE now works as both:
# 1. Detector: Decodes activations to latents
# 2. Controller: Can modify activations via concept manipulation
```

### Concept Manipulation Through SAE

```python
# Enable text tracking (detector functionality)
sae.concepts.enable_text_tracking(top_k=10)

# Run inference - SAE decodes activations
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])

# Manipulate concepts (controller functionality)
sae.concepts.manipulate_concept(neuron_idx=42, scale=1.5)

# Run again - activations are modified
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])
```

The SAE automatically handles the dual role - you don't need to manage it as separate hooks.

## Multi-Layer Hook Coordination

Coordinating hooks across multiple layers enables complex interventions:

### Sequential Layer Processing

```python
# Register hooks on multiple layers
detectors = {}
for i in [0, 5, 10]:
    layer_name = f"transformer.h.{i}.attn.c_attn"
    det = LayerActivationDetector(layer_name)
    detectors[i] = det
    lm.layers.register_hook(layer_name, det)

# Process activations sequentially
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])

# Analyze across layers
for i, det in detectors.items():
    acts = det.get_captured()
    print(f"Layer {i}: mean={acts.mean().item()}")
```

### Cascading Interventions

```python
# Modify early layer, then late layer
early_controller = FunctionController("layer_0", lambda x: x * 1.2)
late_controller = FunctionController("layer_10", lambda x: x * 0.8)

hook1 = lm.layers.register_hook("layer_0", early_controller)
hook2 = lm.layers.register_hook("layer_10", late_controller)

# Both modifications apply in sequence
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])

lm.layers.unregister_hook(hook1)
lm.layers.unregister_hook(hook2)
```

### Cross-Layer Communication

```python
class CoordinatedController(Controller):
    """Controller that uses information from other layers."""
    
    def __init__(self, layer_signature, reference_detector):
        super().__init__(hook_type=HookType.FORWARD, layer_signature=layer_signature)
        self.reference_detector = reference_detector
    
    def modify_activations(self, module, inputs, output):
        # Get activations from reference layer
        ref_activations = self.reference_detector.get_captured()
        
        if ref_activations is not None:
            # Modify based on reference layer
            scale = ref_activations.mean().item()
            return output * (1.0 + 0.1 * scale)
        
        return output

# Setup
ref_detector = LayerActivationDetector("layer_0")
lm.layers.register_hook("layer_0", ref_detector)

coordinated = CoordinatedController("layer_5", ref_detector)
lm.layers.register_hook("layer_5", coordinated)
```

## Conditional Hook Execution

Hooks can conditionally execute based on various criteria:

### Activation-Based Conditions

```python
class ConditionalController(Controller):
    """Only modifies when activation magnitude exceeds threshold."""
    
    def __init__(self, layer_signature, threshold=1.0):
        super().__init__(hook_type=HookType.FORWARD, layer_signature=layer_signature)
        self.threshold = threshold
    
    def modify_activations(self, module, inputs, output):
        if output is None:
            return output
        
        # Check condition
        if output.abs().mean().item() > self.threshold:
            # Only modify if condition met
            return output * 1.5
        
        return output
```

### Token-Based Conditions

```python
class TokenConditionalController(Controller):
    """Modifies only for specific token positions."""
    
    def __init__(self, layer_signature, token_indices):
        super().__init__(hook_type=HookType.FORWARD, layer_signature=layer_signature)
        self.token_indices = set(token_indices)
    
    def modify_activations(self, module, inputs, output):
        if output is None:
            return output
        
        modified = output.clone()
        # Modify only specified token positions
        for idx in self.token_indices:
            if idx < modified.shape[1]:  # seq_len dimension
                modified[:, idx, :] *= 2.0
        
        return modified
```

### Batch-Based Conditions

```python
class BatchConditionalController(Controller):
    """Modifies only for certain batches."""
    
    def __init__(self, layer_signature):
        super().__init__(hook_type=HookType.FORWARD, layer_signature=layer_signature)
        self.batch_count = 0
    
    def modify_activations(self, module, inputs, output):
        self.batch_count += 1
        
        # Modify only every 10th batch
        if self.batch_count % 10 == 0:
            return output * 1.5
        
        return output
```

## Hook Composition Patterns

### Pipeline of Transformations

```python
class PipelineController(Controller):
    """Applies multiple transformations in sequence."""
    
    def __init__(self, layer_signature, transformations):
        super().__init__(hook_type=HookType.FORWARD, layer_signature=layer_signature)
        self.transformations = transformations
    
    def modify_activations(self, module, inputs, output):
        result = output
        for transform in self.transformations:
            result = transform(result)
        return result

# Usage
pipeline = PipelineController(
    "layer_0",
    transformations=[
        lambda x: x * 1.2,           # Scale
        lambda x: torch.clamp(x, -2, 2),  # Clamp
        lambda x: (x - x.mean()) / (x.std() + 1e-8)  # Normalize
    ]
)
```

### Conditional Composition

```python
class ConditionalPipeline(Controller):
    """Applies different pipelines based on condition."""
    
    def __init__(self, layer_signature, condition_fn, pipeline_a, pipeline_b):
        super().__init__(hook_type=HookType.FORWARD, layer_signature=layer_signature)
        self.condition_fn = condition_fn
        self.pipeline_a = pipeline_a
        self.pipeline_b = pipeline_b
    
    def modify_activations(self, module, inputs, output):
        if self.condition_fn(output):
            return self.pipeline_a(output)
        else:
            return self.pipeline_b(output)
```

## Performance Considerations

### Hook Overhead

Hooks add overhead to forward passes. Minimize it:

```python
# ❌ Slow - creates new tensors every time
class SlowController(Controller):
    def modify_activations(self, module, inputs, output):
        return output.clone() * 2.0  # Unnecessary clone

# ✅ Fast - in-place when safe, or reuse operations
class FastController(Controller):
    def modify_activations(self, module, inputs, output):
        return output * 2.0  # No clone needed
```

### Batch Processing Optimization

```python
# Process multiple examples efficiently
hook_ids = []
try:
    # Register once
    for layer in layers:
        det = LayerActivationDetector(layer)
        hook_ids.append(lm.layers.register_hook(layer, det))
    
    # Process all batches
    for batch in dataset:
        outputs, encodings = lm.inference.execute_inference(batch)
        # Access data after batch
finally:
    # Cleanup once
    for hook_id in hook_ids:
        lm.layers.unregister_hook(hook_id)
```

### Memory Management

```python
class MemoryEfficientDetector(Detector):
    """Detector that clears old data automatically."""
    
    def __init__(self, layer_signature, max_batches=100):
        super().__init__(hook_type=HookType.FORWARD, layer_signature=layer_signature)
        self.max_batches = max_batches
        self.batch_count = 0
    
    def process_activations(self, module, input, output):
        self.batch_count += 1
        
        # Clear old data periodically
        if self.batch_count > self.max_batches:
            self.tensor_metadata.clear()
            self.batch_count = 0
        
        # Store current batch (moved to CPU)
        if output is not None:
            self.tensor_metadata['activations'] = output.detach().cpu()
```

## Memory Management with Hooks

### Moving to CPU

Always move large tensors to CPU in detectors:

```python
def process_activations(self, module, input, output):
    if output is not None:
        # Move to CPU to save GPU memory
        self.tensor_metadata['activations'] = output.detach().cpu()
```

### Clearing Old Data

```python
# Clear detector data periodically
if batch_count % 100 == 0:
    detector.clear_captured()
    detector.tensor_metadata.clear()
```

### Limiting Accumulation

```python
class LimitedAccumulator(Detector):
    """Only keeps last N batches."""
    
    def __init__(self, layer_signature, max_batches=10):
        super().__init__(hook_type=HookType.FORWARD, layer_signature=layer_signature)
        self.max_batches = max_batches
        self.batches = []
    
    def process_activations(self, module, input, output):
        self.batches.append(output.detach().cpu())
        if len(self.batches) > self.max_batches:
            self.batches.pop(0)  # Remove oldest
```

## Debugging Hook Execution

### Logging Hook Calls

```python
class LoggingController(Controller):
    """Logs all modifications for debugging."""
    
    def modify_activations(self, module, inputs, output):
        import logging
        logger = logging.getLogger(__name__)
        
        if output is not None:
            logger.debug(
                f"Hook {self.id}: shape={output.shape}, "
                f"mean={output.mean().item()}, std={output.std().item()}"
            )
        
        return output * 2.0
```

### Tracking Hook Statistics

```python
class StatisticsController(Controller):
    """Tracks statistics about modifications."""
    
    def __init__(self, layer_signature):
        super().__init__(hook_type=HookType.FORWARD, layer_signature=layer_signature)
        self.modification_count = 0
        self.total_scale = 0.0
    
    def modify_activations(self, module, inputs, output):
        self.modification_count += 1
        scale = 2.0
        self.total_scale += scale
        
        return output * scale
    
    def get_stats(self):
        return {
            'modifications': self.modification_count,
            'avg_scale': self.total_scale / self.modification_count if self.modification_count > 0 else 0
        }
```

## Common Pitfalls and Solutions

### Pitfall 1: In-place Modification

```python
# ❌ Wrong - modifies in place
def modify_fn(x):
    x *= 2.0
    return x

# ✅ Correct - returns new tensor
def modify_fn(x):
    return x * 2.0
```

### Pitfall 2: Not Handling None

```python
# ❌ Wrong - crashes if output is None
def modify_fn(x):
    return x * 2.0

# ✅ Correct - handles None
def modify_fn(x):
    if x is None:
        return None
    return x * 2.0
```

### Pitfall 3: Memory Leaks

```python
# ❌ Wrong - accumulates without clearing
def process_activations(self, module, input, output):
    self.all_activations.append(output)  # Never cleared!

# ✅ Correct - clear periodically
def process_activations(self, module, input, output):
    if len(self.all_activations) > 100:
        self.all_activations.clear()
    self.all_activations.append(output.detach().cpu())
```

### Pitfall 4: Not Unregistering

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

## Next Steps

- **[Hook Registration](registration.md)** - Managing complex hook setups
- **[Workflows](../workflows/activation-control.md)** - Real-world hook usage
- **[Best Practices](../best-practices.md)** - General best practices

