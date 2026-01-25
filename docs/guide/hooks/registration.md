# Hook Registration and Management

This guide covers registering hooks on layers, managing hook lifecycles, and best practices for hook management.

## Registering Hooks on Layers

### Basic Registration

The simplest way to register a hook:

```python
from mi_crow.hooks import LayerActivationDetector

# Create hook
detector = LayerActivationDetector(layer_signature="transformer.h.0.attn.c_attn")

# Register on model
hook_id = lm.layers.register_hook(
    layer_signature="transformer.h.0.attn.c_attn",
    hook=detector
)

# Hook is now active and will execute during forward passes
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])
```

The `register_hook` method:
- Returns the hook's unique ID
- Automatically sets the hook's layer signature
- Attaches the hook to the PyTorch layer
- Adds the hook to the registry

### Finding Layers

Before registering, you need to know the layer name:

```python
# List all available layers
layer_names = lm.layers.list_layers()
print(layer_names)

# Example output:
# ['transformer.wte', 'transformer.h.0.attn.c_attn', ...]
```

You can also use layer indices:

```python
# Register on first layer (index 0)
hook_id = lm.layers.register_hook(0, detector)
```

### Layer Signatures

Layer signatures can be:
- **String**: Exact layer name (e.g., `"transformer.h.0.attn.c_attn"`)
- **Integer**: Layer index (e.g., `0` for first layer)

The system automatically resolves layer names to actual PyTorch modules.

## Hook ID Management

### Automatic IDs

If you don't specify an ID, one is auto-generated:

```python
detector = LayerActivationDetector("layer_0")
hook_id = lm.layers.register_hook("layer_0", detector)
print(hook_id)  # e.g., "550e8400-e29b-41d4-a716-446655440000"
```

### Custom IDs

You can specify custom IDs for easier management:

```python
detector = LayerActivationDetector(
    layer_signature="layer_0",
    hook_id="my-detector-1"
)
hook_id = lm.layers.register_hook("layer_0", detector)
assert hook_id == "my-detector-1"
```

**Use cases for custom IDs**:
- Organizing hooks by experiment
- Easier debugging and logging
- Referencing hooks in configuration files

### ID Uniqueness

Hook IDs must be unique across all registered hooks:

```python
# ❌ This will raise ValueError
detector1 = LayerActivationDetector("layer_0", hook_id="same-id")
detector2 = LayerActivationDetector("layer_1", hook_id="same-id")  # Error!

# ✅ Use unique IDs
detector1 = LayerActivationDetector("layer_0", hook_id="detector-layer-0")
detector2 = LayerActivationDetector("layer_1", hook_id="detector-layer-1")
```

## Multiple Hooks on Same Layer

### Restrictions

**Important**: Only one hook class type (Detector or Controller) can be registered per layer.

```python
# ✅ This works - both are Detectors
detector1 = LayerActivationDetector("layer_0")
detector2 = ModelInputDetector()
# But wait - you can't register two Detectors on the same layer

# ❌ This raises ValueError - mixing Detector and Controller
detector = LayerActivationDetector("layer_0")
controller = FunctionController("layer_0", lambda x: x * 2.0)
lm.layers.register_hook("layer_0", detector)
lm.layers.register_hook("layer_0", controller)  # Error!
```

### Multiple Hooks on Different Layers

You can register hooks on multiple layers:

```python
# Register detectors on different layers
detector1 = LayerActivationDetector("layer_0")
detector2 = LayerActivationDetector("layer_5")
detector3 = LayerActivationDetector("layer_10")

hook1_id = lm.layers.register_hook("layer_0", detector1)
hook2_id = lm.layers.register_hook("layer_5", detector2)
hook3_id = lm.layers.register_hook("layer_10", detector3)

# All execute during forward pass
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])
```

## Unregistering Hooks

Always unregister hooks when done to prevent memory leaks:

### By Hook ID

```python
# Register
hook_id = lm.layers.register_hook("layer_0", detector)

# Use hook
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])

# Unregister
lm.layers.unregister_hook(hook_id)
```

### By Hook Instance

```python
# Register
hook_id = lm.layers.register_hook("layer_0", detector)

# Unregister using hook instance
lm.layers.unregister_hook(detector)  # Works too!
```

### Unregistering Multiple Hooks

```python
hook_ids = []

# Register multiple hooks
for layer_name in ["layer_0", "layer_5", "layer_10"]:
    det = LayerActivationDetector(layer_name)
    hook_id = lm.layers.register_hook(layer_name, det)
    hook_ids.append(hook_id)

# Unregister all
for hook_id in hook_ids:
    lm.layers.unregister_hook(hook_id)
```

### Safe Unregistering

Unregistering returns `True` if successful, `False` if hook not found:

```python
success = lm.layers.unregister_hook(hook_id)
if not success:
    print(f"Hook {hook_id} not found")
```

## Listing and Querying Registered Hooks

### Get All Hooks

```python
# Get all registered hooks
all_hooks = lm.layers.get_hooks()
print(f"Total hooks: {len(all_hooks)}")
```

### Get Hooks by Layer

```python
# Get hooks on a specific layer
layer_hooks = lm.layers.get_hooks(layer_signature="layer_0")
print(f"Hooks on layer_0: {len(layer_hooks)}")
```

### Get Hooks by Type

```python
from mi_crow.hooks import Detector, Controller

# Get only detectors
detectors = lm.layers.get_detectors()
print(f"Detectors: {len(detectors)}")

# Get only controllers
controllers = lm.layers.get_controllers()
print(f"Controllers: {len(controllers)}")
```

### Check if Hook is Registered

```python
# Check by ID
if hook_id in lm.layers.context._hook_id_map:
    print("Hook is registered")

# Or try to get it
hook_info = lm.layers.context._hook_id_map.get(hook_id)
if hook_info:
    layer, hook_type, hook = hook_info
    print(f"Hook on {layer}, type: {hook_type}")
```

## Hook Registry Inspection

The hook registry is accessible through the context:

```python
# Access registry directly
registry = lm.layers.context._hook_registry

# Structure: {layer_signature: {hook_type: [(hook, handle), ...]}}
for layer_name, hook_types in registry.items():
    print(f"Layer: {layer_name}")
    for hook_type, hooks in hook_types.items():
        print(f"  {hook_type}: {len(hooks)} hooks")
```

### ID Map

The ID map provides quick lookup:

```python
id_map = lm.layers.context._hook_id_map

# Structure: {hook_id: (layer_signature, hook_type, hook)}
for hook_id, (layer, hook_type, hook) in id_map.items():
    print(f"{hook_id}: {hook.__class__.__name__} on {layer}")
```

## Best Practices for Hook Lifecycle Management

### 1. Use Context Managers (Recommended Pattern)

```python
class HookContext:
    """Context manager for hook lifecycle."""
    
    def __init__(self, layers, hook, layer_signature):
        self.layers = layers
        self.hook = hook
        self.layer_signature = layer_signature
        self.hook_id = None
    
    def __enter__(self):
        self.hook_id = self.layers.register_hook(
            self.layer_signature, 
            self.hook
        )
        return self.hook
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_id:
            self.layers.unregister_hook(self.hook_id)

# Usage
with HookContext(lm.layers, detector, "layer_0") as hook:
    outputs, encodings = lm.inference.execute_inference(["Hello, world!"])
    activations = hook.get_captured()
# Hook automatically unregistered
```

### 2. Register in Try/Finally

```python
hook_id = None
try:
    detector = LayerActivationDetector("layer_0")
    hook_id = lm.layers.register_hook("layer_0", detector)
    outputs, encodings = lm.inference.execute_inference(["Hello, world!"])
finally:
    if hook_id:
        lm.layers.unregister_hook(hook_id)
```

### 3. Track Hook IDs

```python
class Experiment:
    def __init__(self):
        self.hook_ids = []
    
    def add_hook(self, layers, hook, layer_signature):
        hook_id = layers.register_hook(layer_signature, hook)
        self.hook_ids.append(hook_id)
        return hook_id
    
    def cleanup(self, layers):
        for hook_id in self.hook_ids:
            layers.unregister_hook(hook_id)
        self.hook_ids.clear()

# Usage
exp = Experiment()
exp.add_hook(lm.layers, detector, "layer_0")
# ... use hooks ...
exp.cleanup(lm.layers)
```

### 4. Disable Instead of Unregister

For temporary disabling:

```python
# Disable temporarily
detector.disable()
outputs, encodings = lm.inference.execute_inference(["Hello"])  # Hook doesn't execute

# Re-enable
detector.enable()
outputs, encodings = lm.inference.execute_inference(["Hello"])  # Hook executes again

# Unregister when truly done
lm.layers.unregister_hook(hook_id)
```

## Common Patterns

### Conditional Registration

```python
def register_if_needed(layers, hook, layer_signature, condition):
    if condition:
        return layers.register_hook(layer_signature, hook)
    return None
```

### Batch Processing with Hooks

```python
hook_ids = []

try:
    # Register hooks
    for layer in ["layer_0", "layer_5"]:
        det = LayerActivationDetector(layer)
        hook_id = lm.layers.register_hook(layer, det)
        hook_ids.append(hook_id)
    
    # Process batches
    for batch in dataset:
        outputs, encodings = lm.inference.execute_inference(batch)
        # Access detector data
        for hook_id in hook_ids:
            # Get hook and access data
            pass
finally:
    # Cleanup
    for hook_id in hook_ids:
        lm.layers.unregister_hook(hook_id)
```

## Troubleshooting

### Hook Not Executing

- Check hook is registered: `hook_id in lm.layers.context._hook_id_map`
- Verify hook is enabled: `hook.is_enabled()`
- Check layer signature is correct: `lm.layers.list_layers()`

### Memory Leaks

- Always unregister hooks when done
- Use context managers or try/finally blocks
- Check registry size: `len(lm.layers.context._hook_id_map)`

### Registration Errors

- Ensure hook IDs are unique
- Don't mix Detector and Controller on same layer
- Verify layer signature exists

## Next Steps

- **[Advanced Patterns](advanced.md)** - Complex hook management patterns
- **[Using Detectors](detectors.md)** - Detector-specific registration
- **[Using Controllers](controllers.md)** - Controller-specific registration

