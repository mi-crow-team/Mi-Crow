# Hooks Fundamentals

This guide covers the fundamental concepts of the hooks system: the base Hook class, hook types, lifecycle, and basic usage patterns.

## Understanding the Hook Base Class

All hooks in mi-crow inherit from the `Hook` base class, which provides:

- **Unique identification**: Each hook has a unique ID
- **Layer association**: Hooks know which layer they're attached to
- **Type specification**: FORWARD or PRE_FORWARD
- **Enable/disable**: Toggle hook execution without unregistering
- **Context access**: Access to the language model context

### Hook Initialization

```python
from mi_crow.hooks.hook import Hook, HookType

# Hooks are typically created by subclasses
# But you can see the initialization parameters:

hook = SomeHook(
    layer_signature="transformer.h.0.attn.c_attn",  # Optional: layer name
    hook_type=HookType.FORWARD,                     # FORWARD or PRE_FORWARD
    hook_id="my-custom-id"                          # Optional: custom ID
)
```

### Hook ID

Every hook gets a unique ID, either:
- Auto-generated UUID if not provided
- Custom ID if specified during creation

The ID is used for:
- Unregistering hooks
- Looking up hooks in the registry
- Error reporting

## Hook Types: FORWARD vs PRE_FORWARD

Hooks execute at different points in the forward pass:

### FORWARD Hooks

Execute **after** a layer produces its output.

```python
# Hook receives the layer's output
def hook_fn(module, input, output):
    # output is the layer's activation
    # Can modify and return new output
    return modified_output
```

**When to use**:
- Most common hook type
- Operating on layer activations (outputs)
- SAE decoding
- Activation analysis
- Concept manipulation

### PRE_FORWARD Hooks

Execute **before** a layer processes its input.

```python
# Hook receives the layer's input
def hook_fn(module, input):
    # input is the layer's input tuple
    # Can modify and return new input
    return modified_input
```

**When to use**:
- Modifying inputs before processing
- Early intervention in the forward pass
- Input preprocessing

### Choosing Hook Type

Most use cases use FORWARD hooks because:
- Activations (outputs) are what we typically analyze
- SAEs decode outputs, not inputs
- Concept manipulation operates on outputs

Use PRE_FORWARD only when you need to modify inputs.

## Hook Lifecycle

Understanding the hook lifecycle is crucial for proper usage:

### 1. Creation

```python
from mi_crow.hooks import LayerActivationDetector

detector = LayerActivationDetector(
    layer_signature="transformer.h.0.attn.c_attn"
)
```

At this point, the hook exists but isn't active.

### 2. Registration

```python
hook_id = lm.layers.register_hook(
    layer_signature="transformer.h.0.attn.c_attn",
    hook=detector
)
```

Registration:
- Attaches the hook to the specified layer
- Creates a PyTorch hook handle
- Adds hook to the registry
- Returns the hook ID

### 3. Execution

During inference, the hook automatically executes:

```python
# Hook executes automatically during forward pass
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])
```

The hook's `_hook_fn` method is called for each forward pass.

### 4. Enable/Disable

You can temporarily disable hooks without unregistering:

```python
# Disable
detector.disable()

# Hook won't execute
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])

# Re-enable
detector.enable()

# Hook executes again
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])
```

### 5. Cleanup

Always unregister hooks when done:

```python
lm.layers.unregister_hook(hook_id)
```

This:
- Removes the PyTorch hook handle
- Removes hook from registry
- Prevents memory leaks

## Hook Context and Layer Signatures

### Layer Signatures

Layer signatures identify which layer to attach a hook to:

```python
# By name (string)
layer_signature = "transformer.h.0.attn.c_attn"

# By index (integer)
layer_signature = 0  # First layer

# Register hook
hook_id = lm.layers.register_hook(layer_signature, hook)
```

You can find available layers:

```python
# List all layer names
layer_names = lm.layers.list_layers()
print(layer_names)
```

### Hook Context

When registered, hooks receive access to the language model context:

```python
# Context is automatically set during registration
# Access it in your hook implementation:

class MyHook(Hook):
    def _hook_fn(self, module, input, output):
        # Access context
        context = self._context
        model = context.language_model
        # Use context for advanced operations
        ...
```

The context provides access to:
- The language model instance
- The layers manager
- The store
- Other registered hooks

## Enabling and Disabling Hooks

Hooks can be enabled/disabled without unregistering:

```python
# Disable a hook
hook.disable()

# Check if enabled
if hook.is_enabled():
    print("Hook is active")

# Re-enable
hook.enable()
```

**Use cases**:
- Temporarily skip hook execution
- A/B testing (with vs without hook)
- Performance optimization
- Conditional execution

## Hook Error Handling

Hooks have built-in error handling:

```python
from mi_crow.hooks.hook import HookError

try:
    outputs, encodings = lm.inference.execute_inference(["Hello, world!"])
except HookError as e:
    print(f"Hook {e.hook_id} failed: {e.original_error}")
```

Hook errors:
- Don't crash the entire forward pass
- Are wrapped in `HookError` with context
- Include hook ID and original error
- Allow graceful degradation

### Best Practices

1. **Handle errors in hooks**: Don't let exceptions propagate
2. **Validate inputs**: Check tensor shapes and types
3. **Use try/except**: Catch and handle errors gracefully
4. **Log errors**: Use logging for debugging

## Basic Usage Pattern

Here's the standard pattern for using hooks:

```python
from mi_crow.hooks import LayerActivationDetector
from mi_crow.language_model import LanguageModel
from mi_crow.store import LocalStore

# 1. Setup
store = LocalStore(base_path="./store")
lm = LanguageModel.from_huggingface("gpt2", store=store)

# 2. Create hook
detector = LayerActivationDetector(
    layer_signature="transformer.h.0.attn.c_attn"
)

# 3. Register hook
hook_id = lm.layers.register_hook("transformer.h.0.attn.c_attn", detector)

try:
    # 4. Use hook (runs automatically)
    outputs, encodings = lm.inference.execute_inference(["Hello, world!"])
    
    # 5. Access hook data
    activations = detector.tensor_metadata.get("activations")
    
finally:
    # 6. Always cleanup
    lm.layers.unregister_hook(hook_id)
```

## Common Patterns

### Multiple Hooks on Different Layers

```python
# Register hooks on multiple layers
hook1_id = lm.layers.register_hook("layer_0", detector1)
hook2_id = lm.layers.register_hook("layer_10", detector2)

# All hooks execute during forward pass
outputs, encodings = lm.inference.execute_inference(["Hello, world!"])

# Cleanup all
lm.layers.unregister_hook(hook1_id)
lm.layers.unregister_hook(hook2_id)
```

### Conditional Hook Execution

```python
class ConditionalHook(Detector):
    def _hook_fn(self, module, input, output):
        if some_condition:
            # Only process when condition is met
            self.process_activation(output)
```

### Hook Composition

```python
# Register multiple hooks on same layer (if compatible)
# Note: Only one hook class type (Detector or Controller) per layer
detector_id = lm.layers.register_hook("layer_0", detector)
# Can't register another detector on same layer
# But can register a controller if needed
```

## Next Steps

Now that you understand the fundamentals:

- **[Using Detectors](detectors.md)** - Learn about detector hooks
- **[Using Controllers](controllers.md)** - Learn about controller hooks
- **[Hook Registration](registration.md)** - Detailed registration guide
- **[Advanced Patterns](advanced.md)** - Complex hook patterns

