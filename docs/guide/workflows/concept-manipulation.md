# Concept Manipulation

This guide covers controlling model behavior by manipulating discovered SAE concepts.

## Overview

Concept manipulation allows you to:
- Amplify or suppress specific concepts
- Compare model behavior with/without interventions
- Steer model outputs toward desired behaviors
- Run controlled intervention experiments

## Prerequisites

Before manipulating concepts, you need:
- A trained SAE attached to the model
- Discovered concepts (see [Concept Discovery](concept-discovery.md))
- The SAE registered on the target layer

## Basic Concept Manipulation

### Amplify a Concept

```python
# Amplify neuron 42 (which represents a concept)
sae.concepts.manipulate_concept(neuron_idx=42, scale=1.5)

# Run inference with amplified concept
outputs, encodings = lm.inference.execute_inference(["Your prompt here"])
```

**Scale values**:
- `scale > 1.0`: Amplify (increase concept strength)
- `scale < 1.0`: Suppress (decrease concept strength)
- `scale = 0.0`: Completely remove concept

### Suppress a Concept

```python
# Suppress neuron 42
sae.concepts.manipulate_concept(neuron_idx=42, scale=0.5)

# Run inference
outputs, encodings = lm.inference.execute_inference(["Your prompt here"])
```

### Reset Manipulation

```python
# Reset to original (scale = 1.0)
sae.concepts.manipulate_concept(neuron_idx=42, scale=1.0)

# Or reset all manipulations
sae.concepts.reset_manipulations()
```

## A/B Testing Interventions

Compare model behavior with and without concept manipulation:

### Baseline (No Manipulation)

```python
# Get baseline output
baseline_outputs, _ = lm.inference.execute_inference(
    ["Your prompt here"],
    with_controllers=False  # Disable SAE manipulation
)
```

### With Manipulation

```python
# Apply concept manipulation
sae.concepts.manipulate_concept(neuron_idx=42, scale=1.5)

# Get manipulated output
manipulated_outputs, _ = lm.inference.execute_inference(
    ["Your prompt here"],
    with_controllers=True  # Enable SAE manipulation
)
```

### Compare Results

```python
# Compare logits
difference = manipulated_outputs.logits - baseline_outputs.logits

# Compare predictions
baseline_pred = baseline_outputs.logits.argmax(dim=-1)
manipulated_pred = manipulated_outputs.logits.argmax(dim=-1)

print(f"Baseline prediction: {baseline_pred}")
print(f"Manipulated prediction: {manipulated_pred}")
```

## Multiple Concept Manipulation

### Manipulate Multiple Neurons

```python
# Amplify multiple concepts
sae.concepts.manipulate_concept(neuron_idx=42, scale=1.5)  # Concept A
sae.concepts.manipulate_concept(neuron_idx=100, scale=2.0)  # Concept B
sae.concepts.manipulate_concept(neuron_idx=200, scale=0.5)  # Suppress Concept C

# All manipulations apply simultaneously
outputs, encodings = lm.inference.execute_inference(["Your prompt here"])
```

### Batch Manipulation

```python
# Manipulate multiple concepts at once
manipulations = {
    42: 1.5,   # Amplify concept A
    100: 2.0,  # Amplify concept B
    200: 0.5   # Suppress concept C
}

for neuron_idx, scale in manipulations.items():
    sae.concepts.manipulate_concept(neuron_idx=neuron_idx, scale=scale)
```

## Real-Time Control

### Dynamic Manipulation

```python
# Change manipulation during generation
def generate_with_control(prompt, concept_idx, scale):
    # Set manipulation
    sae.concepts.manipulate_concept(neuron_idx=concept_idx, scale=scale)
    
    # Generate
    outputs, encodings = lm.inference.execute_inference([prompt])
    
    return outputs

# Use different scales
output1 = generate_with_control("Tell me about", neuron_idx=42, scale=1.0)
output2 = generate_with_control("Tell me about", neuron_idx=42, scale=1.5)
output3 = generate_with_control("Tell me about", neuron_idx=42, scale=2.0)
```

### Conditional Manipulation

```python
# Manipulate based on input
def conditional_manipulate(prompt):
    if "science" in prompt.lower():
        # Amplify scientific concepts
        sae.concepts.manipulate_concept(neuron_idx=200, scale=1.5)
    elif "family" in prompt.lower():
        # Amplify family concepts
        sae.concepts.manipulate_concept(neuron_idx=42, scale=1.5)
    
    return lm.inference.execute_inference([prompt])
```

## Concept Configurations

Save and load concept manipulation configurations:

### Save Configuration

```python
# Get current manipulations
config = sae.concepts.get_manipulation_config()

# Save to file
import json
with open("concept_config.json", "w") as f:
    json.dump(config, f, indent=2)
```

### Load Configuration

```python
# Load from file
with open("concept_config.json", "r") as f:
    config = json.load(f)

# Apply configuration
for neuron_idx, scale in config.items():
    sae.concepts.manipulate_concept(neuron_idx=int(neuron_idx), scale=scale)
```

## Use Cases

### Steering Generation

```python
# Steer model toward specific topics
sae.concepts.manipulate_concept(neuron_idx=42, scale=2.0)  # Science concept
outputs, encodings = lm.inference.execute_inference(["Write about"])
```

### Reducing Bias

```python
# Suppress potentially biased concepts
biased_concept_neurons = [100, 150, 200]  # Identified through analysis

for neuron_idx in biased_concept_neurons:
    sae.concepts.manipulate_concept(neuron_idx=neuron_idx, scale=0.3)

outputs, encodings = lm.inference.execute_inference(["Your prompt"])
```

### Enhancing Specific Features

```python
# Enhance desired features
desired_features = {
    42: 1.5,   # Clarity
    100: 1.3,  # Accuracy
    200: 1.2   # Helpfulness
}

for neuron_idx, scale in desired_features.items():
    sae.concepts.manipulate_concept(neuron_idx=neuron_idx, scale=scale)
```

## Advanced Patterns

### Gradual Manipulation

```python
# Gradually increase concept strength
for scale in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
    sae.concepts.manipulate_concept(neuron_idx=42, scale=scale)
    outputs, encodings = lm.inference.execute_inference(["Your prompt"])
    print(f"Scale {scale}: {outputs.logits[0, 0, :5]}")  # First 5 logits
```

### Concept Interaction Studies

```python
# Study interactions between concepts
concept_a = 42
concept_b = 100

# Individual effects
sae.concepts.manipulate_concept(neuron_idx=concept_a, scale=1.5)
output_a, _ = lm.inference.execute_inference(["Prompt"])

sae.concepts.reset_manipulations()
sae.concepts.manipulate_concept(neuron_idx=concept_b, scale=1.5)
output_b, _ = lm.inference.execute_inference(["Prompt"])

# Combined effect
sae.concepts.reset_manipulations()
sae.concepts.manipulate_concept(neuron_idx=concept_a, scale=1.5)
sae.concepts.manipulate_concept(neuron_idx=concept_b, scale=1.5)
output_combined, _ = lm.inference.execute_inference(["Prompt"])

# Compare
print("Individual A:", output_a.logits)
print("Individual B:", output_b.logits)
print("Combined:", output_combined.logits)
```

## Best Practices

1. **Start small**: Use moderate scales (1.2-1.5) initially
2. **Test systematically**: Compare baseline vs manipulated
3. **Document effects**: Record what each manipulation does
4. **Reset between experiments**: Use `reset_manipulations()`
5. **Validate concepts**: Ensure concepts are well-understood

## Common Issues

### No Effect

```python
# Check SAE is attached
assert sae in lm.layers.get_hooks()

# Check manipulation is applied
config = sae.concepts.get_manipulation_config()
print(f"Current manipulations: {config}")

# Ensure with_controllers=True
outputs, encodings = lm.inference.execute_inference(["Prompt"], with_controllers=True)
```

### Too Strong Effect

```python
# Reduce scale
sae.concepts.manipulate_concept(neuron_idx=42, scale=1.2)  # Instead of 2.0
```

### Unexpected Behavior

```python
# Verify concept is correct
top_texts = sae.concepts.get_top_texts()
print(f"Neuron 42 top texts: {top_texts.get(42, [])[:5]}")
```

## Next Steps

After learning concept manipulation:

- **[Activation Control](activation-control.md)** - Direct activation manipulation
- **[Hooks: Controllers](../hooks/controllers.md)** - Custom controller hooks
- **[Examples](../examples.md)** - See example notebooks

## Related Examples

- `examples/03_load_concepts.ipynb` - Complete concept manipulation example
- `experiments/verify_sae_training/05_show_concepts.ipynb` - Concept visualization

