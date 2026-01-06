# Concept Discovery

This guide covers discovering interpretable concepts by analyzing SAE neuron activations and collecting top activating texts.

## Overview

Concept discovery involves:
1. Attaching a trained SAE to the model
2. Enabling text tracking
3. Running inference on a dataset
4. Analyzing top activating texts for each neuron
5. Manually curating concept names

## Prerequisites

Before discovering concepts, you need:
- A trained SAE (see [Training SAE Models](training-sae.md))
- A dataset for inference
- The SAE attached to the model

## Step 1: Load and Attach SAE

```python
from mi_crow.mechanistic.sae import TopKSae
import torch

# Load trained SAE
sae = TopKSae(n_latents=4096, n_inputs=768, k=32, device="cuda")
sae.load_state_dict(torch.load("sae_model.pt"))
sae.eval()

# Attach to model
lm.attach_sae(sae, layer_signature="transformer.h.0.attn.c_attn")
```

The `attach_sae` method:
- Registers the SAE as a hook on the specified layer
- Enables SAE decoding during inference
- Makes the SAE available for concept operations

## Step 2: Enable Text Tracking

```python
# Enable automatic text tracking
sae.concepts.enable_text_tracking(top_k=10)
```

**Parameters**:
- `top_k`: Number of top activating texts to track per neuron
- Higher k = more examples but more memory

Text tracking:
- Automatically collects text snippets during inference
- Tracks which texts activate each neuron most strongly
- Accumulates data across batches

## Step 3: Run Inference on Dataset

```python
from mi_crow.datasets import TextDataset

# Prepare dataset
texts = [
    "The cat sat on the mat.",
    "Dogs are loyal companions.",
    "Science requires careful observation.",
    # ... more texts
] * 100  # Repeat for more examples

dataset = TextDataset(texts=texts)

# Run inference - text tracking happens automatically
outputs, encodings = lm.inference.execute_inference(dataset.texts)
```

During inference:
- SAE decodes activations to sparse latents
- Text tracker records which texts activate each neuron
- Top-K texts are maintained for each neuron

## Step 4: Get Top Texts

```python
# Get top activating texts for all neurons
top_texts = sae.concepts.get_top_texts()

# Access texts for a specific neuron
neuron_42_texts = top_texts.get(42, [])
print(f"Neuron 42 top texts:")
for text, activation in neuron_42_texts:
    print(f"  Activation: {activation:.4f} - {text}")
```

### Understanding Top Texts

Each entry contains:
- **Text**: The input text that activated the neuron
- **Activation**: The activation strength (higher = stronger)

Patterns in top texts reveal what the neuron detects:
- Semantic concepts (e.g., "family", "science")
- Syntactic patterns (e.g., "question words", "past tense")
- Domain-specific (e.g., "medical terms", "programming")

## Step 5: Export for Manual Curation

```python
# Export top texts to JSON
import json

with open("top_texts.json", "w") as f:
    json.dump(top_texts, f, indent=2)

# Or export in a more readable format
export_data = []
for neuron_idx, texts in top_texts.items():
    for text, activation in texts:
        export_data.append({
            "neuron": neuron_idx,
            "text": text,
            "activation": activation
        })

with open("top_texts_flat.json", "w") as f:
    json.dump(export_data, f, indent=2)
```

## Step 6: Manual Concept Curation

Create a CSV file mapping neurons to concept names:

```csv
neuron_idx,concept_name,confidence
0,family relationships,0.9
0,parent-child interactions,0.8
1,nature and outdoors,0.9
1,animals and wildlife,0.8
2,scientific terminology,0.95
2,academic language,0.85
```

**Curation Tips**:
- Look for common themes in top texts
- Assign multiple concepts if neuron is polysemous
- Use confidence scores to indicate certainty
- Review multiple top texts, not just the first

## Step 7: Load Curated Concepts

```python
import pandas as pd

# Load curated concepts
concepts_df = pd.read_csv("curated_concepts.csv")

# Create concept dictionary
concepts = {}
for _, row in concepts_df.iterrows():
    neuron_idx = int(row['neuron_idx'])
    concept_name = row['concept_name']
    confidence = float(row['confidence'])
    
    if neuron_idx not in concepts:
        concepts[neuron_idx] = []
    
    concepts[neuron_idx].append({
        'name': concept_name,
        'confidence': confidence
    })

# Save to SAE
sae.concepts.load_concepts(concepts)
```

## Analyzing Concepts

### Most Active Neurons

```python
# Find neurons with highest average activation
neuron_activity = {}
for neuron_idx, texts in top_texts.items():
    if texts:
        avg_activation = sum(act for _, act in texts) / len(texts)
        neuron_activity[neuron_idx] = avg_activation

# Sort by activity
sorted_neurons = sorted(neuron_activity.items(), key=lambda x: x[1], reverse=True)
print("Most active neurons:")
for neuron_idx, activity in sorted_neurons[:10]:
    print(f"  Neuron {neuron_idx}: {activity:.4f}")
```

### Concept Distribution

```python
# Analyze concept coverage
concept_counts = {}
for neuron_idx, concepts_list in concepts.items():
    for concept in concepts_list:
        name = concept['name']
        concept_counts[name] = concept_counts.get(name, 0) + 1

print("Concept distribution:")
for concept, count in sorted(concept_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {concept}: {count} neurons")
```

## Validation

### Check Concept Quality

```python
# Verify concepts make sense
for neuron_idx in [0, 1, 2]:  # Check first few
    texts = top_texts.get(neuron_idx, [])
    concepts_list = concepts.get(neuron_idx, [])
    
    print(f"\nNeuron {neuron_idx}:")
    print(f"  Concepts: {[c['name'] for c in concepts_list]}")
    print(f"  Top texts:")
    for text, act in texts[:3]:
        print(f"    {act:.4f}: {text[:50]}...")
```

### Test Concept Consistency

```python
# Run inference on new texts and check if concepts activate
test_texts = [
    "My family loves to travel together.",
    "The scientist conducted experiments.",
    "Dogs are friendly animals."
]

outputs, encodings = lm.inference.execute_inference(test_texts)

# Check which concepts activated
# (Implementation depends on SAE concept API)
```

## Advanced Techniques

### Filtering by Activation Threshold

```python
# Only consider texts above threshold
threshold = 0.5
filtered_texts = {}

for neuron_idx, texts in top_texts.items():
    filtered = [(text, act) for text, act in texts if act > threshold]
    if filtered:
        filtered_texts[neuron_idx] = filtered
```

### Clustering Similar Neurons

```python
# Group neurons with similar activation patterns
from sklearn.cluster import KMeans
import numpy as np

# Create activation matrix (neurons x samples)
# Then cluster neurons
# (Implementation depends on your data structure)
```

## Best Practices

1. **Use diverse datasets**: Include various topics and styles
2. **Review multiple top texts**: Don't judge by first example alone
3. **Check for polysemy**: Neurons may detect multiple concepts
4. **Validate on held-out data**: Test concepts on new texts
5. **Iterate**: Refine concepts based on analysis

## Common Issues

### No Texts Collected

```python
# Solution: Ensure text tracking is enabled
sae.concepts.enable_text_tracking(top_k=10)

# And run inference
outputs, encodings = lm.inference.execute_inference(dataset.texts)
```

### Too Many/Few Examples

```python
# Adjust top_k
sae.concepts.enable_text_tracking(top_k=20)  # More examples
# or
sae.concepts.enable_text_tracking(top_k=5)   # Fewer examples
```

### Unclear Concepts

```python
# Solution: Use more diverse dataset
# Or check if SAE training was successful
# Or increase dataset size
```

## Next Steps

After discovering concepts:

- **[Concept Manipulation](concept-manipulation.md)** - Use concepts to control model
- **[Training SAE Models](training-sae.md)** - Improve SAE if concepts unclear
- **[Examples](../examples.md)** - See example notebooks

## Related Examples

- `examples/02_attach_sae_and_save_texts.ipynb` - Complete concept discovery example
- `experiments/verify_sae_training/04_name_sae_concepts.ipynb` - Concept naming

