# Saving Activations

This workflow guide covers collecting and saving activations from model layers for analysis and SAE training.

## When and Why to Save Activations

Activations are the internal representations that models use to process information. Saving them enables:

- **SAE Training**: Train sparse autoencoders to discover interpretable features
- **Analysis**: Understand what models learn at different layers
- **Debugging**: Inspect model internals during inference
- **Research**: Build datasets for interpretability studies

## Basic Workflow

### Step 1: Load Model and Create Store

```python
from mi_crow.language_model import LanguageModel
from mi_crow.store import LocalStore

store = LocalStore(base_path="./store")
lm = LanguageModel.from_huggingface("gpt2", store=store)
```

### Step 2: Prepare Dataset

```python
from mi_crow.datasets import TextDataset

# Simple text dataset
texts = ["The cat sat on the mat."] * 100
dataset = TextDataset(texts=texts)

# Or use HuggingFace dataset
from mi_crow.datasets import HuggingFaceDataset
dataset = HuggingFaceDataset(
    name="wikitext",
    split="train",
    text_field="text"
)

# For large datasets, you can sample a subset
dataset = TextDataset.from_huggingface(
    "roneneldan/TinyStories",
    split="train",
    store=store,
    text_field="text"
)

# Randomly sample 1000 items (useful for testing or smaller experiments)
sampled_dataset = dataset.random_sample(1000, seed=42)
```

### Step 3: Find Layer Name

```python
# List available layers
layer_names = lm.layers.list_layers()
print(layer_names)

# Example output:
# ['transformer.wte', 'transformer.h.0.attn.c_attn', ...]
```

### Step 4: Save Activations

```python
# Save activations from a specific layer
run_id = lm.activations.save(
    layer_signature="transformer.h.0.attn.c_attn",
    dataset=dataset,
    sample_limit=1000,  # Number of samples to process
    batch_size=4,        # Batch size for processing
    shard_size=64        # Activations per shard file
)

print(f"Saved activations with run_id: {run_id}")
```

The `save` method:
- Processes the dataset in batches
- Captures activations using detector hooks
- Saves to the store in organized shards
- Returns a run_id for later reference

## Layer Selection Strategies

### Choosing the Right Layer

Different layers capture different information:

- **Early layers**: Low-level features (token patterns, syntax)
- **Middle layers**: Semantic combinations
- **Late layers**: High-level concepts (task-specific)

### Common Layer Types

```python
# Attention layers (common choice)
layer = "transformer.h.0.attn.c_attn"

# MLP layers
layer = "transformer.h.0.mlp.c_fc"

# Residual stream (post-attention)
layer = "transformer.h.0"  # If available

# Embedding layer
layer = "transformer.wte"
```

### Finding Layer Names

```python
# List all layers
all_layers = lm.layers.list_layers()

# Filter by pattern
attention_layers = [l for l in all_layers if "attn" in l]
print(f"Found {len(attention_layers)} attention layers")
```

## Batch Processing

### Configuring Batch Size

```python
# Small batch size (lower memory, slower)
run_id = lm.activations.save(
    layer_signature="layer_0",
    dataset=dataset,
    batch_size=2,  # Small batches
    sample_limit=100
)

# Large batch size (higher memory, faster)
run_id = lm.activations.save(
    layer_signature="layer_0",
    dataset=dataset,
    batch_size=32,  # Larger batches
    sample_limit=1000
)
```

**Considerations**:
- GPU memory limits batch size
- Larger batches = faster processing
- Start small and increase if memory allows

### Processing Large Datasets

#### Option 1: Random Sampling

For large datasets, use `random_sample()` to create a manageable subset:

```python
# Load full dataset
dataset = TextDataset.from_huggingface(
    "large-dataset",
    split="train",
    store=store
)

# Sample a subset for activation saving
sampled_dataset = dataset.random_sample(10000, seed=42)
run_id = lm.activations.save(
    layer_signature="layer_0",
    dataset=sampled_dataset,
    sample_limit=10000,
    batch_size=16
)
```

#### Option 2: Process in Chunks

Alternatively, process the dataset in chunks:

```python
# Process in chunks
chunk_size = 1000
total_samples = 10000

for i in range(0, total_samples, chunk_size):
    chunk_dataset = TextDataset(texts=texts[i:i+chunk_size])
    run_id = lm.activations.save(
        layer_signature="layer_0",
        dataset=chunk_dataset,
        sample_limit=chunk_size,
        batch_size=16
    )
    print(f"Processed chunk {i//chunk_size + 1}")
```

## Attention Mask Handling

When saving activations, attention masks ensure only valid tokens are processed:

```python
# Activations are automatically masked
# Only tokens that should be attended to are saved

# The save method handles:
# - Padding tokens (excluded)
# - Special tokens (configurable)
# - Sequence boundaries
```

### Special Token Handling

```python
# By default, special tokens are included
# You can configure this if needed

# Check tokenizer special tokens
print(lm.tokenizer.special_tokens_map)
```

## Storage Organization

Saved activations are organized in the store:

```
store/
└── activations/
    └── <run_id>/
        ├── batch_0/
        │   └── <layer_name>/
        │       └── activations.safetensors
        ├── batch_1/
        │   └── <layer_name>/
        │       └── activations.safetensors
        └── meta.json
```

### Metadata

Each run includes metadata in `meta.json`:

```python
# Access metadata
import json
with open(f"store/activations/{run_id}/meta.json") as f:
    metadata = json.load(f)

print(metadata)
# Contains: layer_name, sample_count, batch_info, etc.
```

### Shard Size

Control how activations are split into files:

```python
# Small shards (more files, easier to load)
run_id = lm.activations.save(
    layer_signature="layer_0",
    dataset=dataset,
    shard_size=32  # 32 samples per file
)

# Large shards (fewer files, faster loading)
run_id = lm.activations.save(
    layer_signature="layer_0",
    dataset=dataset,
    shard_size=256  # 256 samples per file
)
```

## Advanced Usage

### Saving from Multiple Layers

```python
# Save from multiple layers sequentially
layers = ["transformer.h.0.attn.c_attn", "transformer.h.5.attn.c_attn"]

run_ids = {}
for layer in layers:
    run_id = lm.activations.save(
        layer_signature=layer,
        dataset=dataset,
        sample_limit=1000
    )
    run_ids[layer] = run_id
```

### Custom Activation Saving

```python
from mi_crow.hooks import LayerActivationDetector

# Manual saving with custom detector
detector = LayerActivationDetector("transformer.h.0.attn.c_attn")
hook_id = lm.layers.register_hook("transformer.h.0.attn.c_attn", detector)

# Process dataset
for batch in dataset:
    outputs, encodings = lm.inference.execute_inference(batch)
    activations = detector.get_captured()
    # Save manually
    detector.clear_captured()
```

## Verification

After saving, verify the activations:

```python
# Check run exists
from pathlib import Path
run_path = Path(f"store/activations/{run_id}")
assert run_path.exists()

# Check metadata
import json
with open(run_path / "meta.json") as f:
    meta = json.load(f)
    print(f"Samples: {meta['sample_count']}")
    print(f"Batches: {meta['batch_count']}")
```

## Common Issues

### Out of Memory

```python
# Solution: Reduce batch size
run_id = lm.activations.save(
    layer_signature="layer_0",
    dataset=dataset,
    batch_size=1,  # Minimal batch size
    sample_limit=100
)
```

### Layer Not Found

```python
# Solution: List available layers first
layers = lm.layers.list_layers()
print("Available layers:", layers)

# Use exact layer name from list
```

### Slow Processing

```python
# Solution: Increase batch size (if memory allows)
run_id = lm.activations.save(
    layer_signature="layer_0",
    dataset=dataset,
    batch_size=32,  # Larger batches
    sample_limit=1000
)
```

## Next Steps

After saving activations:

- **[Training SAE Models](training-sae.md)** - Train SAEs on saved activations
- **[Hooks: Detectors](../hooks/detectors.md)** - Learn about detector hooks
- **[Examples](../examples.md)** - See example notebooks

## Related Examples

- `examples/04_save_inputs_and_outputs.ipynb` - Saving inputs and outputs
- `examples/06_save_activations_with_attention_masks.ipynb` - Attention mask handling
- `examples/07_save_activations_and_attention_masks.ipynb` - Advanced saving

