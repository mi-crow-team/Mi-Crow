# Datasets API

Dataset loading and management utilities for text and classification datasets.

::: mi_crow.datasets


## TextDataset.random_sample()

The `TextDataset.random_sample()` method creates a new `TextDataset` instance with randomly sampled items from the original dataset. This is useful for creating smaller subsets of large datasets for testing or training.

### Parameters

- `n` (int): Number of items to sample. Must be greater than 0.
- `seed` (Optional[int]): Optional random seed for reproducibility. If provided, ensures the same random sample is generated across runs.

### Returns

A new `TextDataset` instance containing the randomly sampled items.

### Example

```python
from mi_crow.datasets import TextDataset
from mi_crow.store import LocalStore

store = LocalStore(base_path="./store")

# Load a large dataset
dataset = TextDataset.from_huggingface(
    "roneneldan/TinyStories",
    split="train",
    store=store,
    text_field="text"
)

print(f"Original dataset size: {len(dataset)}")  # e.g., 2119719

# Sample 1000 random items
sampled_dataset = dataset.random_sample(1000, seed=42)
print(f"Sampled dataset size: {len(sampled_dataset)}")  # 1000

# Use the sampled dataset for activation saving or training
run_id = lm.activations.save_activations_dataset(
    dataset=sampled_dataset,
    layer_signature="layer_0",
    batch_size=4
)
```

### Notes

- Works with `MEMORY` and `DISK` loading strategies only. Not supported for `STREAMING` datasets.
- If `n >= len(dataset)`, returns all items in random order.
- The method preserves the original dataset's loading strategy, store, and text field configuration.
- For reproducible results, always specify a `seed` parameter.
