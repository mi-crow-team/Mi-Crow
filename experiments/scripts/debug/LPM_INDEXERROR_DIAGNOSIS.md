# LPM IndexError Diagnosis & Fix

## The Error

```
IndexError: index 498 is out of bounds for dimension 0 with size 181
```

Occurred in `activation_aggregation.py` when trying to index activations with attention mask indices.

## CORRECTED Analysis

After reviewing the actual code, the issue is **NOT** a hardcoded batch size. The code correctly uses:
```python
batch_size = tensor.shape[0]  # From actual activations
batch_idx = self._seen_samples // batch_size
```

## The Real Problem: Sequence Length Mismatch

The error indicates:
- Attention mask expects sequence length of 498+ tokens
- But activations only have 181 tokens

This is a **sequence length mismatch**, not a batch index issue.

### Possible Causes

1. **Different tokenization between save and inference** (UNLIKELY - same model, same max_length)

2. **Sample misalignment** - Wrong attention mask applied to wrong sample (LIKELY!)

3. **Non-deterministic dataset loading** - Dataset order differs between attention mask saving and inference (POSSIBLE)

4. **Concurrent run collision** - Multiple experiments writing to same run_id simultaneously (POSSIBLE)

### Why Sample Misalignment Could Happen

Looking at the batch indexing logic in LPM:

```python
batch_idx = self._seen_samples // batch_size
```

This calculation is **correct** IF:
- Batches are processed sequentially  
- `_seen_samples` increments correctly
- No skipped batches

But it **fails** IF:
- Detector is called multiple times per batch
- `_seen_samples` increments incorrectly
- Batches are processed out of order

### The Inference Loop Issue

In `run_lpm_experiment.py`, inference is done in a loop:

```python
for batch_idx in range(num_batches):
    # ...
    _ = lm.inference.execute_inference(batch_texts, ...)
```

Each `execute_inference` call triggers the LPM detector, which:
1. Increments `_seen_samples` 
2. Calculates `batch_idx = _seen_samples // batch_size`
3. Loads attention mask for that batch_idx

**Problem**: If `_seen_samples` doesn't reset properly or increments incorrectly, the calculation becomes wrong!

## Hypothesis

The most likely cause is **`_seen_samples` tracking issue**:

1. LPM loads attention masks with batch indices: 0, 1, 2, 3
2. During first inference batch: `_seen_samples=0`, `batch_idx=0 // 64=0` ✓
3. After first batch: `_seen_samples` should be 64
4. During second batch: `_seen_samples=64`, `batch_idx=64 // 64=1` ✓
5. But if `_seen_samples` is incremented PER SAMPLE (not per batch), it goes 0,1,2,3...64
6. Or if detector is called multiple times per batch, count becomes wrong

## Verification Needed

The verification script should check:
1. Are dataset samples in the same order when loading?
2. Do attention mask batch indices match dataset batches?
3. Is `_seen_samples` incrementing correctly in LPM?
4. Are sequence lengths consistent for the same samples?

## Recommended Fix

Add validation and logging:

```python
# In LPM.process_activations
if self._inference_attention_masks is not None:
    batch_idx = self._seen_samples // batch_size
    
    if batch_idx not in self._inference_attention_masks:
        raise RuntimeError(
            f"Batch index {batch_idx} not found in loaded attention masks. "
            f"Have batches: {list(self._inference_attention_masks.keys())}, "
            f"_seen_samples={self._seen_samples}, batch_size={batch_size}"
        )
    
    attention_mask = self._inference_attention_masks[batch_idx]
    
    # Validate sequence length matches
    if attention_mask.shape[1] != tensor.shape[1]:
        raise ValueError(
            f"Sequence length mismatch at batch {batch_idx}: "
            f"attention_mask has {attention_mask.shape[1]} positions, "
            f"but activations have {tensor.shape[1]} positions. "
            f"This indicates wrong attention mask is being applied!"
        )
```

## Additional Investigation Needed

Run the verification script to:
1. Check if dataset loading is deterministic
2. Verify attention mask batch structure
3. Simulate the inference process to find where misalignment occurs
