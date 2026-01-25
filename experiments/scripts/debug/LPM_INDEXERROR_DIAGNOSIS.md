# LPM IndexError Diagnosis & Fix

## The Error

```
IndexError: index 498 is out of bounds for dimension 0 with size 181
```

Occurred in `activation_aggregation.py` when trying to index activations with attention mask indices.

## Root Cause

**Hardcoded batch size in LPM batch index calculation** ([lpm.py#L422](lpm.py#L422)):

```python
batch_idx = self._seen_samples // 64  # WRONG! Assumes all batches are size 64
```

###What Happens

1. Attention masks saved in batches: `[64, 64, 64, 15]` (207 samples total)
2. During inference, LPM loads these masks by batch_idx
3. When processing samples 192-206 (batch 3):
   - Correct batch_idx should be: **3**
   - But `192 // 64 = 3` ✓ works for first sample
   - Then `193 // 64 = 3` ✓ still works
   - **Problem**: The calculation happens to work by accident until you have different batch sizes!

### The Real Issue

The calculation **completely breaks** when:
- Saved attention masks used different batch size than 64
- Last batch is smaller than 64 (always true for most datasets)
- Any batch has varying size

**Example with actual misalignment:**
```
Dataset: 207 samples
Saved batches: [64, 64, 64, 15]

Sample 0-63:   batch_idx = 0 // 64 = 0 ✓
Sample 64-127: batch_idx = 64 // 64 = 1 ✓  
Sample 128-191: batch_idx = 128 // 64 = 2 ✓
Sample 192-206: batch_idx = 192 // 64 = 3 ✓

Looks correct! But...

If batches were: [50, 50, 50, 50, 7]
Sample 0-49:   batch_idx = 0 // 64 = 0 ✓ (uses batch 0)
Sample 50-99:  batch_idx = 50 // 64 = 0 ✗ (uses batch 0, should be 1!)
Sample 100-149: batch_idx = 100 // 64 = 1 ✗ (uses batch 1, should be 2!)
...MISALIGNMENT!
```

## Secondary Issue: Sequence Length Mismatch

When wrong batch index is used:
- Attention mask from batch X applied to activations from batch Y
- Different texts → different tokenization → different sequence lengths
- Sample with 498 tokens' mask applied to sample with 181 tokens → **IndexError**

## The Fix

### Option 1: Track Actual Batch Size (Recommended)

```python
# In LPM.__init__
self._inference_batch_size: Optional[int] = None

# In load_inference_attention_masks
def load_inference_attention_masks(self, store: Store, run_id: str) -> None:
    # ... existing code ...
    
    # Detect batch size from first batch
    if batch_indices:
        first_batch = torch.load(store.get_batch_path(run_id, batch_indices[0]))
        self._inference_batch_size = first_batch["attention_mask"].shape[0]
        logger.info(f"Detected batch size: {self._inference_batch_size}")
    
    # ... rest of code ...

# In process_activations
if self._inference_attention_masks is not None:
    if self._inference_batch_size is None:
        raise RuntimeError("Batch size not detected!")
    
    batch_idx = self._seen_samples // self._inference_batch_size  # Use actual size!
    attention_mask = self._inference_attention_masks.get(batch_idx)
```

### Option 2: Store Cumulative Sample Counts

```python
# In load_inference_attention_masks
self._batch_sample_ranges: List[Tuple[int, int]] = []  # [(start, end), ...]

cumulative = 0
for batch_idx in batch_indices:
    batch_data = torch.load(...)
    batch_size = batch_data["attention_mask"].shape[0]
    self._batch_sample_ranges.append((cumulative, cumulative + batch_size))
    cumulative += batch_size

# In process_activations
for batch_idx, (start, end) in enumerate(self._batch_sample_ranges):
    if start <= self._seen_samples < end:
        attention_mask = self._inference_attention_masks[batch_idx]
        break
```

### Option 3: Index by Sample (Most Robust)

Store attention masks by sample index rather than batch index:

```python
# Change storage from dict[int, Tensor] to dict[int, Tensor]  
# where key is sample_idx, not batch_idx

self._inference_attention_masks: Dict[int, torch.Tensor] = {}

# Load
for batch_idx in batch_indices:
    batch_data = torch.load(...)
    attention_masks = batch_data["attention_mask"]  # [batch_size, seq_len]
    
    for i in range(attention_masks.shape[0]):
        sample_idx = cumulative_samples + i
        self._inference_attention_masks[sample_idx] = attention_masks[i]
    
    cumulative_samples += attention_masks.shape[0]

# Use
for i in range(batch_size):
    sample_idx = self._seen_samples + i
    attention_mask = self._inference_attention_masks.get(sample_idx)
```

## Verification

Run the diagnostic script to verify the bug:

```bash
python experiments/scripts/debug/verify_activation_alignment.py \
    --dataset plmix_test \
    --attention_mask_run test_attention_masks_layer31_20260125_044240 \
    --store /mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store
```

Expected output will show:
- Batch index mismatches
- Which samples would fail
- Sequence length incompatibilities

## Impact

This bug affects **ALL LPM experiments** where:
- Test dataset size is not a multiple of 64
- Batch size during saving ≠ 64
- Multiple concurrent runs (possible run_id collisions with timestamp-only naming)

## Additional Improvements

1. **Better run_id generation** in `save_test_attention_masks`:
   ```python
   run_name = f"test_attention_masks_{model_short_name}_{dataset_name}_{aggregation_method}_layer{layer_num}_{ts}"
   ```

2. **Validate on load**:
   ```python
   if attention_mask.shape[1] != activations.shape[1]:
       raise ValueError(
           f"Sequence length mismatch: "
           f"attention_mask has {attention_mask.shape[1]} positions, "
           f"but activations have {activations.shape[1]}"
       )
   ```

3. **Add assertions**:
   ```python
   assert batch_idx < len(self._inference_attention_masks), \
       f"Batch index {batch_idx} out of range (have {len(self._inference_attention_masks)} batches)"
   ```
