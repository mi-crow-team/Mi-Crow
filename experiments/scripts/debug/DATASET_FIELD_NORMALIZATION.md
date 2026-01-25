# Dataset Field Normalization Issue - Analysis

## The Error

```
KeyError: 'prompt'
```

When running `verify_dataset_determinism.py`, the script tried to access `item["prompt"]` but the key didn't exist in the item dictionary.

## Root Cause

**ClassificationDataset normalizes the text field name to "text" in returned items.**

When you call `dataset.iter_items()`, the `_extract_item_from_row()` method (lines 96-128 in classification_dataset.py) does the following:

```python
def _extract_item_from_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
    # Extract text from row using self._text_field (e.g., "prompt")
    if self._text_field in row:
        text = row[self._text_field]
    elif "text" in row:
        text = row["text"]
    else:
        raise ValueError(...)
    
    # CRITICAL: Always use "text" as the key, not the original field name
    item = {"text": text}
    
    # But category fields keep their original names
    for cat_field in self._category_fields:
        category = row.get(cat_field)
        item[cat_field] = category
    
    return item
```

**Result:**
- Original dataset has columns: `["prompt", "prompt_harm_label", "subcategory"]`
- Items returned by `iter_items()`: `{"text": "...", "prompt_harm_label": "...", "subcategory": "..."}`
- The `"prompt"` key becomes `"text"`
- But `"prompt_harm_label"` stays as `"prompt_harm_label"`

## Why run_lpm_experiment.py Didn't Have This Issue

Looking at line 490 of run_lpm_experiment.py:

```python
ground_truth = [item[test_config["category_field"]] for item in test_dataset.iter_items()]
```

It accesses the **category field**, not the text field. Category fields preserve their original names, so `item["prompt_harm_label"]` works fine.

## The Fix

In `verify_dataset_determinism.py`, changed from:
```python
"sample_hashes": [hash_text(item[config["text_field"]]) for item in items],
```

To:
```python
"sample_hashes": [hash_text(item["text"]) for item in items],  # Always use "text" key
```

## Are the Saved Datasets Corrupted?

**No.** The saved datasets are fine. The Arrow files on disk contain all original columns:
- `wgmix_train`: `["prompt", "prompt_harm_label", "subcategory"]`
- `plmix_train`: `["text", "text_harm_label"]`

The normalization only happens when **reading** items via the ClassificationDataset API (`iter_items()`, `__getitem__()`, etc.), not in the stored data.

## Verification

You can verify the saved datasets contain correct columns by running:

```bash
python experiments/scripts/debug/inspect_saved_datasets.py --store /mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store
```

This script loads the raw Arrow files and displays:
- Column names
- Column types
- First 3 rows with all columns

## Design Rationale

The normalization to `"text"` is intentional - it provides a consistent interface:
- Users don't need to remember if it's "prompt", "text", "content", etc.
- They can always access `item["text"]` regardless of the original column name
- Category fields keep original names to allow multiple label columns with distinct meanings

## Recommendation

When working with ClassificationDataset items, always use:
- `item["text"]` for the text content
- `item[original_category_field_name]` for labels (e.g., `item["prompt_harm_label"]`)
