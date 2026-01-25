# Dataset Format Restriction Analysis

## The Puzzle

The inspection script showed:
```
Columns: ['prompt', 'adversarial', 'response', 'prompt_harm_label', ...]  # 7 columns
KeyError: 'adversarial'  # Can't access column 2!
```

**Why can we see column names but not access them?**

## The Answer: HuggingFace Dataset Formatting

### What `set_format()` Does

HuggingFace Datasets has a `set_format()` method that controls:
- **Which columns** are returned when you index the dataset
- **Does NOT modify** the underlying Arrow files
- **Does NOT change** the schema metadata

```python
ds = load_dataset("allenai/wildguardmix", split="train")
# ds.column_names shows: ['prompt', 'adversarial', 'response', 'prompt_harm_label', ...]

ds.set_format("python", columns=["prompt", "prompt_harm_label"])
# ds.column_names STILL shows: ['prompt', 'adversarial', 'response', ...]
# But ds[0].keys() only shows: ['prompt', 'prompt_harm_label']
```

### How ClassificationDataset Uses This

When you create a ClassificationDataset:

```python
dataset = ClassificationDataset.from_huggingface(
    "allenai/wildguardmix",
    text_field="prompt",
    category_field=["prompt_harm_label", "subcategory"],
    ...
)
```

The `__init__` method applies formatting:

```python
# In ClassificationDataset.__init__ (line ~61 in classification_dataset.py)
format_columns = [text_field] + self._category_fields
# format_columns = ["prompt", "prompt_harm_label", "subcategory"]

ds.set_format("python", columns=format_columns)
```

Then when saving (in BaseDataset):
- The **full dataset** with all original columns is saved to Arrow files
- The **format setting is preserved** in the dataset metadata
- When loaded back, only formatted columns are accessible

## Why This Design?

**Memory Efficiency:**
- Large datasets may have dozens of columns
- You only need text + labels for classification
- Format restriction prevents loading unnecessary data into memory

**Preservation:**
- Original data is NOT lost
- Arrow files contain everything
- You can change format later if needed

## What This Means for Our Datasets

### wgmix_train
**Prepared with:**
```python
text_field="prompt"
category_field=["prompt_harm_label", "subcategory"]
```

**Arrow files contain** (7 columns from original dataset):
```
['prompt', 'adversarial', 'response', 'prompt_harm_label', 
 'response_refusal_label', 'response_harm_label', 'subcategory']
```

**But only accessible** (3 columns):
```
['prompt', 'prompt_harm_label', 'subcategory']
```

### plmix_train
**Prepared with:**
```python
text_field="text"
category_field="text_harm_label"
```

**Arrow files contain** (3 columns from CSV):
```
['text', 'text_harm_label', 'text_harm_category']
```

**But only accessible** (2 columns):
```
['text', 'text_harm_label']
```

Note: `text_harm_category` was in the original CSV but not specified in `category_field`, so it's in the Arrow files but not accessible!

## Diagnostic: Updated inspect_saved_datasets.py

The updated script now:
1. Shows both schema columns AND accessible columns
2. Iterates only through accessible columns (no KeyError)
3. Warns when schema ≠ accessible

This helps diagnose:
- What was originally saved (schema)
- What you can actually access (accessible columns)
- Whether format restrictions are applied

## Conclusion

**Your datasets are NOT corrupted.** They contain:
- ✅ Full original data in Arrow files
- ✅ Correct format restrictions for efficient access
- ✅ Only the columns you specified in `text_field` + `category_field`

The format restriction is intentional and beneficial for memory efficiency. If you ever need the other columns, you can:
1. Reset format: `ds.reset_format()`
2. Or re-prepare with different `category_field` values

## Action Items

✅ **No changes needed to your datasets**
✅ **Fixed inspect_saved_datasets.py** to show accessible vs schema columns
✅ **Fixed verify_dataset_determinism.py** to use `"text"` key for normalized access

Run the updated inspection script to see the complete picture:
```bash
python experiments/scripts/debug/inspect_saved_datasets.py --store /mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store
```
