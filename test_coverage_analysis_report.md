# Test Coverage Analysis Report

## Executive Summary

This report analyzes test coverage for core files in the Amber codebase, identifying gaps where tests only verify compilation rather than actual behavior. For each file, we identify:
1. Key use cases and expected behaviors
2. Current test coverage
3. Gaps where tests need to verify actual outputs, state changes, and side effects

---

## 1. LanguageModel (`src/amber/core/language_model.py`)

### Key Methods and Use Cases

#### `__init__`
- **Use Case 1**: Creates default store at `store/{model_id}/` when no store provided
- **Use Case 2**: Extracts model_id from `model.config.name_or_path` (replacing `/` with `_`)
- **Use Case 3**: Falls back to `model.__class__.__name__` if no config.name_or_path
- **Use Case 4**: Initializes all context components (layers, tokenizer, activations)

**Current Test Coverage**: ❌ **GAP** - No tests verify:
- Default store path is `store/{model_id}/`
- Model ID extraction from config vs class name fallback
- Store is actually created at correct location

#### `_inference`
- **Use Case 1**: Calls `InputTracker.set_current_texts()` when enabled (before tokenization)
- **Use Case 2**: Returns `(output, enc)` tuple when `discard_output=False`
- **Use Case 3**: Returns `(input_ids, attn)` tuple when `discard_output=True, save_inputs=True`
- **Use Case 4**: Returns `None` when `discard_output=True, save_inputs=False`
- **Use Case 5**: Temporarily disables controllers when `with_controllers=False` and restores them
- **Use Case 6**: Handles device transfer (CPU/CUDA) correctly
- **Use Case 7**: Applies autocast when enabled and device is CUDA
- **Use Case 8**: Restores controllers even if exception occurs during inference

**Current Test Coverage**: ⚠️ **PARTIAL**
- ✅ `test_inference_invokes_text_trackers_and_forwards_returns_output_and_enc()` verifies text tracker is called
- ✅ `test_tokenizer_sets_pad_from_eos_and_updates_model_config()` verifies `discard_output=True, save_inputs=True` returns correct types
- ❌ **GAP** - No tests verify:
  - `InputTracker.set_current_texts()` is actually called (not just old tracker)
  - Return value types/shapes for `discard_output=True` cases
  - Controllers are properly restored after exception
  - Device handling in `_inference` (tensors moved to correct device)
  - Autocast behavior

#### `forwards`
- **Use Case 1**: Wrapper that calls `_inference` with `discard_output=False`
- **Use Case 2**: Returns `(output, enc)` tuple

**Current Test Coverage**: ✅ **GOOD** - Basic test exists

#### `_ensure_input_tracker`
- **Use Case 1**: Creates InputTracker singleton if doesn't exist
- **Use Case 2**: Returns existing InputTracker if already created
- **Use Case 3**: Logs debug message when creating

**Current Test Coverage**: ❌ **GAP** - No direct tests for singleton behavior

#### `from_huggingface`
- **Use Case 1**: Creates LanguageModel from HuggingFace model name
- **Use Case 2**: Extracts model_id correctly (replacing `/` with `_`)
- **Use Case 3**: Passes tokenizer_params and model_params correctly

**Current Test Coverage**: ❌ **GAP** - No tests verify model_id extraction

#### `register_activation_text_tracker` / `unregister_activation_text_tracker`
- **Use Case 1**: Adds tracker to list (no duplicates)
- **Use Case 2**: Removes tracker (idempotent)
- **Use Case 3**: Tracker is called during `_inference`

**Current Test Coverage**: ✅ **GOOD** - Basic tests exist

### Test Gaps Summary for LanguageModel

**Critical Gaps:**
1. ❌ Default store path verification (`store/{model_id}/`)
2. ❌ Model ID extraction from config vs class name
3. ❌ `_inference` return value types for `discard_output=True` cases
4. ❌ Controller restoration after exception
5. ❌ Device handling in `_inference`
6. ❌ InputTracker singleton creation/reuse
7. ❌ `from_huggingface` model_id extraction

**Recommended Test Additions:**
```python
def test_language_model_default_store_path():
    """Verify default store is created at store/{model_id}/"""
    model = MockModel()
    model.config.name_or_path = "test/model"
    lm = LanguageModel(model, MockTokenizer())
    assert lm.store.base_path == Path.cwd() / "store" / "test_model"

def test_language_model_model_id_from_config():
    """Verify model_id extracted from config.name_or_path"""
    model = MockModel()
    model.config.name_or_path = "huggingface/gpt2"
    lm = LanguageModel(model, MockTokenizer())
    assert lm.model_id == "huggingface_gpt2"

def test_language_model_model_id_fallback():
    """Verify model_id falls back to class name"""
    model = MockModel()
    del model.config.name_or_path
    lm = LanguageModel(model, MockTokenizer())
    assert lm.model_id == "MockModel"

def test_inference_discard_output_save_inputs():
    """Verify _inference returns (input_ids, attn) tuple"""
    lm = LanguageModel(MockModel(), MockTokenizer())
    result = lm._inference(["test"], discard_output=True, save_inputs=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], torch.Tensor)  # input_ids
    assert isinstance(result[1], torch.Tensor)  # attention_mask

def test_inference_discard_output_no_save():
    """Verify _inference returns None"""
    lm = LanguageModel(MockModel(), MockTokenizer())
    result = lm._inference(["test"], discard_output=True, save_inputs=False)
    assert result is None

def test_inference_restores_controllers_after_exception():
    """Verify controllers are restored even if exception occurs"""
    # Create controller that raises exception
    # Verify it's re-enabled after exception

def test_inference_input_tracker_called():
    """Verify InputTracker.set_current_texts is called when enabled"""
    lm = LanguageModel(MockModel(), MockTokenizer())
    tracker = lm._ensure_input_tracker()
    tracker.enable()
    lm._inference(["test"])
    assert tracker.get_current_texts() == ["test"]

def test_ensure_input_tracker_singleton():
    """Verify InputTracker is singleton"""
    lm = LanguageModel(MockModel(), MockTokenizer())
    tracker1 = lm._ensure_input_tracker()
    tracker2 = lm._ensure_input_tracker()
    assert tracker1 is tracker2
```

---

## 2. TopKSae (`src/amber/mechanistic/autoencoder/modules/topk_sae.py`)

### Key Methods and Use Cases

#### `encode`
- **Use Case 1**: Returns sparse TopK activations (k non-zero values per sample)
- **Use Case 2**: Returns shape `[batch, n_latents]`
- **Use Case 3**: Uses `sae_engine.encode()` which returns `(pre_codes, codes)`, returns only `codes`

**Current Test Coverage**: ⚠️ **PARTIAL**
- ✅ `test_topk_sae_modules_encode_decode()` verifies shape
- ❌ **GAP** - No test verifies:
  - Actually returns k non-zero values per sample
  - Sparsity is correct (exactly k non-zero per sample)

#### `decode`
- **Use Case 1**: Reconstructs from latents
- **Use Case 2**: Returns shape `[batch, n_inputs]`

**Current Test Coverage**: ✅ **GOOD** - Shape verified

#### `forward`
- **Use Case 1**: Full forward pass returns reconstruction
- **Use Case 2**: Returns shape `[batch, n_inputs]`
- **Use Case 3**: Uses `sae_engine.forward()` which returns `(pre_codes, codes, x_reconstructed)`

**Current Test Coverage**: ✅ **GOOD** - Shape verified

#### `modify_activations`
- **Use Case 1**: Handles FORWARD hooks (extracts from output)
- **Use Case 2**: Handles PRE_FORWARD hooks (extracts from inputs)
- **Use Case 3**: Extracts tensor from various output types (Tensor, tuple, list, object with last_hidden_state)
- **Use Case 4**: Reshapes 3D inputs `[B, T, D]` to `[B*T, D]` and back
- **Use Case 5**: Uses `pre_codes` (full activations) for text tracking, not sparse `codes`
- **Use Case 6**: Applies concept manipulation (multiplication/bias) when set
- **Use Case 7**: Returns modified activations in same format as input

**Current Test Coverage**: ⚠️ **PARTIAL**
- ✅ `test_topk_sae_modules_modify_activations()` verifies basic functionality
- ✅ `test_topk_sae_modules_modify_activations_with_concepts()` verifies concept manipulation
- ✅ Edge case tests exist for various output types
- ❌ **GAP** - No tests verify:
  - `pre_codes` (not sparse codes) are passed to text tracking
  - Reshaping for 3D inputs is correct
  - Return format matches input format exactly

#### `save`
- **Use Case 1**: Saves `sae_state_dict` under key `"sae_state_dict"`
- **Use Case 2**: Saves `amber_metadata` with all context info
- **Use Case 3**: Creates directory if doesn't exist
- **Use Case 4**: Saves to `{path}/{name}.pt`

**Current Test Coverage**: ✅ **GOOD** - Basic save/load tested

#### `load`
- **Use Case 1**: Loads `sae_state_dict` from payload
- **Use Case 2**: Handles backward compatibility (old format with `"model"` key)
- **Use Case 3**: Restores multiplication/bias parameters
- **Use Case 4**: Restores context metadata (layer_signature, model_id)

**Current Test Coverage**: ✅ **GOOD** - Basic load tested

### Test Gaps Summary for TopKSae

**Critical Gaps:**
1. ❌ `encode` sparsity verification (k non-zero values)
2. ❌ `modify_activations` uses `pre_codes` for text tracking (not sparse codes)
3. ❌ `modify_activations` 3D reshaping correctness
4. ❌ Return format matches input format exactly

**Recommended Test Additions:**
```python
def test_topk_sae_encode_sparsity():
    """Verify encode returns exactly k non-zero values per sample"""
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4)
    x = torch.randn(5, 16)
    encoded = topk_sae.encode(x)
    # Each row should have exactly k non-zero values
    for i in range(5):
        non_zero = (encoded[i] != 0).sum().item()
        assert non_zero == 4

def test_modify_activations_uses_pre_codes_for_text_tracking():
    """Verify modify_activations passes pre_codes (full activations) to text tracking"""
    # Mock concepts.update_top_texts_from_latents
    # Verify it receives pre_codes (not sparse codes)
    # Verify pre_codes have non-zero values where codes are zero

def test_modify_activations_3d_reshaping():
    """Verify 3D inputs are correctly reshaped and restored"""
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4)
    x = torch.randn(2, 3, 16)  # [B, T, D]
    original_shape = x.shape
    modified = topk_sae.modify_activations(None, (), x)
    assert modified.shape == original_shape

def test_modify_activations_return_format_matches_input():
    """Verify return format matches input format (tuple/list/object)"""
    # Test with tuple output
    # Test with list output
    # Test with object output (hasattr last_hidden_state)
```

---

## 3. AutoencoderConcepts (`src/amber/mechanistic/autoencoder/concepts/autoencoder_concepts.py`)

### Key Methods and Use Cases

#### `enable_text_tracking`
- **Use Case 1**: Creates InputTracker singleton via `lm._ensure_input_tracker()`
- **Use Case 2**: Enables InputTracker
- **Use Case 3**: Sets `_text_tracking_enabled` flag on SAE
- **Use Case 4**: Stores k and negative parameters from context

**Current Test Coverage**: ✅ **GOOD** - Basic test exists

#### `update_top_texts_from_latents`
- **Use Case 1**: Tracks only max activation per text (not every token position)
- **Use Case 2**: Handles 2D inputs `[B, D]` (all tokens at index 0)
- **Use Case 3**: Handles 3D inputs `[B, T, D]` (finds max over tokens per text)
- **Use Case 4**: Skips zero scores
- **Use Case 5**: Updates existing text entry if new activation is better
- **Use Case 6**: Respects k limit (heap size)
- **Use Case 7**: Handles negative tracking mode (finds min, negates for heap)

**Current Test Coverage**: ⚠️ **PARTIAL**
- ✅ `test_enable_and_disable_text_tracking_positive_max_over_tokens()` verifies max per text
- ✅ `test_negative_tracking_uses_min_over_tokens_and_asc_sort()` verifies negative mode
- ❌ **GAP** - No tests verify:
  - Existing text entry is updated if better activation
  - Zero scores are skipped
  - k limit is respected (heap doesn't exceed k)

#### `get_top_texts_for_neuron`
- **Use Case 1**: Returns sorted list (descending for positive, ascending for negative)
- **Use Case 2**: Decodes tokens using `_decode_token`
- **Use Case 3**: Returns empty list if no texts or invalid neuron_idx

**Current Test Coverage**: ✅ **GOOD** - Basic tests exist

#### `_decode_token`
- **Use Case 1**: Decodes token from text using tokenizer
- **Use Case 2**: Handles out-of-range token_idx
- **Use Case 3**: Handles decode errors gracefully
- **Use Case 4**: Returns placeholder if tokenizer missing

**Current Test Coverage**: ⚠️ **PARTIAL**
- ✅ Edge case tests exist
- ❌ **GAP** - No test verifies actual token decoding (not just error handling)

#### `export_top_texts_to_json` / `export_top_texts_to_csv`
- **Use Case 1**: Creates valid JSON/CSV files
- **Use Case 2**: Includes all required fields (text, score, token_str, token_idx)
- **Use Case 3**: Creates parent directories if needed
- **Use Case 4**: Raises ValueError if no texts available

**Current Test Coverage**: ⚠️ **PARTIAL**
- ✅ Export tests exist in `test_autoencoder_concepts_export.py`
- ❌ **GAP** - No tests verify:
  - JSON is valid and parseable
  - CSV has correct columns
  - All fields are present

#### `manipulate_concept`
- **Use Case 1**: Updates multiplication parameter
- **Use Case 2**: Updates bias parameter
- **Use Case 3**: Logs warning if dictionary not set

**Current Test Coverage**: ⚠️ **PARTIAL**
- ✅ Edge case tests exist
- ❌ **GAP** - No test verifies actual parameter values are updated

### Test Gaps Summary for AutoencoderConcepts

**Critical Gaps:**
1. ❌ `update_top_texts_from_latents` updates existing entries if better
2. ❌ `update_top_texts_from_latents` skips zero scores
3. ❌ `update_top_texts_from_latents` respects k limit
4. ❌ `_decode_token` actual token decoding (not just error handling)
5. ❌ Export JSON/CSV validity and structure
6. ❌ `manipulate_concept` parameter value updates

**Recommended Test Additions:**
```python
def test_update_top_texts_updates_existing_if_better():
    """Verify existing text entry is updated if new activation is better"""
    # Add text with score 1.0
    # Update with same text but score 2.0
    # Verify only one entry exists with score 2.0

def test_update_top_texts_skips_zero_scores():
    """Verify zero scores are not added to heap"""
    # Update with all-zero activations
    # Verify heap remains empty

def test_update_top_texts_respects_k_limit():
    """Verify heap doesn't exceed k limit"""
    # Add k+1 texts
    # Verify only k entries in heap
    # Verify lowest-scoring entry is removed

def test_decode_token_actual_decoding():
    """Verify _decode_token actually decodes tokens correctly"""
    # Use real tokenizer
    # Verify decoded token matches expected string

def test_export_json_valid_and_parseable():
    """Verify exported JSON is valid and parseable"""
    # Export top texts
    # Load JSON and verify structure
    # Verify all required fields present

def test_export_csv_correct_columns():
    """Verify exported CSV has correct columns"""
    # Export top texts
    # Read CSV and verify columns: neuron_idx, text, score, token_str, token_idx

def test_manipulate_concept_updates_parameters():
    """Verify manipulate_concept actually updates parameter values"""
    # Set multiplication and bias
    # Verify tensor values are updated
    assert concepts.multiplication.data[0] == new_value
    assert concepts.bias.data[0] == new_value
```

---

## 4. SaeTrainer (`src/amber/mechanistic/autoencoder/sae_trainer.py`)

### Key Methods and Use Cases

#### `train`
- **Use Case 1**: Moves model to correct device
- **Use Case 2**: Creates optimizer with correct learning rate
- **Use Case 3**: Creates criterion function matching overcomplete signature
- **Use Case 4**: Uses ReusableStoreDataLoader
- **Use Case 5**: Calls overcomplete `train_sae` or `train_sae_amp` with correct parameters
- **Use Case 6**: Converts logs to history format with keys: `loss`, `recon_mse`, `l1`
- **Use Case 7**: Handles empty store gracefully

**Current Test Coverage**: ⚠️ **PARTIAL**
- ✅ Smoke tests exist
- ✅ Edge case tests exist
- ❌ **GAP** - No tests verify:
  - Model is actually moved to device
  - Optimizer has correct learning rate
  - History has correct keys and structure
  - Criterion function is correct

#### `ReusableStoreDataLoader.__iter__`
- **Use Case 1**: Can be iterated multiple times (reusable)
- **Use Case 2**: Skips non-dict batches
- **Use Case 3**: Skips batches without "activations" key
- **Use Case 4**: Reshapes 3D tensors to 2D
- **Use Case 5**: Handles dtype conversion
- **Use Case 6**: Respects max_batches limit
- **Use Case 7**: Handles empty store gracefully

**Current Test Coverage**: ✅ **GOOD** - Edge case tests exist

### Test Gaps Summary for SaeTrainer

**Critical Gaps:**
1. ❌ `train` moves model to device
2. ❌ `train` creates optimizer with correct learning rate
3. ❌ `train` returns history with correct keys
4. ❌ Criterion function correctness

**Recommended Test Additions:**
```python
def test_train_moves_model_to_device():
    """Verify train moves model to specified device"""
    # Create model on CPU
    # Train with device="cuda" (if available)
    # Verify model is on CUDA

def test_train_optimizer_learning_rate():
    """Verify optimizer has correct learning rate"""
    # Train with lr=0.001
    # Verify optimizer.param_groups[0]['lr'] == 0.001

def test_train_history_structure():
    """Verify train returns history with correct keys"""
    history = trainer.train(...)
    assert "loss" in history
    assert "recon_mse" in history
    assert "l1" in history
    assert isinstance(history["loss"], list)

def test_train_criterion_function():
    """Verify criterion function matches overcomplete signature"""
    # Mock overcomplete train_sae
    # Verify criterion is called with correct arguments
    # Verify criterion returns tensor
```

---

## 5. InputTracker (`src/amber/mechanistic/autoencoder/concepts/input_tracker.py`)

### Key Methods and Use Cases

#### `enable` / `disable`
- **Use Case 1**: Toggles `_enabled` flag
- **Use Case 2**: `set_current_texts` only saves when enabled

**Current Test Coverage**: ✅ **GOOD** - Edge case tests exist

#### `set_current_texts`
- **Use Case 1**: Saves texts only when enabled
- **Use Case 2**: Overwrites previous texts
- **Use Case 3**: Converts sequence to list

**Current Test Coverage**: ✅ **GOOD** - Edge case tests exist

#### `get_current_texts`
- **Use Case 1**: Returns copy of texts (not reference)
- **Use Case 2**: Modifying returned list doesn't affect internal

**Current Test Coverage**: ✅ **GOOD** - Test exists

#### `reset`
- **Use Case 1**: Clears stored texts

**Current Test Coverage**: ✅ **GOOD** - Test exists

### Test Gaps Summary for InputTracker

**Status**: ✅ **GOOD COVERAGE** - All use cases are tested

---

## 6. ConceptDictionary (`src/amber/mechanistic/autoencoder/concepts/concept_dictionary.py`)

### Key Methods and Use Cases

#### `add`
- **Use Case 1**: Replaces existing concept (one per neuron)
- **Use Case 2**: Raises IndexError for out-of-bounds index

**Current Test Coverage**: ✅ **GOOD** - Test exists

#### `get` / `get_many`
- **Use Case 1**: Returns None for missing concepts
- **Use Case 2**: Returns Concept object for existing concepts

**Current Test Coverage**: ✅ **GOOD** - Test exists

#### `save` / `load`
- **Use Case 1**: Creates valid JSON with correct structure
- **Use Case 2**: Handles old format (list) vs new format (dict)
- **Use Case 3**: Roundtrip preserves all concepts

**Current Test Coverage**: ⚠️ **PARTIAL**
- ✅ Basic save/load tested
- ❌ **GAP** - No test verifies:
  - JSON structure is correct
  - Old format (list) is handled correctly

#### `from_csv` / `from_json`
- **Use Case 1**: Takes highest-scoring concept per neuron
- **Use Case 2**: Handles both old (list) and new (dict) formats for JSON

**Current Test Coverage**: ⚠️ **PARTIAL**
- ✅ Basic tests exist
- ❌ **GAP** - No test verifies:
  - Highest score is selected when multiple concepts per neuron
  - Old format (list) is handled correctly

### Test Gaps Summary for ConceptDictionary

**Critical Gaps:**
1. ❌ `save` JSON structure verification
2. ❌ `load` old format (list) handling
3. ❌ `from_csv` highest score selection
4. ❌ `from_json` old format (list) handling

**Recommended Test Additions:**
```python
def test_save_json_structure():
    """Verify save creates correct JSON structure"""
    cd = ConceptDictionary(n_size=3)
    cd.add(0, "concept1", 0.5)
    cd.add(1, "concept2", 0.8)
    path = cd.save(directory=tmp_path)
    # Load JSON and verify structure
    with open(path) as f:
        data = json.load(f)
    assert "n_size" in data
    assert "concepts" in data
    assert data["concepts"]["0"]["name"] == "concept1"
    assert data["concepts"]["0"]["score"] == 0.5

def test_load_old_format_list():
    """Verify load handles old format (list of concepts)"""
    # Create JSON with old format: {"0": [{"name": "c1", "score": 0.5}, {"name": "c2", "score": 0.8}]}
    # Load and verify only highest-scoring concept is kept

def test_from_csv_highest_score():
    """Verify from_csv selects highest-scoring concept per neuron"""
    # Create CSV with multiple concepts per neuron
    # Verify only highest-scoring is kept

def test_from_json_old_format_list():
    """Verify from_json handles old format (list)"""
    # Create JSON with old format
    # Verify only highest-scoring concept is kept
```

---

## 7. Store / LocalStore (`src/amber/store.py`)

### Key Methods and Use Cases

#### `put_run_batch` / `get_run_batch`
- **Use Case 1**: Saves dict as-is
- **Use Case 2**: Converts list to dict with `item_N` keys
- **Use Case 3**: Reconstructs list from `item_N` keys
- **Use Case 4**: Returns dict if keys don't match `item_N` pattern

**Current Test Coverage**: ⚠️ **PARTIAL**
- ✅ Basic tests exist in activation save tests
- ❌ **GAP** - No dedicated tests verify:
  - List conversion to `item_N` keys
  - List reconstruction from `item_N` keys
  - Dict preservation

#### `list_run_batches`
- **Use Case 1**: Returns sorted list of batch indices
- **Use Case 2**: Returns empty list if no batches

**Current Test Coverage**: ✅ **GOOD** - Test exists

#### `iter_run_batch_range`
- **Use Case 1**: Iterates over range with step
- **Use Case 2**: Skips missing batches (FileNotFoundError)
- **Use Case 3**: Raises ValueError for invalid step/start

**Current Test Coverage**: ❌ **GAP** - No tests exist

#### `put_run_meta` / `get_run_meta`
- **Use Case 1**: Creates valid JSON
- **Use Case 2**: Roundtrip preserves metadata
- **Use Case 3**: Returns empty dict if file missing

**Current Test Coverage**: ⚠️ **PARTIAL**
- ✅ Basic tests exist
- ❌ **GAP** - No test verifies JSON validity

#### `delete_run`
- **Use Case 1**: Removes all batch files
- **Use Case 2**: Handles non-existent run gracefully

**Current Test Coverage**: ❌ **GAP** - No tests exist

#### `put_tensor` / `get_tensor`
- **Use Case 1**: Roundtrip preserves tensor values and shape

**Current Test Coverage**: ❌ **GAP** - No tests exist

### Test Gaps Summary for Store/LocalStore

**Critical Gaps:**
1. ❌ `put_run_batch` / `get_run_batch` list conversion/reconstruction
2. ❌ `iter_run_batch_range` behavior
3. ❌ `put_run_meta` / `get_run_meta` JSON validity
4. ❌ `delete_run` file removal
5. ❌ `put_tensor` / `get_tensor` roundtrip

**Recommended Test Additions:**

```python
def test_put_get_run_batch_list_conversion():
    """Verify list is converted to item_N keys and reconstructed"""
    store = LocalStore(tmp_path)
    tensors = [torch.randn(5, 8), torch.randn(3, 8)]
    store.put_run_batch("run1", 0, tensors)
    loaded = store.get_run_batch("run1", 0)
    assert isinstance(loaded, list)
    assert len(loaded) == 2
    assert torch.allclose(loaded[0], tensors[0])


def test_put_get_run_batch_dict_preservation():
    """Verify dict is preserved as-is"""
    store = LocalStore(tmp_path)
    batch = {"activations": torch.randn(5, 8), "labels": torch.randint(0, 10, (5,))}
    store.put_run_batch("run1", 0, batch)
    loaded = store.get_run_batch("run1", 0)
    assert isinstance(loaded, dict)
    assert set(loaded.keys()) == {"activations", "labels"}


def test_iter_run_batch_range():
    """Verify iter_run_batch_range behavior"""
    store = LocalStore(tmp_path)
    # Create batches 0, 1, 2, 3
    # Iterate with start=0, stop=4, step=2
    # Verify batches 0, 2 are returned
    # Test with missing batch (skip FileNotFoundError)


def test_put_get_run_meta_json_validity():
    """Verify put_run_meta creates valid JSON"""
    store = LocalStore(tmp_path)
    meta = {"run_name": "test", "epochs": 10}
    store.put_run_metadata("run1", meta)
    loaded = store.get_run_metadata("run1")
    assert loaded == meta
    # Verify JSON file is valid and parseable


def test_delete_run_removes_files():
    """Verify delete_run removes all batch files"""
    store = LocalStore(tmp_path)
    # Create batches
    # Delete run
    # Verify files are removed
    assert len(store.list_run_batches("run1")) == 0


def test_put_get_tensor_roundtrip():
    """Verify put_tensor/get_tensor roundtrip"""
    store = LocalStore(tmp_path)
    tensor = torch.randn(5, 8)
    store.put_tensor("test_tensor", tensor)
    loaded = store.get_tensor("test_tensor")
    assert torch.allclose(loaded, tensor)
    assert loaded.shape == tensor.shape
```

---

## Priority Ranking

### High Priority (Critical Functionality)
1. **LanguageModel**: Default store path, model_id extraction, `_inference` return values
2. **TopKSae**: `encode` sparsity, `modify_activations` uses `pre_codes` for text tracking
3. **AutoencoderConcepts**: `update_top_texts_from_latents` updates existing entries, respects k limit
4. **Store/LocalStore**: `put_run_batch`/`get_run_batch` list conversion, `delete_run`

### Medium Priority (Important but Less Critical)
1. **LanguageModel**: Controller restoration, device handling, InputTracker singleton
2. **TopKSae**: 3D reshaping, return format matching
3. **AutoencoderConcepts**: Export JSON/CSV validity, `manipulate_concept` parameter updates
4. **SaeTrainer**: Device movement, optimizer learning rate, history structure
5. **ConceptDictionary**: JSON structure, old format handling

### Low Priority (Edge Cases)
1. **LanguageModel**: Autocast behavior
2. **AutoencoderConcepts**: `_decode_token` actual decoding (not just error handling)
3. **Store/LocalStore**: `iter_run_batch_range`, `put_tensor`/`get_tensor`

---

## Summary Statistics

- **Total Files Analyzed**: 7
- **Files with Good Coverage**: 1 (InputTracker)
- **Files with Partial Coverage**: 5 (LanguageModel, TopKSae, AutoencoderConcepts, SaeTrainer, ConceptDictionary)
- **Files with Poor Coverage**: 1 (Store/LocalStore)
- **Total Critical Gaps Identified**: 25
- **Total Recommended Test Additions**: ~30 tests

---

## Next Steps

1. **Create test files** for each component with the recommended test additions
2. **Prioritize** based on the ranking above
3. **Run tests** and verify they catch actual bugs (not just compilation errors)
4. **Update coverage report** after adding tests
5. **Consider** adding property-based tests (using Hypothesis) for complex behaviors

