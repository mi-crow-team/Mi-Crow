# End-to-End Tests

This directory contains end-to-end tests that validate complete workflows in the Amber library. These tests are based on the example notebooks and demonstrate real-world usage patterns.

## Test Suite

### 1. `test_e2e_train_sae.py`

**Based on:** `examples/01_train_sae_model.ipynb`

**Tests the complete SAE training workflow:**
- ✅ Load a language model
- ✅ Create a text dataset
- ✅ Save activations from a specific layer
- ✅ Train a Sparse Autoencoder (SAE)
- ✅ Save the trained SAE model
- ✅ Load the SAE and verify reconstruction

**Key validation:**
- Model loads correctly
- Activations are saved to store
- SAE trains without errors
- Model can be saved and reloaded
- Reconstruction produces correct output shapes

---

### 2. `test_e2e_attach_sae_and_track_texts.py`

**Based on:** `examples/02_attach_sae_and_save_texts.ipynb`

**Tests SAE attachment and text tracking:**
- ✅ Load a trained SAE model
- ✅ Attach SAE to language model
- ✅ Enable automatic text tracking
- ✅ Run inference to collect top activating texts
- ✅ Export and validate collected texts

**Key validation:**
- SAE loads and attaches correctly
- Text tracking captures neuron-text associations
- Top texts are sorted by activation score
- Export produces valid JSON

---

### 3. `test_e2e_activation_control.py`

**Based on:** `examples/03_load_and_manipulate_concepts.ipynb`

**Tests activation control with custom controllers:**

#### Test 1: Activation Amplification
- ✅ Create and register a scaling controller
- ✅ Run inference with controller enabled
- ✅ Disable controller and verify normal operation
- ✅ Re-enable and verify activation
- ✅ Cleanup and unregister

#### Test 2: `with_controllers` Parameter
- ✅ Register a capturing controller
- ✅ Run inference with `with_controllers=False` (temporary disable)
- ✅ Verify controller didn't capture during temporary disable
- ✅ Run with `with_controllers=True`
- ✅ Verify controller captured and modified activations
- ✅ Verify modifications match expected values (2x amplification)

#### Test 3: Multiple Controllers
- ✅ Register controllers on multiple layers
- ✅ Run inference with all controllers active
- ✅ Disable specific controllers
- ✅ Verify selective control works
- ✅ Cleanup all controllers

**Key validation:**
- Controllers modify activations correctly
- Enable/disable functionality works
- Multiple controllers can coexist
- Temporary disable preserves controller state
- Modifications are applied correctly (verified numerically)

---

## Running the Tests

### Run all e2e tests:
```bash
pytest tests/e2e/ -v
```

### Run a specific test:
```bash
pytest tests/e2e/test_e2e_train_sae.py -v
```

### Run with detailed output:
```bash
pytest tests/e2e/ -v -s
```

## Test Characteristics

- **Isolated:** Each test uses temporary directories and cleans up after itself
- **Realistic:** Tests use actual models (sshleifer/tiny-gpt2) and real workflows
- **Fast:** Uses small datasets and quick training configs for rapid validation
- **Comprehensive:** Covers the three main workflows: training, tracking, and control

## Requirements

These tests require:
- PyTorch
- Transformers (HuggingFace)
- Datasets library
- Network access (to download models on first run)

## Coverage

These e2e tests focus on workflow validation rather than line coverage. They ensure that:
1. Components work together correctly
2. User-facing APIs function as documented
3. Real-world usage patterns succeed
4. Edge cases in integration are handled

For detailed unit test coverage, see `tests/unit/`.

## Maintenance

When updating the example notebooks:
1. Review corresponding e2e tests
2. Update tests to match new workflows
3. Ensure tests still validate core functionality
4. Add new tests for new features

## Debugging Failed Tests

If a test fails:

1. **Check logs:** Run with `-s` flag to see detailed output
2. **Verify models:** Ensure sshleifer/tiny-gpt2 is accessible
3. **Check temp dirs:** Tests create temporary directories that are cleaned up
4. **Isolate:** Run individual tests to identify the specific failure

Common issues:
- Network connectivity (model downloads)
- Disk space (temporary files)
- PyTorch version compatibility

## Test Timing

Approximate run times on CPU:
- `test_e2e_train_sae.py`: ~3-4 seconds
- `test_e2e_attach_sae_and_track_texts.py`: ~3-4 seconds  
- `test_e2e_activation_control.py`: ~2-3 seconds (3 tests)

Total: ~10-12 seconds for all tests

