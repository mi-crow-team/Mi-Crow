# Observations: Missing Core Functionality

This document tracks core functionality that is missing from the library and would be useful for SAE training experiments.

## 1. Concept Naming with LLM

**Status**: Not Implemented

**Location**: `src/mi_crow/mechanistic/sae/concepts/concept_dictionary.py`

**Issue**: The `ConceptDictionary._generate_concept_names_llm()` method raises `NotImplementedError`. The method signature exists but requires implementation of LLM provider integration.

**Current Workaround**: 
- Export top texts manually using `sae.concepts.export_top_texts_to_json()`
- Use external tools or manual inspection to name concepts
- Create `ConceptDictionary` manually with `concept_dict.add(index, name, score)`

**Recommended Implementation**:
- Add support for OpenAI API
- Add support for Anthropic API
- Add support for local LLM models
- Make LLM provider configurable via environment variables or config

## 2. Training Visualization Utilities

**Status**: Not Available

**Location**: N/A (would be new module)

**Issue**: No built-in plotting utilities for training metrics. Users must create custom visualizations using matplotlib/seaborn.

**Current Workaround**:
- Manually create plots from training history dictionary
- Use external plotting libraries (matplotlib, seaborn, plotly)

**Recommended Implementation**:
- Add `mi_crow.mechanistic.sae.visualization` module
- Provide functions like `plot_training_history()`, `plot_sparsity_metrics()`, etc.
- Support multiple backends (matplotlib, plotly)

## 3. Layer Selection Assistance

**Status**: Not Available

**Location**: N/A (would be enhancement)

**Issue**: No automatic layer recommendation. Users must manually inspect available layers using `lm.layers.print_layer_names()` and choose appropriate layer.

**Current Workaround**:
- Manually inspect layer names
- Use heuristics (e.g., choose attention or MLP layers)
- Trial and error

**Recommended Implementation**:
- Add `lm.layers.recommend_layers_for_sae()` method
- Suggest layers based on:
  - Layer type (attention vs MLP)
  - Layer depth
  - Activation statistics
- Provide layer comparison utilities

## 4. SAE Weight Inspection Utilities

**Status**: Partially Available

**Location**: `src/mi_crow/mechanistic/sae/modules/topk_sae.py`

**Issue**: Accessing encoder/decoder weights requires accessing internal `sae_engine.state_dict()` which may have different key names depending on the underlying overcomplete implementation.

**Current Workaround**:
- Access `sae.sae_engine.state_dict()` directly
- Parse state dict keys manually
- Handle different naming conventions

**Recommended Implementation**:
- Add `sae.get_encoder_weights()` and `sae.get_decoder_weights()` methods
- Provide weight statistics utilities
- Add weight visualization helpers

## 5. Concept Dictionary Export/Import Formats

**Status**: Available but Limited

**Location**: `src/mi_crow/mechanistic/sae/concepts/concept_dictionary.py`

**Issue**: Concept dictionary supports JSON and CSV formats, but could benefit from more formats and better integration.

**Current Status**:
- ✅ JSON export/import: `ConceptDictionary.save()` and `ConceptDictionary.from_json()`
- ✅ CSV export/import: `ConceptDictionary.from_csv()`
- ❌ No YAML support
- ❌ No integration with concept naming tools

**Recommended Enhancement**:
- Add YAML export/import
- Add concept dictionary validation utilities
- Better integration with concept naming workflows

## 6. Training Configuration Presets

**Status**: Not Available

**Location**: N/A (would be enhancement)

**Issue**: No preset training configurations for common scenarios (small model, large model, quick test, full training, etc.).

**Current Workaround**:
- Manually configure `SaeTrainingConfig` for each experiment
- Copy-paste configurations between experiments

**Recommended Implementation**:
- Add `SaeTrainingConfigPresets` class with common configurations:
  - `QUICK_TEST`: Few epochs, small batch size
  - `STANDARD`: Balanced configuration
  - `FULL_TRAINING`: Many epochs, large batch size
  - `LARGE_MODEL`: Optimized for large models

## 7. Experiment Tracking Integration

**Status**: Partial (WandB only)

**Location**: `src/mi_crow/mechanistic/sae/sae_trainer.py`

**Issue**: Only WandB integration exists. No support for other experiment tracking tools (MLflow, TensorBoard, etc.).

**Current Status**:
- ✅ WandB support via `SaeTrainingConfig.use_wandb`
- ❌ No MLflow support
- ❌ No TensorBoard support
- ❌ No generic experiment tracking interface

**Recommended Implementation**:
- Create abstract `ExperimentTracker` interface
- Implement WandB, MLflow, TensorBoard trackers
- Make tracking backend configurable

## Summary

The library provides solid core functionality for SAE training, but several convenience features and integrations are missing. The most critical gaps are:

1. **Concept naming with LLM** - Required for automated concept discovery
2. **Training visualization utilities** - Would significantly improve user experience
3. **Layer selection assistance** - Would help new users get started faster

These missing features don't prevent SAE training experiments, but they require more manual work and custom code from users.
