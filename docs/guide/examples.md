# Examples

This guide provides an overview of the example notebooks in the `examples/` directory and a recommended learning path.

## Overview

The examples directory contains Jupyter notebooks demonstrating mi-crow functionality. Each example builds on previous ones, creating a complete learning path from basic usage to advanced techniques.

## Recommended Learning Path

### Beginner Path

1. **01_train_sae_model.ipynb** - Train your first SAE
2. **02_attach_sae_and_save_texts.ipynb** - Discover concepts
3. **03_load_concepts.ipynb** - Manipulate concepts

### Intermediate Path

4. **04_save_inputs_and_outputs.ipynb** - Advanced data collection
5. **08_inference_with_hooks.ipynb** - Direct hook usage
6. **09_activations_two_modes.ipynb** - Activation analysis

### Advanced Path

7. **05_special_token_mask.ipynb** - Special token handling
8. **06_save_activations_with_attention_masks.ipynb** - Attention masks
9. **07_save_activations_and_attention_masks.ipynb** - Combined techniques

## Example Notebooks

### 01_train_sae_model.ipynb

**Purpose**: Train a Sparse Autoencoder (SAE) on model activations

**What you'll learn**:
- Load a language model
- Save activations from a specific layer
- Train an SAE to learn interpretable features
- Save the trained SAE for future use

**Key concepts**:
- Activation saving
- SAE architecture
- Training configuration
- Model persistence

**Output files**:
- `outputs/sae_model.pt`
- `outputs/training_metadata.json`
- `store/activations/<run_id>/`

**Related guides**:
- [Saving Activations](workflows/saving-activations.md)
- [Training SAE Models](workflows/training-sae.md)

---

### 02_attach_sae_and_save_texts.ipynb

**Purpose**: Collect top activating texts for each SAE neuron

**What you'll learn**:
- Load a trained SAE model
- Enable automatic text tracking during inference
- Run inference to collect neuron-text associations
- Export top texts for manual concept curation

**Key concepts**:
- SAE attachment
- Text tracking
- Concept discovery
- Data export

**Output files**:
- `outputs/top_texts.json`
- `outputs/attachment_metadata.json`

**Related guides**:
- [Concept Discovery](workflows/concept-discovery.md)
- [Hooks: Detectors](hooks/detectors.md)

---

### 03_load_concepts.ipynb

**Purpose**: Control model behavior using learned concepts

**What you'll learn**:

#### Part 1: SAE-Level Manipulation
- Load curated concepts (neuron → concept name mappings)
- Use `manipulate_concept()` to amplify/suppress specific SAE neurons
- Compare model behavior with different concept strengths

#### Part 2: Activation Control
- Create custom activation controllers
- Amplify or suppress layer activations during inference
- Enable/disable controllers dynamically
- Use `with_controllers` parameter for A/B testing
- Control model generation in real-time

**Key concepts**:
- Concept manipulation
- Activation controllers
- Dynamic control
- A/B testing

**Output files**:
- Modified model outputs
- Comparison results

**Related guides**:
- [Concept Manipulation](workflows/concept-manipulation.md)
- [Activation Control](workflows/activation-control.md)
- [Hooks: Controllers](hooks/controllers.md)

---

### 04_save_inputs_and_outputs.ipynb

**Purpose**: Save model inputs and outputs alongside activations

**What you'll learn**:
- Capture model inputs (tokenized text)
- Save model outputs (logits, predictions)
- Correlate inputs, activations, and outputs
- Analyze input-output relationships

**Key concepts**:
- Input/output detection
- Data correlation
- Analysis workflows

**Related guides**:
- [Saving Activations](workflows/saving-activations.md)
- [Hooks: Detectors](hooks/detectors.md)

---

### 05_special_token_mask.ipynb

**Purpose**: Handle special tokens when saving activations

**What you'll learn**:
- Identify special tokens
- Mask special tokens from activations
- Filter activations by token type
- Analyze token-specific patterns

**Key concepts**:
- Token masking
- Special token handling
- Activation filtering

**Related guides**:
- [Saving Activations](workflows/saving-activations.md)

---

### 06_save_activations_with_attention_masks.ipynb

**Purpose**: Save activations with proper attention mask handling

**What you'll learn**:
- Use attention masks during activation saving
- Handle padding tokens correctly
- Save masked activations
- Process variable-length sequences

**Key concepts**:
- Attention masks
- Padding handling
- Sequence processing

**Related guides**:
- [Saving Activations](workflows/saving-activations.md)

---

### 07_save_activations_and_attention_masks.ipynb

**Purpose**: Advanced activation saving with attention masks

**What you'll learn**:
- Combined input/output/activation saving
- Attention mask integration
- Complex data collection workflows

**Key concepts**:
- Multi-modal data collection
- Mask integration
- Advanced workflows

**Related guides**:
- [Saving Activations](workflows/saving-activations.md)

---

### 08_inference_with_hooks.ipynb

**Purpose**: Direct hook usage for activation control

**What you'll learn**:
- Create custom detector hooks
- Create custom controller hooks
- Register hooks on layers
- Modify activations directly
- Multi-layer interventions

**Key concepts**:
- Hook creation
- Hook registration
- Activation modification
- Hook management

**Related guides**:
- [Activation Control](workflows/activation-control.md)
- [Hooks: Fundamentals](hooks/fundamentals.md)
- [Hooks: Controllers](hooks/controllers.md)

---

### 09_activations_two_modes.ipynb

**Purpose**: Compare activations in different modes

**What you'll learn**:
- Save activations in training vs evaluation mode
- Compare activation patterns
- Understand mode effects on activations

**Key concepts**:
- Model modes
- Activation comparison
- Mode effects

**Related guides**:
- [Saving Activations](workflows/saving-activations.md)

---

### datasets/load_wildguardmix.ipynb

**Purpose**: Load custom datasets for experiments

**What you'll learn**:
- Load datasets from various sources
- Prepare data for mi-crow
- Handle dataset formats

**Key concepts**:
- Dataset loading
- Data preparation
- Format conversion

**Related guides**:
- [Core Concepts](core-concepts.md) - Datasets section

## Quick Reference Table

| Example | Purpose | Key Concepts | Output Files |
|---------|---------|--------------|--------------|
| 01_train_sae_model | Train SAE | Activation saving, SAE training | `sae_model.pt`, `training_metadata.json` |
| 02_attach_sae_and_save_texts | Concept discovery | Text tracking, concept discovery | `top_texts.json` |
| 03_load_concepts | Concept manipulation | Concept control, activation controllers | Modified outputs |
| 04_save_inputs_and_outputs | Data collection | Input/output detection | Input/output files |
| 05_special_token_mask | Token handling | Token masking, filtering | Masked activations |
| 06_save_activations_with_attention_masks | Mask handling | Attention masks, padding | Masked activations |
| 07_save_activations_and_attention_masks | Advanced collection | Multi-modal collection | Combined data |
| 08_inference_with_hooks | Hook usage | Custom hooks, activation control | Hook examples |
| 09_activations_two_modes | Mode comparison | Training vs eval modes | Comparison data |

## Using Examples

The example notebooks are located in the `examples/` directory. Each notebook demonstrates specific mi-crow functionality and can be opened in Jupyter or JupyterLab.

The examples are designed to be run sequentially, with each building on concepts from previous ones.

## Output Directory Structure

After running examples, you'll have:

```
examples/
├── outputs/
│   ├── sae_model.pt
│   ├── training_metadata.json
│   ├── attachment_metadata.json
│   ├── top_texts.json
│   └── curated_concepts.csv
├── store/
│   ├── activations/
│   │   └── <run_id>/
│   └── runs/
│       └── <training_run_id>/
└── cache/
    └── huggingface/
```

## Tips for Learning

1. **Run in order**: Examples build on each other
2. **Read the code**: Understand what each cell does
3. **Modify parameters**: Experiment with different values
4. **Check outputs**: Verify results match expectations
5. **Read related guides**: Deepen understanding with documentation

## Getting Help

If you encounter issues:

1. Check [Troubleshooting](troubleshooting.md)
2. Review [Best Practices](best-practices.md)
3. Consult [API Reference](../api/index.md)
4. Check example notebook comments

## Next Steps

After working through examples:

- **[Workflows](workflows/index.md)** - Detailed workflow guides
- **[Hooks System](hooks/index.md)** - Deep dive into hooks
- **[Experiments](../experiments/index.md)** - Real-world experiments

