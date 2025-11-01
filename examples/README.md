# Amber Examples

This directory contains example notebooks demonstrating the core functionality of the Amber library for interpretable AI research.

## Example Flow

The examples build on each other and should be run in order:

### 01_train_sae_model.ipynb
**Purpose:** Train a Sparse Autoencoder (SAE) on model activations

**What you'll learn:**
- Load a language model
- Save activations from a specific layer
- Train an SAE to learn interpretable features
- Save the trained SAE for future use

**Output:** `outputs/sae_model.pt`, `outputs/training_metadata.json`

---

### 02_attach_sae_and_save_texts.ipynb
**Purpose:** Collect top activating texts for each SAE neuron

**What you'll learn:**
- Load a trained SAE model
- Enable automatic text tracking during inference
- Run inference to collect neuron-text associations
- Export top texts for manual concept curation

**Output:** `outputs/top_texts.json`, `outputs/attachment_metadata.json`

**Key functionality:**
- Text tracking runs automatically during model inference
- Each neuron tracks its top-K most activating text snippets
- Results can be exported for manual concept labeling

---

### 03_load_and_manipulate_concepts.ipynb
**Purpose:** Control model behavior using learned concepts

**What you'll learn:**

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

**Key concepts:**
- **SAE manipulation:** Control specific learned features
- **Activation controllers:** Fine-grained control over any layer
- **Dynamic control:** Toggle interventions on/off
- **Temporary control:** Compare with/without interventions

---

## Quick Start

```bash
# Run all examples in order
jupyter notebook examples/01_train_sae_model.ipynb
jupyter notebook examples/02_attach_sae_and_save_texts.ipynb
jupyter notebook examples/03_load_and_manipulate_concepts.ipynb
```

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- Jupyter

Install dependencies:
```bash
pip install -e .
```

## Output Directory Structure

```
examples/outputs/
├── sae_model.pt                    # Trained SAE model
├── training_metadata.json          # Training configuration
├── attachment_metadata.json        # Attachment configuration
├── top_texts.json                  # Top activating texts per neuron
├── curated_concepts.csv           # Manual concept labels (create this!)
├── store/                         # Saved activations
│   └── activations/
│       └── sae_training_*/
│           ├── batch_*.safetensors
│           └── meta.json
└── cache/                         # HuggingFace cache
```

## Creating Curated Concepts

After running Example 2, manually create `outputs/curated_concepts.csv`:

```csv
neuron_idx,concept_name,score
0,family relationships,0.9
0,parent-child interactions,0.8
1,nature and outdoors,0.9
1,animals and wildlife,0.8
```

This file maps neurons to human-interpretable concept names for use in Example 3.

## Advanced Usage

### Custom Controllers

Create your own controllers by inheriting from `Controller`:

```python
from amber.hooks import Controller, HookType

class MyController(Controller):
    def __init__(self, layer_signature):
        super().__init__(layer_signature, HookType.FORWARD)
    
    def modify_activations(self, module, inputs, output):
        # Your custom modification logic
        return output * 2.0

# Register and use
controller = MyController("layer_name")
hook_id = model.layers.register_hook("layer_name", controller)
```

### Multiple Controllers

You can register multiple controllers on different layers:

```python
# Amplify early layer
early_amplifier = SimpleActivationController("layer_0", scale_factor=2.0)
model.layers.register_hook("layer_0", early_amplifier)

# Suppress late layer
late_suppressor = SimpleActivationController("layer_10", scale_factor=0.5)
model.layers.register_hook("layer_10", late_suppressor)

# All controllers active during inference
output = model.forwards(texts)
```

### A/B Testing

Use `with_controllers` to compare results:

```python
# Get baseline
baseline = model.forwards(prompt, with_controllers=False)

# Get controlled version
controlled = model.forwards(prompt, with_controllers=True)

# Compare
difference = controlled.logits - baseline.logits
```

## Tips

1. **Start small:** Use small models (like `sshleifer/tiny-gpt2`) for quick experimentation
2. **Monitor activations:** Check activation ranges when scaling to avoid instability
3. **Compare results:** Always compare controlled vs uncontrolled inference
4. **Clean up:** Unregister controllers when done to avoid memory leaks

## Support

For issues or questions, please open an issue on the GitHub repository.
