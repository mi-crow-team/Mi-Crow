# Verify SAE Training Experiment

This experiment demonstrates a complete workflow for training and validating SAE models on the Bielik model using the TinyStories dataset.

## Overview

This experiment walks through:
1. Saving activations from a production model
2. Training SAEs with proper hyperparameters
3. Validating training success
4. Concept discovery and naming
5. Analysis and visualization

## Prerequisites

- Python 3.8+
- PyTorch
- Required packages: `mi_crow`, `torch`, `transformers`, `datasets`, `overcomplete`, `matplotlib`, `seaborn`
- Access to Bielik model or similar
- Sufficient GPU memory (or use CPU for smaller experiments)

## Experiment Structure

```
verify_sae_training/
├── 01_save_activations.py      # Step 1: Save activations from dataset
├── 02_train_sae.py              # Step 2: Train SAE model
├── 03_analyze_training.ipynb   # Step 3: Analyze training metrics and verify learning
├── 04_name_sae_concepts.ipynb  # Step 4: Export top texts for each neuron
├── 05_show_concepts.ipynb       # Step 5: Display and explore concepts
├── observations.md              # Findings and observations
└── README.md                    # Experiment documentation
```

## Step-by-Step Instructions

### Step 1: Save Activations

**File**: `01_save_activations.py`

This script:
- Loads the Bielik model
- Uses resid_mid layer (post_attention_layernorm) at layer 16
- Loads TinyStories dataset
- Saves activations from the specified layer
- Stores run ID in `store/run_id.txt`

**Configuration**:
- **Model**: `speakleash/Bielik-1.5B-v3.0-Instruct`
- **Dataset**: `roneneldan/TinyStories` (train split)
- **Layer**: `llamaforcausallm_model_layers_16_post_attention_layernorm`
- **Store location**: `experiments/verify_sae_training/store/`

**To change the layer**: Edit `LAYER_SIGNATURE` in the script (e.g., use `_0_` for first layer, `_31_` for last layer).

**Run**:
```bash
cd experiments/verify_sae_training
python 01_save_activations.py
```

### Step 2: Train SAE

**File**: `02_train_sae.py`

This script:
- Loads the saved activations
- Creates a TopKSAE model
- Trains the SAE on the activations
- Saves the trained model to `store/sae_model/topk_sae.pt`
- Saves training history to `store/training_history.json`

**Configuration**:
- `N_LATENTS_MULTIPLIER`: Overcompleteness factor (default: 4x)
- `TOP_K`: Sparsity parameter (default: 8)
- `EPOCHS`: Number of training epochs (default: 100)
- `BATCH_SIZE_TRAIN`: Training batch size (default: 1024)

**Run**:
```bash
python 02_train_sae.py
```

### Step 3: Analyze Training

**File**: `03_analyze_training.ipynb`

This notebook demonstrates:
- Accessing training history from `store/training_history.json`
- Visualizing training metrics (loss, R², L0, dead features)
- Validating SAE training success using mi-crow APIs
- Analyzing reconstruction quality

**Key validation checks**:
- Loss should decrease over time
- R² should increase (better reconstruction)
- L0 should match expected TopK sparsity
- Dead features should be minimal
- Weight variance should be significant

### Step 4: Export Top Texts

**File**: `04_name_sae_concepts.ipynb`

This notebook demonstrates:
- Loading trained SAE using mi-crow APIs
- Attaching SAE to language model with `lm.attach_sae()`
- Enabling text tracking with `sae.concepts.enable_text_tracking()`
- Collecting top texts using mi-crow's concept discovery features
- Exporting results to JSON format

**Output**: JSON file mapping neuron indices to top activating text snippets.

### Step 5: Show Concepts

**File**: `05_show_concepts.ipynb`

This notebook demonstrates:
- Loading exported top texts
- Using mi-crow APIs to access concept data
- Analyzing neuron activation patterns
- Exploring concept relationships

## Expected Outputs

After running all steps, you'll have:

- `store/run_id.txt` - Run ID for the activation saving run
- `store/runs/<run_id>/` - Saved activations
- `store/sae_model/topk_sae.pt` - Trained SAE model
- `store/training_history.json` - Training metrics
- `store/top_texts.json` - Exported top texts for each neuron

## Analysis and Validation

### Training Metrics

Check that:
- Loss decreases over epochs
- R² score improves (target: > 0.9)
- L0 matches expected TopK
- Dead features < 10% of total

### Concept Quality

Verify that:
- Top texts show coherent patterns
- Neurons detect meaningful concepts
- Concepts are interpretable
- Multiple neurons may detect similar concepts (redundancy is normal)

## Troubleshooting

### Layer Signature Not Found

If you get an error about layer signature:
1. Run `01_save_activations.py` first to see available layers
2. Copy one of the layer names
3. Set `LAYER_SIGNATURE` in `01_save_activations.py`

### Out of Memory

If you run out of memory:
- Reduce `DATA_LIMIT` in `01_save_activations.py`
- Reduce `BATCH_SIZE_SAVE` in `01_save_activations.py`
- Reduce `BATCH_SIZE_TRAIN` in `02_train_sae.py`
- Use CPU instead of GPU (set `DEVICE = "cpu"`)

### Missing Dependencies

Install missing packages:
```bash
pip install torch transformers datasets overcomplete matplotlib seaborn
```

## Notes

- The experiment uses a relatively small dataset (1000 samples) for quick testing
- For production use, increase `DATA_LIMIT` and training epochs
- See `observations.md` for findings and missing functionality notes

## Related Documentation

- **[Training SAE Models](../guide/workflows/training-sae.md)** - Detailed training guide
- **[Concept Discovery](../guide/workflows/concept-discovery.md)** - Concept discovery workflow
- **[Best Practices](../guide/best-practices.md)** - General best practices
- **[Troubleshooting](../guide/troubleshooting.md)** - Common issues

## Next Steps

After completing this experiment:
- Try different layers
- Experiment with hyperparameters
- Scale up to larger datasets
- Create your own experiments

