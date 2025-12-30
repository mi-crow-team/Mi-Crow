# SAE Training Experiment

This experiment demonstrates a complete workflow for training a Sparse Autoencoder (SAE) on activations from the Bielik model using the TinyStories dataset.

## Structure

```
verify_sae_training/
├── 01_save_activations.py      # Step 1: Save activations from dataset
├── 02_train_sae.py              # Step 2: Train SAE model
├── 03_analyze_training.ipynb   # Step 3: Analyze training metrics and verify learning
├── 04_name_sae_concepts.ipynb  # Step 4: Export top texts for each neuron
├── 05_show_concepts.ipynb       # Step 5: Display and explore concepts
├── observations.md          # Missing core functionality documentation
└── README.md                # This file
```

## Prerequisites

- Python 3.8+
- PyTorch
- Required packages: `mi_crow`, `torch`, `transformers`, `datasets`, `overcomplete`, `matplotlib`, `seaborn`

## Usage

### Step 1: Save Activations

```bash
cd experiments/verify_sae_training
python 01_save_activations.py
```

**Layer configuration**: The script uses the `resid_mid` layer (post_attention_layernorm) - the residual stream after attention and before MLP. It's hardcoded to use layer 16 (middle layer for Bielik 1.5B model with 32 layers).

**Layer name**: `llamaforcausallm_model_layers_16_post_attention_layernorm`

**To change the layer**: Edit `LAYER_SIGNATURE` in `01_save_activations.py` (e.g., use `_0_` for first layer, `_31_` for last layer).

This script will:
- Load the Bielik model
- Use resid_mid layer (post_attention_layernorm) at layer 12
- Load TinyStories dataset
- Save activations from the specified layer
- Store run ID in `store/run_id.txt`

### Step 2: Train SAE

```bash
python 02_train_sae.py
```

This script will:
- Load the saved activations
- Create a TopKSAE model
- Train the SAE on the activations
- Save the trained model to `store/sae_model/topk_sae.pt`
- Save training history to `store/training_history.json`

**Configuration**: Edit the script to adjust:
- `N_LATENTS_MULTIPLIER`: Overcompleteness factor (default: 4x)
- `TOP_K`: Sparsity parameter (default: 8)
- `EPOCHS`: Number of training epochs (default: 100)
- `BATCH_SIZE_TRAIN`: Training batch size (default: 1024)

### Step 3: Analyze Training

Open `03_analyze_training.ipynb` in Jupyter and run all cells.

This notebook will:
- Visualize training metrics (loss, R², L0, dead features)
- Verify sparsity (check actual L0 vs expected TopK)
- Verify learning (check weight variance, ensure weights are not uniform)
- Analyze reconstruction quality

### Step 4: Export Top Texts

Open `04_name_sae_concepts.ipynb` in Jupyter and run all cells.

This notebook will:
- Load the trained SAE
- Attach it to the language model
- Enable text tracking
- Run inference on dataset to collect top texts
- Export top texts to `store/top_texts.json`

### Step 5: Show Concepts

Open `05_show_concepts.ipynb` in Jupyter and run all cells.

This notebook will:
- Load exported top texts
- Display most active neurons
- Show concepts for specific neurons
- Analyze token patterns
- Provide interactive exploration

## Configuration

All scripts use the following default configuration:

- **Model**: `speakleash/Bielik-1.5B-v3.0-Instruct`
- **Dataset**: `roneneldan/TinyStories` (train split)
- **Store location**: `experiments/verify_sae_training/store/`
- **Device**: Auto-detected (CUDA if available, else CPU)

## Output Files

After running all steps, you'll have:

- `store/run_id.txt` - Run ID for the activation saving run
- `store/runs/<run_id>/` - Saved activations
- `store/sae_model/topk_sae.pt` - Trained SAE model
- `store/training_history.json` - Training metrics
- `store/top_texts.json` - Exported top texts for each neuron

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
- See `observations.md` for missing core functionality that would improve the workflow
