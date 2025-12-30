# SLURM SAE Pipeline for Bielik 1.5B Instruct

This directory contains scripts for running the full SAE (Sparse Autoencoder) training pipeline on SLURM clusters using the Bielik 1.5B Instruct model.

## Structure

```
slurm_sae_pipeline/
├── 01_save_activations.py    # Step 1: Save activations from dataset
├── 02_train_sae.py            # Step 2: Train SAE model
├── discover_layers.py         # Helper: Discover model layers and suggest layer number
├── submit_save_activations.sh # Example SLURM script for step 1
├── submit_train_sae.sh        # Example SLURM script for step 2
└── README.md                  # This file
```

## Prerequisites

- Python 3.8+
- PyTorch with CUDA support (for GPU clusters)
- Required packages: `mi_crow`, `torch`, `transformers`, `datasets`
- Access to SLURM cluster with GPU nodes
- **HuggingFace Access**: The Bielik 1.5B-v3.0-Instruct model is publicly available. You may need to authenticate with HuggingFace: `huggingface-cli login` or set `HF_TOKEN` environment variable

## Configuration

### Model
- **Model**: `speakleash/Bielik-1.5B-v3.0-Instruct`
- **Layer**: Defaults to layer 16 (middle layer for 32-layer model), configurable via `LAYER_NUM` environment variable

### Layer Selection for SAE Training

#### Recommended Layer Type: `post_attention_layernorm` (resid_mid)

The scripts extract activations from the **`post_attention_layernorm`** layer, also known as **`resid_mid`** - the residual stream after the attention mechanism and before the MLP (Multi-Layer Perceptron).

**Why this layer type?**

1. **Feature Richness**: This position captures representations that have been processed by attention (capturing token relationships and context) but not yet transformed by the MLP. This provides a good balance of contextual and semantic information.

2. **Sparsity Properties**: Research shows that activations at this position often exhibit natural sparsity, making them ideal for sparse autoencoder training. The SAE can learn to identify and reconstruct these sparse patterns effectively.

3. **Interpretability**: Features learned from `resid_mid` activations tend to be more interpretable, as they represent intermediate semantic concepts that are neither too low-level (like early layers) nor too task-specific (like late layers).

4. **Proven Practice**: This layer type is widely used in mechanistic interpretability research (e.g., Anthropic's SAE work) and has been validated in our own experiments with Bielik 1.5B.

**Layer Format**: `llamaforcausallm_model_layers_{LAYER_NUM}_post_attention_layernorm`

#### Recommended Layer Position: Middle Layers

The default layer number is **16**, which is the middle layer for Bielik 1.5B's 32 transformer layers. This represents a middle layer position.

**Why middle layers?**

1. **Balanced Representation**: 
   - **Early layers** (0-8): Capture low-level features like token patterns, syntax, and local dependencies
   - **Middle layers** (12-24): Balance between general linguistic patterns and task-specific features
   - **Late layers** (28+): Focus on high-level, task-specific representations

2. **Optimal for Interpretability**: Middle layers provide the best trade-off between:
   - General linguistic knowledge (useful across tasks)
   - Specific semantic concepts (interpretable features)
   - Feature diversity (not too specialized)

3. **Empirical Evidence**: Our experiments with Bielik 1.5B (32 layers) use layer 16 (middle), which produces interpretable features.

**How to determine the correct layer number:**

1. **Check model configuration**: The number of layers can be found in the model's config:
   ```python
   from transformers import AutoConfig
   config = AutoConfig.from_pretrained("speakleash/Bielik-1.5B-v3.0-Instruct")
   num_layers = config.num_hidden_layers  # or config.num_layers
   middle_layer = num_layers // 2
   ```

2. **Use the discovery script**: Run `python discover_layers.py` to automatically discover the model's layer count and get a recommended layer number. This script will:
   - Check the model configuration for layer count
   - Load the model and inspect available layers
   - List all `post_attention_layernorm` layers
   - Suggest the optimal middle layer number

3. **Discover available layers manually**: Run the activation script once and it will print available layers if the specified layer is not found. Look for layers matching the pattern `*_layers_{N}_post_attention_layernorm`.

3. **Common configurations**:
   - **32 layers** (Bielik 1.5B): Use layer **16** (default)
   - **40 layers**: Use layer **20**
   - **48 layers**: Use layer **24**
   - **56 layers**: Use layer **28**

**Alternative layer positions:**

- **Early layers** (e.g., layer 8-12): Better for capturing syntactic patterns and basic linguistic structures
- **Late layers** (e.g., layer 32+): Better for task-specific features and high-level semantics
- **Multiple layers**: You can train separate SAEs on different layers and compare results

### Environment Variables

Both scripts support the following environment variables:

#### Common Variables
- `MODEL_ID`: HuggingFace model ID (default: `speakleash/Bielik-1.5B-v3.0-Instruct`)
- `STORE_DIR`: Directory to store activations and models (default: `./store` or `$SCRATCH` if available)
- `DEVICE`: Device to use (`cuda` or `cpu`, default: auto-detect)

#### Activation Saving (01_save_activations.py)
- `HF_DATASET`: HuggingFace dataset name (default: `roneneldan/TinyStories`)
- `DATA_SPLIT`: Dataset split to use (default: `train`)
- `TEXT_FIELD`: Field name containing text in dataset (default: `text`)
- `DATA_LIMIT`: Number of samples to process (default: `10000`)
- `MAX_LENGTH`: Maximum sequence length (default: `128`)
- `BATCH_SIZE_SAVE`: Batch size for saving activations (default: `16`)
- `LAYER_NUM`: Layer number to extract activations from (default: `16`)

#### SAE Training (02_train_sae.py)
- `N_LATENTS_MULTIPLIER`: Overcompleteness factor (default: `4`)
- `TOP_K`: Sparsity parameter (default: `8`)
- `EPOCHS`: Number of training epochs (default: `10`)
- `BATCH_SIZE_TRAIN`: Training batch size (default: `32`)
- `LR`: Learning rate (default: `1e-3`)
- `L1_LAMBDA`: L1 regularization strength (default: `1e-4`)

## Usage

### Step 1: Save Activations

```bash
# Basic usage
uv run python 01_save_activations.py

# With custom configuration
DATA_LIMIT=50000 \
BATCH_SIZE_SAVE=32 \
LAYER_NUM=20 \
STORE_DIR=/scratch/user/sae_store \
uv run python 01_save_activations.py

# With custom run_id
uv run python 01_save_activations.py --run_id my_custom_run_id
```

This script will:
- Load the Bielik 1.5B Instruct model
- Load the specified dataset from HuggingFace
- Extract activations from the specified layer
- Save activations to the store directory
- Save run ID to `store/run_id.txt` for use in training

### Step 2: Train SAE

```bash
# Basic usage (uses run_id.txt from step 1)
uv run python 02_train_sae.py

# With custom configuration
EPOCHS=20 \
BATCH_SIZE_TRAIN=64 \
N_LATENTS_MULTIPLIER=8 \
STORE_DIR=/scratch/user/sae_store \
uv run python 02_train_sae.py

# With custom run_id
uv run python 02_train_sae.py --run_id my_custom_run_id
```

This script will:
- Load the saved activations from step 1
- Create a TopKSAE model
- Train the SAE on the activations
- Save the trained model and training history

## SLURM Job Scripts

### Example SLURM script for saving activations

```bash
#!/bin/bash
#SBATCH --job-name=sae_save_activations
#SBATCH --output=sae_save_activations_%j.out
#SBATCH --error=sae_save_activations_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Load modules (adjust for your cluster)
module load python/3.10
module load cuda/11.8

# Activate virtual environment
source /path/to/venv/bin/activate

# Set environment variables
export STORE_DIR=$SCRATCH/sae_store
export DATA_LIMIT=100000
export BATCH_SIZE_SAVE=32
export LAYER_NUM=16
export DEVICE=cuda

# Run script
cd /path/to/experiments/slurm_sae_pipeline
uv run python 01_save_activations.py
```

### Example SLURM script for training SAE

```bash
#!/bin/bash
#SBATCH --job-name=sae_train
#SBATCH --output=sae_train_%j.out
#SBATCH --error=sae_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Load modules (adjust for your cluster)
module load python/3.10
module load cuda/11.8

# Activate virtual environment
source /path/to/venv/bin/activate

# Set environment variables
export STORE_DIR=$SCRATCH/sae_store
export EPOCHS=20
export BATCH_SIZE_TRAIN=64
export N_LATENTS_MULTIPLIER=8
export TOP_K=8
export DEVICE=cuda

# Run script
cd /path/to/experiments/slurm_sae_pipeline
uv run python 02_train_sae.py
```

## Output Files

After running both scripts, you'll have:

- `store/run_id.txt` - Run ID for the activation saving run
- `store/runs/<run_id>/` - Saved activations (batches)
- `store/runs/<training_run_id>/model.pt` - Trained SAE model
- `store/runs/<training_run_id>/history.json` - Training metrics
- `store/runs/<training_run_id>/meta.json` - Training metadata

## Troubleshooting

### Layer Signature Not Found

If you get an error about layer signature:
1. Check the model architecture - Bielik 4.5B may have a different number of layers
2. Adjust `LAYER_NUM` environment variable
3. The script will print available layers if the specified layer is not found

### Out of Memory

If you run out of memory:
- Reduce `DATA_LIMIT` in activation saving script
- Reduce `BATCH_SIZE_SAVE` in activation saving script
- Reduce `BATCH_SIZE_TRAIN` in training script
- Use gradient accumulation if needed
- Request more GPU memory in SLURM job

### CUDA Errors

- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check CUDA version compatibility
- Verify GPU allocation in SLURM: `nvidia-smi`

### Missing Dependencies

Install required packages:
```bash
pip install torch transformers datasets mi_crow
```

### Model Access

The Bielik 1.5B-v3.0-Instruct model is publicly available on HuggingFace. If you encounter authentication errors, you may need to authenticate:

1. **Authenticate with HuggingFace** (if needed):
   ```bash
   huggingface-cli login
   ```
   Or set the token as an environment variable:
   ```bash
   export HF_TOKEN="your_token_here"
   ```
2. **Verify access**: Run `uv run python discover_layers.py` to verify you can access the model

## Notes

- **Layer Selection**: The default layer (16) is the middle layer for Bielik 1.5B's 32 layers. If using a different model, adjust `LAYER_NUM` to approximately half the total layer count. See the "Layer Selection for SAE Training" section above for detailed guidance.
- For production use, increase `DATA_LIMIT` and training epochs
- Monitor GPU memory usage and adjust batch sizes accordingly
- Consider using mixed precision training (already enabled for CUDA devices)
- The `post_attention_layernorm` layer type is recommended based on research and empirical results, but you can experiment with other layer types if needed

