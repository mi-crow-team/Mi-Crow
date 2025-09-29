from __future__ import annotations

# Converted from playground/train_sae.ipynb into a runnable Python script.
# Run: python playground/train_sae.py

from pathlib import Path
from datetime import datetime
import os

import torch

# Silence Hugging Face tokenizers fork/parallelism warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from amber.store import LocalStore
from amber.adapters.text_snippet_dataset import TextSnippetDataset
from amber.core.language_model import LanguageModel
from amber.mechanistic.autoencoder.autoencoder import Autoencoder
from amber.mechanistic.autoencoder.train import SAETrainer, SAETrainingConfig
from amber.utils import get_logger

logger = get_logger(__name__)

# --- Configuration ---
MODEL_ID = "sshleifer/tiny-gpt2"  # tiny model for quick experimentation
HF_DATASET = "roneneldan/TinyStories"
DATA_SPLIT = "train"
TEXT_FIELD = "text"
DATA_LIMIT = 200  # keep small for a quick demo
MAX_LENGTH = 128
BATCH_SIZE_SAVE = 8

# Choose which layer to hook. You can use an integer index or a layer name.
# Use model.layers.get_layer_names() below to inspect available names.
LAYER_SIGNATURE: int | str = 'gpt2lmheadmodel_transformer_h_1_ln_2'

# Storage locations
CACHE_DIR = Path("./store/tinystories")
STORE_DIR = Path("./store/tiny-gpt2")
RUN_ID = f"tinystories_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = "cpu"
# Choose a memory-friendly dtype where possible
if DEVICE == "cuda":
    DTYPE = torch.float16
elif DEVICE == "mps":
    # Use full precision on MPS for stability (some ops lack bf16 support)
    DTYPE = None
else:
    DTYPE = None  # keep native on CPU unless user explicitly opts-in

# SAE config
SAE_EPOCHS = 2
SAE_MINIBATCH = 64  # mini-batch within stored activation batches
SAE_LR = 1e-3
SAE_L1 = 0.0  # sparsity penalty on latents (set > 0.0 to encourage sparsity)


def main() -> None:
    # --- Initialize model and dataset ---
    model = LanguageModel.from_huggingface(MODEL_ID)
    model.model.to(DEVICE)

    # Optionally, inspect available layer names
    layer_names = model.layers.print_layer_names()
    logger.info("Discovered %d layers. Example names: %s", len(layer_names), layer_names[:5])

    # Load a small text dataset
    dataset = TextSnippetDataset.from_huggingface(
        HF_DATASET,
        split=DATA_SPLIT,
        cache_dir=str(CACHE_DIR),
        text_field=TEXT_FIELD,
        limit=DATA_LIMIT,
    )

    # Prepare a LocalStore for saving activations
    store = LocalStore(STORE_DIR)
    logger.info("Store base path: %s", store.base_path)
    logger.info("Run id: %s", RUN_ID)

    # --- Save activations for the chosen layer ---
    # This will iterate over dataset in small batches, run the model, capture layer outputs,
    # and write per-batch safetensors files under STORE_DIR/runs/{RUN_ID}/
    model.activations.infer_and_save(
        dataset,
        layer_signature=LAYER_SIGNATURE,
        run_name=RUN_ID,
        store=store,
        batch_size=BATCH_SIZE_SAVE,
        max_length=MAX_LENGTH,
        dtype=DTYPE,
        autocast=True,
        save_inputs=True,
        free_cuda_cache_every=50,
    )

    # --- Inspect one saved batch; infer hidden size ---
    first_batch = next(store.iter_run_batches(RUN_ID))
    acts = first_batch["activations"] if isinstance(first_batch, dict) else first_batch[0]
    logger.info("Saved activations shape: %s", tuple(acts.shape))

    # Flatten any leading dims to [N, D] to determine input dim
    hidden_dim = acts.shape[-1]
    logger.info("Inferred hidden_dim: %d", hidden_dim)

    # Safety guard: extremely wide activations (e.g., lm_head logits ~50k) are impractical for an SAE demo.
    if hidden_dim > 8192:
        logger.warning(
            "Hidden dimension %d is too large for this demo script. Choose a smaller hidden layer via LAYER_SIGNATURE.\n"
            "Tip: try an integer index like 0 or 1 from model.layers.get_layer_names(). Example names: %s",
            hidden_dim, layer_names[:10]
        )

    # --- Build the Sparse Autoencoder ---
    # Keep the SAE small by default for memory-friendliness
    n_latents = min(hidden_dim, 2048)

    # Log an estimate of parameter count and memory footprint
    bytes_per_param = 2 if (DTYPE in (torch.float16, torch.bfloat16)) else 4
    est_params = hidden_dim * n_latents * 2  # encoder + decoder (tied=False)
    est_mem_mb = est_params * bytes_per_param / (1024 ** 2)
    logger.info(
        "SAE config: n_inputs=%d n_latents=%d (~%d params; ~%.1f MB for weights)",
        hidden_dim, n_latents, est_params, est_mem_mb
    )

    sae = Autoencoder(n_latents=n_latents, n_inputs=hidden_dim, activation="TopK_4", tied=False, device=DEVICE)

    # --- Train the SAE from stored activations ---
    ckpt_dir = STORE_DIR / "checkpoints" / RUN_ID
    cfg = SAETrainingConfig(
        epochs=SAE_EPOCHS,
        batch_size=SAE_MINIBATCH,
        lr=SAE_LR,
        l1_lambda=SAE_L1,
        device=DEVICE,
        dtype=DTYPE,
        max_batches_per_epoch=None,
        validate_every=None,
        checkpoint_dir=ckpt_dir,
        project_decoder_grads=True,
        renorm_decoder_every=100,  # maintain stable decoder scale periodically
        verbose=True,
        use_amp=True,
        amp_dtype=DTYPE,
        grad_accum_steps=1,
        free_cuda_cache_every=50,
        wandb_enable=True,
        wandb_entity='amber_team',
        wandb_project='amber_playground',
        wandb_run_name=RUN_ID,
    )

    trainer = SAETrainer(sae, store, RUN_ID, cfg)
    history = trainer.train()
    logger.info("Training history: %s", history)

    # Optionally, save final model
    final_dir = STORE_DIR / "sae_models" / RUN_ID
    final_dir.mkdir(parents=True, exist_ok=True)
    sae.save("final", path=str(final_dir))
    logger.info("Saved final SAE to: %s", final_dir)


# Standard entry point
if __name__ == "__main__":
    main()
