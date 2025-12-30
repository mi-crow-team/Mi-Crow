#!/usr/bin/env python3
"""
SLURM script to train SAE on saved activations from Bielik 1.5B Instruct model.

This script:
- Loads saved activations from the activation saving script
- Creates a TopKSAE model
- Trains the SAE on the activations
- Saves the trained model and training history

Usage:
    python 02_train_sae.py
    # or with environment variables:
    STORE_DIR=/path/to/store python 02_train_sae.py
    # or with --run_id flag:
    python 02_train_sae.py --run_id my_custom_run_id
"""

import argparse
import os
import torch
from pathlib import Path

from mi_crow.language_model.language_model import LanguageModel
from mi_crow.mechanistic.sae.modules.topk_sae import TopKSae, TopKSaeTrainingConfig
from mi_crow.store.local_store import LocalStore
from mi_crow.utils import get_logger

logger = get_logger(__name__)

# Model configuration
MODEL_ID = os.getenv("MODEL_ID", "speakleash/Bielik-1.5B-v3.0-Instruct")  # Bielik 1.5B Instruct

# Storage configuration - use SLURM environment variables if available
STORE_DIR = Path(os.getenv("STORE_DIR", os.getenv("SCRATCH", str(Path(__file__).parent / "store"))))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# SAE configuration
N_LATENTS_MULTIPLIER = int(os.getenv("N_LATENTS_MULTIPLIER", "4"))  # Overcompleteness factor
TOP_K = int(os.getenv("TOP_K", "8"))  # Sparsity parameter

# Training configuration
EPOCHS = int(os.getenv("EPOCHS", "10"))
BATCH_SIZE_TRAIN = int(os.getenv("BATCH_SIZE_TRAIN", "32"))
LR = float(os.getenv("LR", "1e-3"))
L1_LAMBDA = float(os.getenv("L1_LAMBDA", "1e-4"))

# Data type configuration
DTYPE = torch.float16 if DEVICE == "cuda" and torch.cuda.is_available() else None


def main():
    parser = argparse.ArgumentParser(description="Train SAE on saved activations")
    parser.add_argument("--run_id", type=str, default=None, help="Run ID to use (default: read from run_id.txt)")
    args = parser.parse_args()
    
    logger.info("🚀 Starting SAE Training")
    logger.info(f"📱 Using device: {DEVICE}")
    logger.info(f"🔧 Model: {MODEL_ID}")
    logger.info(f"📁 Store directory: {STORE_DIR}")

    STORE_DIR.mkdir(parents=True, exist_ok=True)

    # Set CUDA device if using GPU
    if DEVICE == "cuda" and torch.cuda.is_available():
        logger.info(f"🔌 CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    logger.info("📥 Loading language model...")
    store = LocalStore(base_path=STORE_DIR)
    lm = LanguageModel.from_huggingface(MODEL_ID, store=store)
    lm.model.to(DEVICE)
    lm.model.eval()  # Set to evaluation mode

    logger.info(f"✅ Model loaded: {lm.model_id}")

    # Load run ID from argument or previous step
    if args.run_id:
        RUN_ID = args.run_id
        logger.info(f"📁 Using run ID from command line: {RUN_ID}")
    else:
        run_id_file = STORE_DIR / "run_id.txt"
        if not run_id_file.exists():
            logger.error("❌ Error: run_id.txt not found. Please run 01_save_activations.py first or use --run_id flag.")
            logger.error(f"   Expected file: {run_id_file}")
            return

        with open(run_id_file, "r") as f:
            RUN_ID = f.read().strip()

    logger.info(f"📁 Using run ID: {RUN_ID}")

    batches = store.list_run_batches(RUN_ID)
    if not batches:
        logger.error(f"❌ Error: No batches found for run ID: {RUN_ID}")
        logger.error(f"   Check that activations were saved correctly")
        return

    logger.info(f"📦 Found {len(batches)} batches")

    # Get activation dimension from a sample batch
    sample_batch = store.get_run_batch(RUN_ID, batches[0])
    if isinstance(sample_batch, dict):
        activations = sample_batch.get("activations")
    elif isinstance(sample_batch, list):
        activations = sample_batch[0] if sample_batch else None
    else:
        activations = sample_batch

    if activations is None:
        logger.error("❌ Error: Could not find activations in batch")
        return

    if isinstance(activations, torch.Tensor):
        n_inputs = activations.shape[-1]
    else:
        logger.error("❌ Error: Activations is not a tensor")
        return

    n_latents = n_inputs * N_LATENTS_MULTIPLIER

    logger.info(f"📊 Activation dimension: {n_inputs}")
    logger.info(f"📊 SAE latents: {n_latents} ({N_LATENTS_MULTIPLIER}x overcomplete)")
    logger.info(f"📊 TopK: {TOP_K}")

    # Get layer signature from metadata
    try:
        metadata = store.get_run_metadata(RUN_ID)
        layer_signatures = metadata.get("layer_signatures", [])
        if not layer_signatures:
            logger.error("❌ Error: No layer signatures found in metadata")
            return
        layer_signature = layer_signatures[0]
    except Exception as e:
        logger.error(f"❌ Error: Could not get layer signature from metadata: {e}")
        return

    logger.info(f"🎯 Target layer: {layer_signature}")

    logger.info("🏗️  Creating TopKSAE...")
    sae = TopKSae(
        n_latents=n_latents,
        n_inputs=n_inputs,
        k=TOP_K,
        device=DEVICE,
        store=store,
    )
    logger.info(f"✅ SAE created: {n_inputs} -> {n_latents} (TopK={TOP_K})")

    logger.info("🏋️ Training TopKSAE...")
    logger.info(f"   Epochs: {EPOCHS}")
    logger.info(f"   Batch size: {BATCH_SIZE_TRAIN}")
    logger.info(f"   Learning rate: {LR}")
    logger.info(f"   L1 lambda: {L1_LAMBDA}")
    logger.info(f"   Device: {DEVICE}")
    logger.info(f"   Dtype: {DTYPE}")
    logger.info(f"   Monitoring: detailed (level 2)")
    logger.info(f"   Memory efficient: True")

    use_amp = DEVICE != "cpu"
    if not use_amp:
        logger.warning(f"   ⚠️  AMP disabled for CPU device")

    config = TopKSaeTrainingConfig(
        k=TOP_K,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE_TRAIN,
        lr=LR,
        l1_lambda=L1_LAMBDA,
        device=DEVICE,
        dtype=DTYPE,
        verbose=True,
        use_amp=use_amp,
        amp_dtype=DTYPE if use_amp else None,
        clip_grad=1.0,
        monitoring=2,
        memory_efficient=True,
    )

    logger.info("📦 Checking dataloader...")
    from mi_crow.store.store_dataloader import StoreDataloader
    test_dataloader = StoreDataloader(
        store=store,
        run_id=RUN_ID,
        layer=layer_signature,
        key="activations",
        batch_size=BATCH_SIZE_TRAIN,
        dtype=DTYPE,
        max_batches=1,
    )
    logger.info("   Testing dataloader with 1 batch...")
    try:
        test_batch = next(iter(test_dataloader))
        logger.info(
            f"   ✅ Dataloader works! Batch shape: {test_batch.shape if hasattr(test_batch, 'shape') else type(test_batch)}")
    except Exception as e:
        logger.error(f"   ❌ Dataloader error: {e}")
        return

    logger.info("🚀 Starting training (this may take a while)...")

    result = sae.train(store, RUN_ID, layer_signature, config)
    training_run_id = result.get('training_run_id')

    logger.info("✅ Training function returned!")

    if training_run_id:
        logger.info(f"💾 Training outputs saved to: store/runs/{training_run_id}/")
        logger.info(f"   - Model: store/runs/{training_run_id}/model.pt")
        logger.info(f"   - History: store/runs/{training_run_id}/history.json")
        logger.info(f"   - Metadata: store/runs/{training_run_id}/meta.json")
    else:
        logger.warning("⚠️  No training_run_id returned from training")


if __name__ == "__main__":
    main()

