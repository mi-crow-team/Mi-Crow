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
from datetime import datetime

from dotenv import load_dotenv

from config import PipelineConfig

from mi_crow.language_model.language_model import LanguageModel
from mi_crow.mechanistic.sae.modules.topk_sae import TopKSae, TopKSaeTrainingConfig
from mi_crow.store.local_store import LocalStore
from mi_crow.utils import get_logger

logger = get_logger(__name__)

# Load .env file from project root (only for wandb)
script_dir = Path(__file__).parent
project_root = script_dir
while project_root != project_root.parent:
    if (project_root / "pyproject.toml").exists() or (project_root / ".git").exists():
        break
    project_root = project_root.parent

env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)


def main():
    parser = argparse.ArgumentParser(description="Train SAE on saved activations")
    parser.add_argument("--run_id", type=str, default=None, help="Run ID to use (default: read from run_id.txt)")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file (default: config.json in script directory)")
    parser.add_argument("--layer", type=int, default=None, help="Layer index to use from layer_signatures list (0-based, default: use first layer)")
    args = parser.parse_args()
    
    # Load config.json into Pydantic model
    if args.config:
        config_file = Path(args.config)
    else:
        config_file = script_dir / "config.json"
    
    cfg = PipelineConfig.from_json_file(config_file)
    
    # Model configuration
    MODEL_ID = cfg.model.model_id
    
    # Storage configuration
    STORE_DIR = Path(cfg.storage.store_dir or str(script_dir / "store"))
    DEVICE = cfg.storage.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # SAE configuration
    N_LATENTS_MULTIPLIER = cfg.sae.n_latents_multiplier
    TOP_K = cfg.sae.top_k
    
    # Training configuration
    EPOCHS = cfg.training.epochs
    BATCH_SIZE_TRAIN = cfg.training.batch_size_train
    LR = cfg.training.lr
    L1_LAMBDA = cfg.training.l1_lambda
    SNAPSHOT_EVERY_N_EPOCHS = cfg.training.snapshot_every_n_epochs
    SNAPSHOT_BASE_PATH = cfg.training.snapshot_base_path
    
    # Wandb configuration (only from .env)
    USE_WANDB = os.getenv("USE_WANDB", "false").lower() in ("true", "1", "yes")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")
    WANDB_NAME = os.getenv("WANDB_NAME")
    WANDB_MODE = os.getenv("WANDB_MODE", "online")
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    if WANDB_API_KEY:
        WANDB_API_KEY = WANDB_API_KEY.strip().strip('"').strip("'")
    
    # Data type configuration
    # When using AMP, don't set dtype directly - let AMP handle it
    use_amp = DEVICE != "cpu"
    DTYPE = None if use_amp else None
    AMP_DTYPE = torch.float16 if use_amp and torch.cuda.is_available() else None
    
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
    lm = LanguageModel.from_huggingface(MODEL_ID, store=store, device=DEVICE)
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

    # Get layer signature from metadata first
    try:
        metadata = store.get_run_metadata(RUN_ID)
        layer_signatures = metadata.get("layer_signatures", [])
        if not layer_signatures:
            logger.error("❌ Error: No layer signatures found in metadata")
            return
        num_layers = len(layer_signatures)
        
        # Select layer based on --layer argument
        if args.layer is not None:
            if args.layer < 0 or args.layer >= len(layer_signatures):
                logger.error(f"❌ Error: Layer index {args.layer} out of range. Available: 0-{len(layer_signatures)-1}")
                return
            layer_signature = layer_signatures[args.layer]
            logger.info(f"🎯 Using layer index {args.layer}: {layer_signature}")
        else:
            layer_signature = layer_signatures[0]
            if len(layer_signatures) > 1:
                logger.warning(f"⚠️  Multiple layers in config ({len(layer_signatures)}), using first: {layer_signature}")
                logger.warning(f"   Use --layer N to select a different layer (0-{len(layer_signatures)-1})")
            logger.info(f"🎯 Target layer: {layer_signature}")
    except Exception as e:
        logger.error(f"❌ Error: Could not get layer signature from metadata: {e}")
        return

    # Get activation dimension from a sample batch
    sample_batch = store.get_run_batch(RUN_ID, batches[0])
    if isinstance(sample_batch, dict):
        if num_layers == 1:
            activations = sample_batch.get("activations")
        else:
            activations = sample_batch.get(f"activations_{layer_signature}")
    elif isinstance(sample_batch, list):
        activations = sample_batch[0] if sample_batch else None
    else:
        activations = sample_batch

    if activations is None:
        logger.error("❌ Error: Could not find activations in batch")
        if isinstance(sample_batch, dict):
            logger.error(f"   Available keys: {list(sample_batch.keys())}")
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
    logger.info(f"   Snapshots: {'every ' + str(SNAPSHOT_EVERY_N_EPOCHS) + ' epochs' if SNAPSHOT_EVERY_N_EPOCHS else 'disabled'}")
    logger.info(f"   Wandb: {'enabled' if USE_WANDB else 'disabled'}")
    if USE_WANDB:
        logger.info(f"   Wandb project: {WANDB_PROJECT}")
        if WANDB_ENTITY:
            logger.info(f"   Wandb entity: {WANDB_ENTITY}")
        if WANDB_NAME:
            logger.info(f"   Wandb name: {WANDB_NAME}")
        logger.info(f"   Wandb mode: {WANDB_MODE}")
        
        if WANDB_API_KEY:
            try:
                import wandb
                # Ensure API key is clean (strip any remaining whitespace/newlines)
                clean_api_key = WANDB_API_KEY.strip()
                # Validate API key length (wandb API keys are 40 characters)
                if len(clean_api_key) != 40:
                    logger.warning(f"⚠️  Invalid WANDB_API_KEY length: expected 40 characters, got {len(clean_api_key)}")
                    logger.warning("   Wandb will run in offline mode or use existing credentials")
                else:
                    os.environ['WANDB_API_KEY'] = clean_api_key
                    wandb.login(key=clean_api_key)
                    logger.info("✅ Wandb login successful")
            except ImportError:
                logger.warning("⚠️  wandb not installed, skipping wandb login")
            except Exception as e:
                logger.warning(f"⚠️  Wandb login failed: {e}")
                logger.warning("   Wandb will run in offline mode or use existing credentials")
        else:
            logger.info("ℹ️  No WANDB_API_KEY in .env file, wandb will use existing credentials if available")

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
        amp_dtype=AMP_DTYPE,
        clip_grad=1.0,
        monitoring=2,
        memory_efficient=True,
        snapshot_every_n_epochs=SNAPSHOT_EVERY_N_EPOCHS,
        snapshot_base_path=SNAPSHOT_BASE_PATH,
        # Wandb configuration
        use_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
        wandb_name=WANDB_NAME or RUN_ID,  # Use run_id if WANDB_NAME not set
        wandb_mode=WANDB_MODE,
        wandb_api_key=WANDB_API_KEY,
        wandb_tags=["topk-sae", "sae-training", MODEL_ID.split("/")[-1], f"layer-{layer_signature}"],
        wandb_config={
            "model_id": MODEL_ID,
            "layer_signature": layer_signature,
            "n_inputs": n_inputs,
            "n_latents": n_latents,
            "n_latents_multiplier": N_LATENTS_MULTIPLIER,
            "top_k": TOP_K,
            "run_id": RUN_ID,
        },
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

    layer_safe_name = layer_signature.replace("/", "_").replace("\\", "_")
    custom_training_run_id = f"sae_{layer_safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    result = sae.train(store, RUN_ID, layer_signature, config, training_run_id=custom_training_run_id)
    training_run_id = result.get('training_run_id')
    wandb_url = result.get('wandb_url')

    logger.info("✅ Training function returned!")

    if training_run_id:
        logger.info(f"💾 Training outputs saved to: store/runs/{training_run_id}/")
        logger.info(f"   - Model: store/runs/{training_run_id}/model.pt")
        logger.info(f"   - History: store/runs/{training_run_id}/history.json")
        logger.info(f"   - Metadata: store/runs/{training_run_id}/meta.json")
        if SNAPSHOT_EVERY_N_EPOCHS:
            logger.info(f"   - Snapshots: store/runs/{training_run_id}/snapshots/")
        if wandb_url:
            logger.info(f"   - Wandb URL: {wandb_url}")
    else:
        logger.warning("⚠️  No training_run_id returned from training")


if __name__ == "__main__":
    main()

