#!/usr/bin/env python3
"""
SLURM script to save activations from Bielik 1.5B Instruct model.

This script:
- Loads the Bielik 1.5B Instruct model
- Loads a dataset from HuggingFace
- Saves activations from a specified layer
- Stores run ID for use in training script

Usage:
    python 01_save_activations.py
    # or with environment variables:
    STORE_DIR=/path/to/store python 01_save_activations.py
    # or with --run_id flag:
    python 01_save_activations.py --run_id my_custom_run_id
"""

import argparse
import logging
import os
import torch
from pathlib import Path
from datetime import datetime

from amber.datasets import TextDataset
from amber.language_model.language_model import LanguageModel
from amber.store.local_store import LocalStore
from amber.utils import get_logger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

logger = get_logger(__name__)

# Model configuration
MODEL_ID = os.getenv("MODEL_ID", "speakleash/Bielik-1.5B-v3.0-Instruct")  # Bielik 1.5B Instruct

# Dataset configuration
HF_DATASET = os.getenv("HF_DATASET", "chrisociepa/wikipedia-pl-20230401")
DATA_SPLIT = os.getenv("DATA_SPLIT", "train")
TEXT_FIELD = os.getenv("TEXT_FIELD", "text")
DATA_LIMIT = int(os.getenv("DATA_LIMIT", "25000"))  # 25K random rows
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "1000"))  # 1000 tokens max
BATCH_SIZE_SAVE = int(os.getenv("BATCH_SIZE_SAVE", "16"))

# Layer configuration - adjust layer number based on model architecture
# Default: llamaforcausallm_model_layers_16_post_attention_layernorm (middle layer for 32-layer model)
# You can set LAYER_SIGNATURE directly or use LAYER_NUM to construct it
LAYER_SIGNATURE = os.getenv("LAYER_SIGNATURE", None)
if LAYER_SIGNATURE is None:
    LAYER_NUM = int(os.getenv("LAYER_NUM", "16"))  # Middle layer for 32-layer model
    LAYER_SIGNATURE = f"llamaforcausallm_model_layers_{LAYER_NUM}_post_attention_layernorm"

# Storage configuration - use SLURM environment variables if available
STORE_DIR = Path(os.getenv("STORE_DIR", os.getenv("SCRATCH", str(Path(__file__).parent / "store"))))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Save activations from Bielik model")
    parser.add_argument("--run_id", type=str, default=None, help="Custom run ID (default: auto-generated)")
    args = parser.parse_args()
    
    if args.run_id:
        RUN_ID = args.run_id
    else:
        RUN_ID = f"activations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("🚀 Starting Activation Saving")
    logger.info(f"📱 Using device: {DEVICE}")
    logger.info(f"🔧 Model: {MODEL_ID}")
    logger.info(f"📊 Dataset: {HF_DATASET}")
    logger.info(f"📊 Data limit: {DATA_LIMIT}")
    logger.info(f"💾 Run ID: {RUN_ID}")
    logger.info(f"📁 Store directory: {STORE_DIR}")
    logger.info(f"🎯 Target layer: {LAYER_SIGNATURE}")

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
    logger.info(f"📱 Device: {DEVICE}")
    logger.info(f"📁 Store location: {lm.store.base_path}")
    logger.info(f"🎯 Target layer: {LAYER_SIGNATURE}")
    logger.info(f"📍 Layer type: resid_mid (after attention, before MLP)")

    # Verify layer exists
    available_layers = lm.layers.get_layer_names()
    if LAYER_SIGNATURE not in available_layers:
        logger.warning(f"⚠️  Layer '{LAYER_SIGNATURE}' not found in model")
        logger.info("Available layers (first 10):")
        for layer in available_layers[:10]:
            logger.info(f"  - {layer}")
        logger.error("❌ Please set LAYER_NUM environment variable to a valid layer number")
        return

    logger.info("📥 Loading dataset...")
    # Load dataset (we'll shuffle and sample after loading)
    # Load more than needed to ensure we have enough for random sampling
    dataset = TextDataset.from_huggingface(
        HF_DATASET,
        split=DATA_SPLIT,
        store=store,
        text_field=TEXT_FIELD,
        limit=None,  # Load all, then sample randomly
    )
    logger.info(f"✅ Loaded {len(dataset)} text samples from dataset")
    
    # Randomly sample DATA_LIMIT rows
    if len(dataset) > DATA_LIMIT:
        logger.info(f"📊 Randomly sampling {DATA_LIMIT} rows from {len(dataset)} total rows...")
        sampled_texts = dataset.sample(DATA_LIMIT)
        # Create a new dataset from sampled texts
        from datasets import Dataset
        sampled_ds = Dataset.from_dict({"text": [t for t in sampled_texts if t is not None]})
        dataset = TextDataset(sampled_ds, store=store, text_field=TEXT_FIELD)
        logger.info(f"✅ Sampled {len(dataset)} text samples")
    else:
        logger.info(f"📊 Using all {len(dataset)} available rows (less than requested {DATA_LIMIT})")

    logger.info("💾 Saving activations...")
    logger.info(f"   Batch size: {BATCH_SIZE_SAVE}")
    logger.info(f"   Max length: {MAX_LENGTH}")
    
    run_name = lm.activations.save_activations_dataset(
        dataset,
        layer_signature=LAYER_SIGNATURE,
        run_name=RUN_ID,
        batch_size=BATCH_SIZE_SAVE,
        max_length=MAX_LENGTH,
        autocast=False,
        verbose=True,
    )

    batches = lm.store.list_run_batches(run_name)
    logger.info(f"✅ Activations saved!")
    logger.info(f"📁 Run name: {run_name}")
    logger.info(f"📦 Saved {len(batches)} batches to store")
    
    # Save run ID for training script
    run_id_file = STORE_DIR / "run_id.txt"
    with open(run_id_file, "w") as f:
        f.write(run_name)
    logger.info(f"💾 Run ID saved to: {run_id_file}")


if __name__ == "__main__":
    main()

