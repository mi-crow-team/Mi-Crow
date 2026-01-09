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
import json
import logging
import os
import torch
from pathlib import Path
from datetime import datetime

import requests
from dotenv import load_dotenv

from config import PipelineConfig

from mi_crow.datasets import TextDataset
from mi_crow.language_model.language_model import LanguageModel
from mi_crow.store.local_store import LocalStore
from mi_crow.utils import get_logger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

logger = get_logger(__name__)

script_dir = Path(__file__).parent
project_root = script_dir
while project_root != project_root.parent:
    if (project_root / "pyproject.toml").exists() or (project_root / ".git").exists():
        break
    project_root = project_root.parent

env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)


def _download_polemo2_from_api(split: str, output_file: Path, limit: int = None) -> int:
    """Download polemo2 dataset from HuggingFace datasets server API.
    
    Args:
        split: Dataset split to download (train, val, test)
        output_file: Path to save the JSONL file
        limit: Optional limit on number of rows to download
        
    Returns:
        Number of rows downloaded
    """
    base_url = "https://datasets-server.huggingface.co/rows"
    dataset = "clarin-pl%2Fpolemo2-official"
    config = "all_sentence"
    batch_size = 100
    
    all_rows = []
    offset = 0
    
    logger.info(f"📥 Downloading polemo2-official {split} split from HuggingFace datasets server...")
    
    while True:
        url = f"{base_url}?dataset={dataset}&config={config}&split={split}&offset={offset}&length={batch_size}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            rows = data.get("rows", [])
            if not rows:
                break
                
            # Extract text from each row
            for row in rows:
                row_data = row.get("row", {})
                text = row_data.get("text", "")
                if text:
                    all_rows.append({"text": text})
            
            logger.info(f"   Downloaded {len(all_rows)} rows so far...")
            
            if limit and len(all_rows) >= limit:
                all_rows = all_rows[:limit]
                break
                
            if len(rows) < batch_size:
                break
                
            offset += batch_size
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error downloading dataset: {e}")
            raise
    
    # Save to JSONL file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    logger.info(f"✅ Downloaded {len(all_rows)} rows to {output_file}")
    return len(all_rows)


def main():
    parser = argparse.ArgumentParser(description="Save activations from Bielik model")
    parser.add_argument("--run_id", type=str, default=None, help="Custom run ID (default: auto-generated)")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file (default: config.json in script directory)")
    args = parser.parse_args()
    
    # Load config.json into Pydantic model
    if args.config:
        config_file = Path(args.config)
    else:
        config_file = script_dir / "config.json"
    
    cfg = PipelineConfig.from_json_file(config_file)
    
    # Model configuration
    MODEL_ID = cfg.model.model_id
    
    # Dataset configuration
    HF_DATASET = cfg.dataset.hf_dataset
    DATA_SPLIT = cfg.dataset.data_split
    TEXT_FIELD = cfg.dataset.text_field
    DATA_LIMIT = cfg.dataset.data_limit
    MAX_LENGTH = cfg.dataset.max_length
    BATCH_SIZE_SAVE = cfg.dataset.batch_size_save
    
    # Layer configuration
    LAYER_SIGNATURE = cfg.layer.layer_signature
    if LAYER_SIGNATURE is None:
        LAYER_NUM = cfg.layer.layer_num
        LAYER_SIGNATURE = f"llamaforcausallm_model_layers_{LAYER_NUM}_post_attention_layernorm"
    
    # Storage configuration
    STORE_DIR = Path(cfg.storage.store_dir or str(script_dir / "store"))
    DEVICE = cfg.storage.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
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
    # Special handling for datasets with old loading scripts (like polemo2-official)
    if "polemo2-official" in HF_DATASET:
        logger.info("📥 Using HuggingFace datasets server API for polemo2-official...")
        # Download dataset to local file
        cache_dir = script_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        jsonl_file = cache_dir / f"polemo2_{DATA_SPLIT}.jsonl"
        
        # Download if file doesn't exist or is empty
        if not jsonl_file.exists() or jsonl_file.stat().st_size == 0:
            _download_polemo2_from_api(DATA_SPLIT, jsonl_file, limit=DATA_LIMIT * 2)  # Download more than needed for sampling
        
        # Load from local JSONL file
        dataset = TextDataset.from_json(
            jsonl_file,
            store=store,
            text_field=TEXT_FIELD,
        )
    else:
        dataset = TextDataset.from_huggingface(
            HF_DATASET,
            split=DATA_SPLIT,
            store=store,
            text_field=TEXT_FIELD,
            limit=None,
        )
    logger.info(f"✅ Loaded {len(dataset)} text samples from dataset")
    
    if len(dataset) > DATA_LIMIT:
        logger.info(f"📊 Randomly sampling {DATA_LIMIT} rows from {len(dataset)} total rows...")
        dataset = dataset.random_sample(DATA_LIMIT)
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
    
    run_id_file = STORE_DIR / "run_id.txt"
    with open(run_id_file, "w") as f:
        f.write(run_name)
    logger.info(f"💾 Run ID saved to: {run_id_file}")


if __name__ == "__main__":
    main()

