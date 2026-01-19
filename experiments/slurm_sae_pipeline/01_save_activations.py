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
import requests
from pathlib import Path
from datetime import datetime

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
        if cfg.layer.layer_num is None:
            raise ValueError("Either layer_signature or layer_num must be provided in config")
        LAYER_NUM = cfg.layer.layer_num
        LAYER_SIGNATURE = f"llamaforcausallm_model_layers_{LAYER_NUM}_post_attention_layernorm"
    
    # Normalize to list for consistent handling
    if isinstance(LAYER_SIGNATURE, str):
        LAYER_SIGNATURES = [LAYER_SIGNATURE]
    else:
        LAYER_SIGNATURES = LAYER_SIGNATURE
    
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
    if len(LAYER_SIGNATURES) == 1:
        logger.info(f"🎯 Target layer: {LAYER_SIGNATURES[0]}")
    else:
        logger.info(f"🎯 Target layers ({len(LAYER_SIGNATURES)}): {', '.join(LAYER_SIGNATURES)}")

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
    logger.info(f"📱 Device: {DEVICE}")
    logger.info(f"📁 Store location: {lm.store.base_path}")
    if len(LAYER_SIGNATURES) == 1:
        logger.info(f"🎯 Target layer: {LAYER_SIGNATURES[0]}")
    else:
        logger.info(f"🎯 Target layers ({len(LAYER_SIGNATURES)}): {', '.join(LAYER_SIGNATURES)}")
    logger.info(f"📍 Layer type: resid_mid (after attention, before MLP)")

    # Verify layers exist
    available_layers = lm.layers.get_layer_names()
    missing_layers = [layer for layer in LAYER_SIGNATURES if layer not in available_layers]
    if missing_layers:
        logger.warning(f"⚠️  Layer(s) not found in model: {', '.join(missing_layers)}")
        logger.info("Available layers (first 10):")
        for layer in available_layers[:10]:
            logger.info(f"  - {layer}")
        logger.error(f"❌ Please use valid layer signatures")
        return

    logger.info("📥 Loading dataset...")
    # Special handling for datasets with old loading scripts (like polemo2-official)
    if "polemo2-official" in HF_DATASET:
        logger.info("📥 Downloading polemo2-official raw data file...")
        cache_dir = script_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        jsonl_file = cache_dir / f"polemo2_{DATA_SPLIT}.jsonl"
        
        should_download = False
        if not jsonl_file.exists() or jsonl_file.stat().st_size == 0:
            should_download = True
        elif DATA_LIMIT is None:
            cached_lines = sum(1 for _ in open(jsonl_file)) if jsonl_file.exists() else 0
            if cached_lines < 10000:
                logger.info(f"📊 Cached file has {cached_lines} rows, re-downloading to ensure completeness...")
                should_download = True
        
        if should_download:
            if jsonl_file.exists():
                jsonl_file.unlink()
                logger.info(f"🗑️  Removed old cache file to force re-download")
            
            logger.info("📥 Downloading raw data file from HuggingFace...")
            base_url = "https://huggingface.co/datasets/clarin-pl/polemo2-official/resolve/main/data"
            raw_file_url = f"{base_url}/all.sentence.{DATA_SPLIT}.txt"
            
            logger.info(f"   URL: {raw_file_url}")
            response = requests.get(raw_file_url, timeout=300, stream=True)
            response.raise_for_status()
            
            logger.info("💾 Parsing and saving dataset to JSONL...")
            jsonl_file.parent.mkdir(parents=True, exist_ok=True)
            count = 0
            download_limit = None if DATA_LIMIT is None else DATA_LIMIT * 2
            
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for line in response.iter_lines(decode_unicode=True):
                    if not line.strip():
                        continue
                    
                    splitted_line = line.split(" ")
                    if len(splitted_line) < 2:
                        continue
                    
                    text = " ".join(splitted_line[:-1])
                    if text:
                        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        count += 1
                        if count % 1000 == 0:
                            logger.info(f"   Saved {count} rows so far...")
                        if download_limit and count >= download_limit:
                            break
            
            logger.info(f"✅ Saved {count} rows to {jsonl_file}")
        
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
    
    if DATA_LIMIT is not None:
    if len(dataset) > DATA_LIMIT:
        logger.info(f"📊 Randomly sampling {DATA_LIMIT} rows from {len(dataset)} total rows...")
        dataset = dataset.random_sample(DATA_LIMIT)
        logger.info(f"✅ Sampled {len(dataset)} text samples")
    else:
        logger.info(f"📊 Using all {len(dataset)} available rows (less than requested {DATA_LIMIT})")
    else:
        logger.info(f"📊 Using all {len(dataset)} available rows (no data limit)")

    logger.info("💾 Saving activations...")
    logger.info(f"   Batch size: {BATCH_SIZE_SAVE}")
    logger.info(f"   Max length: {MAX_LENGTH}")
    
    run_name = lm.activations.save_activations_dataset(
        dataset,
        layer_signature=LAYER_SIGNATURES if len(LAYER_SIGNATURES) > 1 else LAYER_SIGNATURES[0],
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

