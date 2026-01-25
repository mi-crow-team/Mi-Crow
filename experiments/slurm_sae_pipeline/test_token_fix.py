#!/usr/bin/env python3
"""
Quick test script to verify token ID fix works correctly.
Runs inference with very few batches and checks for token_str errors.
"""

import json
import sys
import torch
from pathlib import Path

# Add script directory to path for config import
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from dotenv import load_dotenv
from config import PipelineConfig
from mi_crow.datasets import TextDataset
from mi_crow.hooks.implementations.model_input_detector import ModelInputDetector
from mi_crow.language_model.language_model import LanguageModel
from mi_crow.mechanistic.sae.modules.topk_sae import TopKSae
from mi_crow.store.local_store import LocalStore
from mi_crow.utils import get_logger

logger = get_logger(__name__)

project_root = script_dir
while project_root != project_root.parent:
    if (project_root / "pyproject.toml").exists() or (project_root / ".git").exists():
        break
    project_root = project_root.parent

env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)

def main():
    config_file = script_dir / "configs" / "test_token_fix.json"
    cfg = PipelineConfig.from_json_file(config_file)
    
    MODEL_ID = cfg.model.model_id
    HF_DATASET = cfg.dataset.hf_dataset
    DATA_SPLIT = cfg.dataset.data_split
    TEXT_FIELD = cfg.dataset.text_field
    DATA_LIMIT = cfg.dataset.data_limit
    MAX_LENGTH = cfg.dataset.max_length
    BATCH_SIZE = 2  # Very small batch size
    
    LAYER_NUM = cfg.layer.layer_num
    LAYER_SIGNATURE = f"llamaforcausallm_model_layers_{LAYER_NUM}_post_attention_layernorm"
    
    STORE_DIR = Path(cfg.storage.store_dir or str(script_dir / "store"))
    DEVICE = cfg.storage.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("🧪 Testing Token ID Fix")
    logger.info(f"📱 Device: {DEVICE}")
    logger.info(f"🔧 Model: {MODEL_ID}")
    logger.info(f"📊 Dataset: {HF_DATASET}, limit: {DATA_LIMIT}")
    logger.info(f"📦 Batch size: {BATCH_SIZE}")
    logger.info(f"🎯 Layer: {LAYER_SIGNATURE}")
    
    # Find an existing SAE
    sae_paths = list(STORE_DIR.glob("runs/sae_*/model.pt"))
    if not sae_paths:
        logger.error("❌ No SAE models found. Please train an SAE first.")
        return
    
    sae_path = sae_paths[0]
    logger.info(f"📦 Using SAE: {sae_path}")
    
    # Load model
    logger.info("📥 Loading language model...")
    store = LocalStore(base_path=STORE_DIR)
    lm = LanguageModel.from_huggingface(MODEL_ID, store=store, device=DEVICE)
    lm.model.eval()
    logger.info(f"✅ Model loaded: {lm.model_id}")
    
    # Load SAE
    logger.info("📥 Loading SAE...")
    sae = TopKSae.load(sae_path)
    sae.context.lm = lm
    sae.context.device = DEVICE
    sae.context.lm_layer_signature = LAYER_SIGNATURE
    
    # Enable text tracking
    sae.context.text_tracking_enabled = True
    sae.context.text_tracking_k = 5
    sae.concepts.enable_text_tracking()
    
    # Register SAE hook
    hook_id = lm.layers.register_hook(LAYER_SIGNATURE, sae)
    logger.info(f"✅ SAE hook registered: {hook_id}")
    
    # Register input detector
    input_detector = ModelInputDetector(layer_signature=None, save_input_ids=True)
    input_hook_id = lm.layers.register_hook(None, input_detector, hook_type="PRE_FORWARD")
    logger.info(f"✅ Input detector registered: {input_hook_id}")
    
    # Load dataset
    logger.info("📥 Loading dataset...")
    dataset = TextDataset.from_huggingface(
        dataset_name=HF_DATASET,
        split=DATA_SPLIT,
        text_field=TEXT_FIELD,
        limit=DATA_LIMIT
    )
    logger.info(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Run inference on just 1 batch
    logger.info("🚀 Running inference (1 batch only)...")
    batch = list(dataset.iter_batches(BATCH_SIZE))[0]
    texts = dataset.extract_texts_from_batch(batch)
    logger.info(f"   Processing {len(texts)} texts")
    
    with torch.inference_mode():
        lm.inference.infer_texts(
            texts,
            run_name=None,
            batch_size=None,
            tok_kwargs={
                "max_length": MAX_LENGTH,
                "padding": True,
                "truncation": True,
                "add_special_tokens": True
            },
            autocast=False,
            stop_after_layer=LAYER_SIGNATURE,
            verbose=True,
        )
    
    logger.info("✅ Inference completed")
    
    # Check for token_str errors
    logger.info("🔍 Checking for token decoding issues...")
    errors_found = []
    for neuron_idx in range(min(10, sae.context.n_latents)):  # Check first 10 neurons
        top_texts = sae.concepts.get_top_texts_for_neuron(neuron_idx, top_m=5)
        for nt in top_texts:
            if "out_of_range" in nt.token_str or "decode_error" in nt.token_str:
                errors_found.append({
                    "neuron": neuron_idx,
                    "text": nt.text[:50] + "...",
                    "token_str": nt.token_str,
                    "token_idx": nt.token_idx,
                    "token_id": nt.token_id
                })
    
    if errors_found:
        logger.error(f"❌ Found {len(errors_found)} token decoding errors:")
        for err in errors_found[:5]:  # Show first 5
            logger.error(f"   Neuron {err['neuron']}: {err['token_str']} (idx={err['token_idx']}, id={err['token_id']})")
        return False
    else:
        logger.info("✅ No token decoding errors found!")
        
        # Show sample results
        logger.info("📊 Sample results (first neuron):")
        sample_texts = sae.concepts.get_top_texts_for_neuron(0, top_m=3)
        for i, nt in enumerate(sample_texts, 1):
            logger.info(f"   {i}. Score: {nt.score:.4f}, Token: '{nt.token_str}' (idx={nt.token_idx}, id={nt.token_id})")
            logger.info(f"      Text: {nt.text[:60]}...")
        
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
