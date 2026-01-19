#!/usr/bin/env python3
"""
SLURM script to run inference with attached SAEs and collect top texts.

This script:
- Loads trained SAEs from provided paths (one per layer)
- Attaches SAEs to corresponding layers
- Enables text tracking on all SAEs
- Attaches ModelInputDetector to capture input texts
- Runs inference on dataset from config
- Exports top texts for each SAE

Usage:
    python 03_run_inference.py --sae_paths path1.pt path2.pt --config config.json
"""

import argparse
import json
import os
import torch
from datetime import datetime
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv

from config import PipelineConfig

from mi_crow.datasets import TextDataset
from mi_crow.hooks.hook import HookType
from mi_crow.hooks.implementations.model_input_detector import ModelInputDetector
from mi_crow.language_model.language_model import LanguageModel
from mi_crow.mechanistic.sae.modules.topk_sae import TopKSae
from mi_crow.mechanistic.sae.modules.l1_sae import L1Sae
from mi_crow.mechanistic.sae.sae import Sae
from mi_crow.store.local_store import LocalStore
from mi_crow.utils import get_logger

try:
    from server.utils import SAERegistry
except ImportError:
    SAERegistry = None

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


def load_sae_auto(sae_path: Path) -> Sae:
    """
    Load SAE and auto-detect class from metadata.json or model file.
    
    Args:
        sae_path: Path to SAE model file
        
    Returns:
        Loaded SAE instance
        
    Raises:
        ValueError: If SAE cannot be loaded
    """
    sae_path = Path(sae_path)
    if not sae_path.exists():
        raise ValueError(f"SAE file not found: {sae_path}")
    
    metadata_path = sae_path.parent / "metadata.json"
    
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text())
            sae_class = meta.get("sae_class") or meta.get("sae_type")
            if sae_class:
                if SAERegistry is not None:
                    try:
                        registry = SAERegistry()
                        cls = registry.get_class(sae_class)
                        return cls.load(sae_path)
                    except Exception as e:
                        logger.warning(f"Could not use SAERegistry for '{sae_class}': {e}, trying direct load")
                
                if sae_class == "TopKSae":
                    return TopKSae.load(sae_path)
                elif sae_class == "L1Sae":
                    return L1Sae.load(sae_path)
                else:
                    logger.warning(f"Unknown SAE class '{sae_class}', trying auto-detection")
        except Exception as e:
            logger.warning(f"Could not read metadata.json: {e}, trying auto-detection")
    
    try:
        return TopKSae.load(sae_path)
    except (ValueError, KeyError) as e:
        try:
            return L1Sae.load(sae_path)
        except Exception as e2:
            raise ValueError(f"Could not load SAE from {sae_path}. Tried TopKSae and L1Sae. TopKSae error: {e}, L1Sae error: {e2}") from e2


def main():
    parser = argparse.ArgumentParser(description="Run inference with attached SAEs and collect top texts")
    parser.add_argument("--sae_paths", type=str, nargs="+", required=True, help="List of SAE model paths (one per layer in config order)")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file (default: config.json in script directory)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of texts to track per neuron (default: 5, e.g., use --top_k 10 to track 10 texts)")
    parser.add_argument("--track_both", action="store_true", help="Track both positive (top) and negative (bottom) activations (saves both top_texts and bottom_texts files). If not set, only tracks positive (top) activations.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for inference (default: 32, capped even if config is higher)")
    parser.add_argument("--stop_after_layer", type=str, default=None, help="Optional layer signature or index to stop forward pass after (saves memory)")
    parser.add_argument("--data_limit", type=int, default=None, help="Limit number of samples (default: from config or None)")
    parser.add_argument("--dump_every_n_batches", type=int, default=250, help="Dump current heaps to JSON every N batches (0 disables)")
    args = parser.parse_args()
    
    if args.config:
        config_file = Path(args.config)
    else:
        config_file = script_dir / "config.json"
    
    cfg = PipelineConfig.from_json_file(config_file)
    
    MODEL_ID = cfg.model.model_id
    
    HF_DATASET = cfg.dataset.hf_dataset
    DATA_SPLIT = cfg.dataset.data_split
    TEXT_FIELD = cfg.dataset.text_field
    DATA_LIMIT = args.data_limit or cfg.dataset.data_limit
    MAX_LENGTH = cfg.dataset.max_length
    default_batch = cfg.dataset.batch_size_save or 32
    BATCH_SIZE = args.batch_size or min(default_batch, 32)
    
    LAYER_SIGNATURE = cfg.layer.layer_signature
    if LAYER_SIGNATURE is None:
        if cfg.layer.layer_num is None:
            raise ValueError("Either layer_signature or layer_num must be provided in config")
        LAYER_NUM = cfg.layer.layer_num
        LAYER_SIGNATURE = f"llamaforcausallm_model_layers_{LAYER_NUM}_post_attention_layernorm"
    
    if isinstance(LAYER_SIGNATURE, str):
        LAYER_SIGNATURES = [LAYER_SIGNATURE]
    else:
        LAYER_SIGNATURES = LAYER_SIGNATURE

    stop_after_layer = args.stop_after_layer
    if stop_after_layer is None and LAYER_SIGNATURES:
        # Stop after the last SAE-attached layer to avoid unnecessary forward passes
        stop_after_layer = LAYER_SIGNATURES[-1]
    
    STORE_DIR = Path(cfg.storage.store_dir or str(script_dir / "store"))
    DEVICE = cfg.storage.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    TOP_K = args.top_k
    TRACK_BOTH = args.track_both
    SAE_PATHS = [Path(p) for p in args.sae_paths]
    
    if len(SAE_PATHS) != len(LAYER_SIGNATURES):
        raise ValueError(
            f"Number of SAE paths ({len(SAE_PATHS)}) must match number of layers ({len(LAYER_SIGNATURES)}). "
            f"Layers: {LAYER_SIGNATURES}"
        )
    
    logger.info("🚀 Starting SAE Inference with Text Tracking")
    logger.info(f"📱 Using device: {DEVICE}")
    logger.info(f"🔧 Model: {MODEL_ID}")
    logger.info(f"📊 Dataset: {HF_DATASET}")
    logger.info(f"📊 Data limit: {DATA_LIMIT}")
    logger.info(f"📁 Store directory: {STORE_DIR}")
    logger.info(f"🎯 Layers ({len(LAYER_SIGNATURES)}): {', '.join(LAYER_SIGNATURES)}")
    logger.info(f"📦 SAE paths ({len(SAE_PATHS)}):")
    for i, path in enumerate(SAE_PATHS):
        logger.info(f"   {i}: {path}")
    logger.info(f"🔢 Top-K texts per neuron: {TOP_K}")
    if TRACK_BOTH:
        logger.info(f"📊 Tracking: both positive (top) and negative (bottom) activations")
    else:
        logger.info(f"📊 Tracking: positive (top) activations only")
    logger.info(f"📦 Batch size: {BATCH_SIZE}")
    
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    
    if DEVICE == "cuda" and torch.cuda.is_available():
        logger.info(f"🔌 CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    logger.info("📥 Loading language model...")
    store = LocalStore(base_path=STORE_DIR)
    lm = LanguageModel.from_huggingface(MODEL_ID, store=store, device=DEVICE)
    lm.model.eval()
    
    logger.info(f"✅ Model loaded: {lm.model_id}")
    
    logger.info("📥 Loading SAEs...")
    sae_hooks = []
    for i, (layer_sig, sae_path) in enumerate(zip(LAYER_SIGNATURES, SAE_PATHS)):
        logger.info(f"   Loading SAE {i+1}/{len(SAE_PATHS)} from {sae_path}...")
        sae = load_sae_auto(sae_path)
        sae.sae_engine.to(DEVICE)
        logger.info(f"   ✅ Loaded {type(sae).__name__}: {sae.context.n_inputs} -> {sae.context.n_latents}")
        
        hook_id = lm.layers.register_hook(layer_sig, sae)
        sae.context.lm = lm
        sae.context.lm_layer_signature = layer_sig
        sae.context.text_tracking_enabled = True
        sae.context.text_tracking_k = TOP_K
        sae.context.text_tracking_negative = False
        sae.concepts.enable_text_tracking()
        sae_hooks.append((sae, hook_id, layer_sig))
        logger.info(f"   ✅ Attached {type(sae).__name__} to {layer_sig}")
    
    logger.info("🔧 Attaching ModelInputDetector...")
    input_layer_sig = "model_inputs"
    root_model = lm.model
    if input_layer_sig not in lm.layers.name_to_layer:
        lm.layers.name_to_layer[input_layer_sig] = root_model
    
    input_detector = ModelInputDetector(
        layer_signature=input_layer_sig,
        hook_id="model_input_detector",
        save_input_ids=True,
        save_attention_mask=False,
    )
    input_hook_id = lm.layers.register_hook(input_layer_sig, input_detector, HookType.PRE_FORWARD)
    logger.info(f"✅ ModelInputDetector attached (hook ID: {input_hook_id})")
    
    logger.info("📥 Loading dataset...")
    if "polemo2-official" in HF_DATASET:
        logger.info("📥 Using HuggingFace datasets server API for polemo2-official...")
        cache_dir = script_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        jsonl_file = cache_dir / f"polemo2_{DATA_SPLIT}.jsonl"
        
        if not jsonl_file.exists() or jsonl_file.stat().st_size == 0:
            download_limit = None if DATA_LIMIT is None else DATA_LIMIT * 2
            
            base_url = "https://datasets-server.huggingface.co/rows"
            dataset_name = "clarin-pl%2Fpolemo2-official"
            config = "all_sentence"
            batch_size = 100
            
            all_rows = []
            offset = 0
            
            logger.info(f"📥 Downloading polemo2-official {DATA_SPLIT} split from HuggingFace datasets server...")
            
            while True:
                url = f"{base_url}?dataset={dataset_name}&config={config}&split={DATA_SPLIT}&offset={offset}&length={batch_size}"
                
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    
                    rows = data.get("rows", [])
                    if not rows:
                        break
                        
                    for row in rows:
                        row_data = row.get("row", {})
                        text = row_data.get("text", "")
                        if text:
                            all_rows.append({"text": text})
                    
                    logger.info(f"   Downloaded {len(all_rows)} rows so far...")
                    
                    if download_limit and len(all_rows) >= download_limit:
                        all_rows = all_rows[:download_limit]
                        break
                        
                    if len(rows) < batch_size:
                        break
                        
                    offset += batch_size
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"❌ Error downloading dataset: {e}")
                    raise
            
            jsonl_file.parent.mkdir(parents=True, exist_ok=True)
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for row in all_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
            logger.info(f"✅ Downloaded {len(all_rows)} rows to {jsonl_file}")
        
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
    
    inference_run_id = f"top_texts_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    inference_run_dir = STORE_DIR / "runs" / inference_run_id
    inference_run_dir.mkdir(parents=True, exist_ok=True)
    
    dump_every_n_batches = args.dump_every_n_batches

    for sae, hook_id, layer_sig in sae_hooks:
        sae.context.text_tracking_negative = TRACK_BOTH
        sae.concepts._text_tracking_negative = TRACK_BOTH
        sae.concepts.reset_top_texts()
        sae.concepts.enable_text_tracking()
    
    logger.info("🚀 Running inference to collect texts...")
    if TRACK_BOTH:
        logger.info("   Tracking both positive (top) and negative (bottom) activations")
    else:
        logger.info("   Tracking positive (top) activations only")
    logger.info(f"   Processing {len(dataset)} samples in batches of {BATCH_SIZE}")
    
    with torch.inference_mode():
        batch_count = 0
        logger.info("[DEBUG] Starting batch loop...")
        for batch in dataset.iter_batches(BATCH_SIZE):
            logger.info(f"[DEBUG] Got batch {batch_count+1}, extracting texts...")
            texts = dataset.extract_texts_from_batch(batch)
            logger.info(f"[DEBUG] Extracted {len(texts)} texts, starting inference...")
            import time
            t0 = time.time()
            try:
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
                    stop_after_layer=stop_after_layer,
                    verbose=True,
                )
                logger.info(f"[DEBUG] Inference completed in {time.time()-t0:.2f}s")
            except Exception as e:
                logger.error(f"[DEBUG] Inference failed after {time.time()-t0:.2f}s: {e}", exc_info=True)
                raise
            
            batch_count += 1
            if batch_count % 10 == 0:
                logger.info(f"   Processed {batch_count} batches...")

            if dump_every_n_batches and dump_every_n_batches > 0 and (batch_count % dump_every_n_batches == 0):
                logger.info(f"💾 Dumping heaps at batch {batch_count} to: {inference_run_dir}")
                for i, (sae, hook_id, layer_sig) in enumerate(sae_hooks):
                    layer_safe = layer_sig.replace("/", "_")
                    dump_top_path = inference_run_dir / f"top_texts_layer_{i}_{layer_safe}_batch_{batch_count}.json"
                    sae.concepts.export_top_texts_to_json(dump_top_path)
                    if TRACK_BOTH:
                        dump_bottom_path = inference_run_dir / f"bottom_texts_layer_{i}_{layer_safe}_batch_{batch_count}.json"
                        sae.concepts.export_bottom_texts_to_json(dump_bottom_path)
    
    logger.info("✅ Inference completed")
    
    logger.info(f"💾 Exporting texts for {len(sae_hooks)} SAE(s) to: {inference_run_dir}")
    for i, (sae, hook_id, layer_sig) in enumerate(sae_hooks):
        top_texts_file = inference_run_dir / f"top_texts_layer_{i}_{layer_sig.replace('/', '_')}.json"
        
        try:
            exported_path = sae.concepts.export_top_texts_to_json(top_texts_file)
            logger.info(f"   ✅ SAE {i+1} ({layer_sig}) - Positive: {exported_path}")
            
            all_top_texts = sae.concepts.get_all_top_texts()
            neurons_with_texts = sum(1 for texts in all_top_texts if texts)
            total_texts = sum(len(texts) for texts in all_top_texts)
            
            logger.info(f"      📊 Neurons with texts: {neurons_with_texts} / {sae.context.n_latents}")
            logger.info(f"      📊 Total texts collected: {total_texts}")
            if sae.context.n_latents > 0:
                logger.info(f"      📊 Average texts per neuron: {total_texts / sae.context.n_latents:.2f}")
        except Exception as e:
            logger.error(f"   ❌ Failed to export top texts for SAE {i+1}: {e}")
        
        if TRACK_BOTH:
            bottom_texts_file = inference_run_dir / f"bottom_texts_layer_{i}_{layer_sig.replace('/', '_')}.json"
            try:
                exported_path = sae.concepts.export_bottom_texts_to_json(bottom_texts_file)
                logger.info(f"   ✅ SAE {i+1} ({layer_sig}) - Negative: {exported_path}")
                
                all_bottom_texts = sae.concepts.get_all_bottom_texts()
                neurons_with_texts = sum(1 for texts in all_bottom_texts if texts)
                total_texts = sum(len(texts) for texts in all_bottom_texts)
                
                logger.info(f"      📊 Neurons with texts: {neurons_with_texts} / {sae.context.n_latents}")
                logger.info(f"      📊 Total texts collected: {total_texts}")
                if sae.context.n_latents > 0:
                    logger.info(f"      📊 Average texts per neuron: {total_texts / sae.context.n_latents:.2f}")
            except Exception as e:
                logger.error(f"   ❌ Failed to export bottom texts for SAE {i+1}: {e}")
    
    logger.info("🧹 Cleaning up hooks...")
    try:
        lm.layers.unregister_hook(input_hook_id)
    except Exception as e:
        logger.warning(f"   ⚠️  Could not unregister input detector: {e}")
    
    for sae, hook_id, layer_sig in sae_hooks:
        try:
            lm.layers.unregister_hook(hook_id)
        except Exception as e:
            logger.warning(f"   ⚠️  Could not unregister SAE hook for {layer_sig}: {e}")
    
    logger.info("✅ Inference script completed!")


if __name__ == "__main__":
    main()
