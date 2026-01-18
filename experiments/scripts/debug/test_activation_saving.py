"""
Debug script to test the optimized activation saving pipeline.
Loads a small subset of WildGuardMix and saves activations from Bielik-1.5B.
"""

from __future__ import annotations

import gc
from time import perf_counter

import torch

from mi_crow.datasets import LoadingStrategy, TextDataset
from mi_crow.language_model.language_model import LanguageModel
from mi_crow.store import LocalStore
from mi_crow.utils import get_logger, set_seed

logger = get_logger(__name__)


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configuration
    model_id = "speakleash/Bielik-1.5B-v3.0-Instruct"
    limit = 500
    batch_size = 64
    target_layer = 15  # Middle layer for testing stop_after_last_layer

    logger.info(f"Starting test on {device}")
    logger.info(f"Model: {model_id}")
    logger.info(f"Dataset: allenai/wildguardmixtrain (limit={limit})")

    # 1. Setup Stores
    results_store = LocalStore(base_path="store/debug_test")

    # 2. Load Dataset
    logger.info("Loading dataset from HF...")
    t0 = perf_counter()
    dataset = TextDataset.from_huggingface(
        repo_id="allenai/wildguardmix",
        store=results_store,
        name="wildguardtrain",
        split="train",
        text_field="prompt",
        limit=limit,
        loading_strategy=LoadingStrategy.MEMORY,
    )
    logger.info(f"Dataset loaded in {perf_counter() - t0:.2f}s. Samples: {len(dataset)}")

    # 3. Load Model
    logger.info("Loading model...")
    model_t0 = perf_counter()
    lm = LanguageModel.from_huggingface(model_id, store=results_store, device=device)
    logger.info(f"Model loaded in {perf_counter() - model_t0:.2f}s")

    layer_names = lm.layers.get_layer_names()
    layer_signature = layer_names["llamaforcausallm_model_layers_15"]
    logger.info(f"Target layer signature: {layer_signature}")

    # 4. Save Activations (Test new optimizations)
    logger.info("Testing save_activations_dataset with optimizations...")
    save_t0 = perf_counter()

    run_name = lm.activations.save_activations_dataset(
        dataset,
        layer_signature=layer_signature,
        batch_size=batch_size,
        dtype=torch.float16,  # Test dtype conversion
        stop_after_last_layer=True,  # Test early stopping
        free_cuda_cache_every=2,  # Test periodic cache clearing
        save_attention_mask=True,
        verbose=True,
    )

    save_duration = perf_counter() - save_t0
    logger.info(f"✅ Activations saved in {save_duration:.2f}s")
    logger.info(f"Run name: {run_name}")

    # 5. Verify Results
    logger.info("Verifying saved data...")
    batches = results_store.list_run_batches(run_name)
    logger.info(f"Total batches: {len(batches)}")

    if batches:
        meta, tensors = results_store.get_detector_metadata(run_name, 0)

        # Check activations
        if layer_signature in tensors:
            act = tensors[layer_signature]["activations"]
            logger.info(f"✅ Layer {target_layer} activations found. Shape: {act.shape}, Dtype: {act.dtype}")
            if act.dtype != torch.float16:
                logger.error(f"❌ Dtype mismatch! Expected torch.float16, got {act.dtype}")
        else:
            logger.error(f"❌ Layer {layer_signature} not found in saved tensors!")

        # Check attention masks
        if "attention_masks" in tensors:
            mask = tensors["attention_masks"]["attention_mask"]
            logger.info(f"✅ Attention masks found. Shape: {mask.shape}")
        else:
            logger.error("❌ Attention masks not found in saved tensors!")

    # Cleanup
    del lm
    del dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Test completed successfully!")


if __name__ == "__main__":
    main()
