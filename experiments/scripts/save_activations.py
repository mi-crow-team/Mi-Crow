"""
Save activations and attention masks for models on datasets.

This script saves activations from a specified layer (default: last layer) along with
attention masks for later analysis. The activations are saved in batches to disk.

Usage:
    # Save activations for Llama on WGMix train
    uv run python -m experiments.scripts.save_activations \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --dataset wgmix_train \
        --layer-num 27 \
        --batch-size 64 \
        --device cuda

    # Save activations for Bielik on PLMix train
    uv run python -m experiments.scripts.save_activations \
        --model speakleash/Bielik-1.5B-v3.0-Instruct \
        --dataset plmix_train \
        --layer-num 31 \
        --batch-size 64 \
        --device cuda
"""

from __future__ import annotations

import argparse
import gc
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

from mi_crow.datasets import TextDataset
from mi_crow.language_model.language_model import LanguageModel
from mi_crow.store import LocalStore
from mi_crow.utils import get_logger, set_seed

logger = get_logger(__name__)

# Model configurations: default last layer indices
MODEL_CONFIGS = {
    "meta-llama/Llama-3.2-3B-Instruct": {
        "default_layer": 27,
        "description": "Meta Llama 3.2 3B Instruct",
    },
    "speakleash/Bielik-1.5B-v3.0-Instruct": {
        "default_layer": 31,
        "description": "Bielik 1.5B v3.0 Instruct",
    },
    "speakleash/Bielik-4.5B-v3.0-Instruct": {
        "default_layer": 59,
        "description": "Bielik 4.5B v3.0 Instruct",
    },
}

# Dataset configurations
DATASET_CONFIGS = {
    "wgmix_train": {
        "store_path": "store/datasets/wgmix_train",
        "text_field": "prompt",
        "description": "WildGuardMix Train (English)",
    },
    "wgmix_test": {
        "store_path": "store/datasets/wgmix_test",
        "text_field": "prompt",
        "description": "WildGuardMix Test (English)",
    },
    "plmix_train": {
        "store_path": "store/datasets/plmix_train",
        "text_field": "text",
        "description": "PL Mix Train (Polish)",
    },
    "plmix_test": {
        "store_path": "store/datasets/plmix_test",
        "text_field": "text",
        "description": "PL Mix Test (Polish)",
    },
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_layer_signature(lm: LanguageModel, layer_num: int) -> str:
    """Get the layer signature for a given layer number.

    Constructs the layer signature in the format: llamaforcausallm_model_layers_{layer_num}
    and validates it exists in the model.
    """
    # Construct the expected layer signature
    layer_signature = f"llamaforcausallm_model_layers_{layer_num}"

    # Verify it exists in the model
    layer_names = lm.layers.get_layer_names()
    if layer_signature not in layer_names:
        logger.error(
            "Layer '%s' not found in model. Available layers matching pattern:",
            layer_signature,
        )
        # Show matching layers for debugging
        matching = [name for name in layer_names if "llamaforcausallm_model_layers_" in name]
        for name in matching[:10]:  # Show first 10
            logger.error("  - %s", name)
        if len(matching) > 10:
            logger.error("  ... and %d more", len(matching) - 10)
        raise ValueError(f"Layer '{layer_signature}' not found in model")

    return layer_signature


def _get_effective_max_length(lm: LanguageModel, reserved_tokens: int = 0) -> int:
    """Calculate safe max input length for the model."""
    tokenizer = lm.context.tokenizer
    model = lm.context.model

    tok_max = getattr(tokenizer, "model_max_length", None)
    cfg_max = getattr(getattr(model, "config", None), "max_position_embeddings", None)

    # Filter out sentinel values
    if isinstance(tok_max, int) and tok_max > 1_000_000:
        tok_max = None

    candidates = [x for x in (tok_max, cfg_max) if isinstance(x, int) and x > 0]
    max_len = min(candidates) if candidates else 2048

    return max(1, max_len - reserved_tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description="Save activations and attention masks for models on datasets")

    # Model and dataset selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use for saving activations",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to process (must be prepared first with prepare_datasets.py)",
    )

    # Layer selection
    parser.add_argument(
        "--layer-num",
        type=int,
        default=None,
        help="Layer number to extract activations from (default: model's last layer)",
    )

    # Processing parameters
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--store", type=str, default="store", help="LocalStore base path for saving results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    script_t0 = perf_counter()
    set_seed(args.seed)

    # Get model and dataset configurations
    model_config = MODEL_CONFIGS[args.model]
    dataset_config = DATASET_CONFIGS[args.dataset]

    # Determine layer number
    if args.layer_num is None:
        layer_num = model_config["default_layer"]
        logger.info("Using default layer number: %d", layer_num)
    else:
        layer_num = args.layer_num
        logger.info("Using specified layer number: %d", layer_num)

    logger.info("=" * 80)
    logger.info("Model: %s (%s)", args.model, model_config["description"])
    logger.info("Dataset: %s (%s)", args.dataset, dataset_config["description"])
    logger.info("Layer number: %d", layer_num)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("Device: %s", args.device)
    logger.info("=" * 80)

    # Load dataset from disk
    logger.info("Loading dataset from disk...")
    dataset_t0 = perf_counter()

    dataset_store_path = dataset_config["store_path"]
    text_field = dataset_config["text_field"]

    dataset_store = LocalStore(base_path=dataset_store_path)
    dataset = TextDataset.from_disk(
        store=dataset_store, 
        text_field=text_field,
    )

    dataset_load_s = perf_counter() - dataset_t0
    logger.info("✅ Dataset loaded: %d samples (%.2fs)", len(dataset), dataset_load_s)

    # Load model
    logger.info("Loading model...")
    model_t0 = perf_counter()

    results_store = LocalStore(args.store)
    lm = LanguageModel.from_huggingface(args.model, store=results_store, device=args.device)

    # Get effective max length
    effective_max_length = _get_effective_max_length(lm)
    logger.info("Effective max input length: %d", effective_max_length)
    max_length = min(effective_max_length, 1024)
    logger.info("Using max length: %d", max_length)

    model_load_s = perf_counter() - model_t0
    logger.info("✅ Model loaded: %s (%.2fs)", lm.model_id, model_load_s)

    # Get layer signature
    layer_signature = _get_layer_signature(lm, layer_num)
    logger.info("Target layer: %s", layer_signature)

    # Generate run name
    model_short = args.model.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    dataset_short = args.dataset
    ts = _timestamp()
    run_name = f"activations_{model_short}_{dataset_short}_layer{layer_num}_{ts}"

    logger.info("Run name: %s", run_name)

    # Save activations with attention masks
    logger.info("Saving activations with attention masks...")
    save_t0 = perf_counter()

    actual_run_name = lm.activations.save_activations_dataset(
        dataset,
        layer_signature=layer_signature,
        run_name=run_name,
        batch_size=args.batch_size,
        max_length=max_length,
        autocast=False,
        verbose=True,
        save_attention_mask=True,
    )

    save_s = perf_counter() - save_t0
    logger.info("✅ Saved activations with attention masks (%.2fs)", save_s)
    logger.info("Actual run name: %s", actual_run_name)

    # Get save location
    run_dir = Path(results_store.base_path) / "runs" / actual_run_name
    logger.info("Save location: %s", run_dir)

    # Verify saved data
    logger.info("Verifying saved data...")
    batches = lm.context.store.list_run_batches(actual_run_name)
    logger.info("✅ Saved %d batches", len(batches))

    # Check first batch structure
    if batches:
        _, tensors = lm.context.store.get_detector_metadata(actual_run_name, 0)
        logger.info("First batch contains layers: %s", list(tensors.keys()))

        if str(layer_signature) in tensors:
            activations = tensors[str(layer_signature)].get("activations")
            if activations is not None:
                logger.info("✅ Activations shape: %s", activations.shape)

        if "attention_masks" in tensors:
            attention_mask = tensors["attention_masks"].get("attention_mask")
            if attention_mask is not None:
                logger.info("✅ Attention mask shape: %s", attention_mask.shape)

    # Cleanup memory
    del lm
    del dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_s = perf_counter() - script_t0
    logger.info("=" * 80)
    logger.info("Total time: %.2fs", total_s)
    logger.info("  Dataset load: %.2fs", dataset_load_s)
    logger.info("  Model load: %.2fs", model_load_s)
    logger.info("  Save activations: %.2fs", save_s)
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
