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
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

from mi_crow.datasets.classification_dataset import ClassificationDataset
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
        "category_field": ["prompt_harm_label", "subcategory"],
        "description": "WildGuardMix Train (English)",
        "lang": "en",
    },
    "wgmix_test": {
        "store_path": "store/datasets/wgmix_test",
        "text_field": "prompt",
        "category_field": ["prompt_harm_label", "subcategory"],
        "description": "WildGuardMix Test (English)",
        "lang": "en",
    },
    "plmix_train": {
        "store_path": "store/datasets/plmix_train",
        "text_field": "text",
        "category_field": "text_harm_label",
        "description": "PL Mix Train (Polish)",
        "lang": "pl",
    },
    "plmix_test": {
        "store_path": "store/datasets/plmix_test",
        "text_field": "text",
        "category_field": "text_harm_label",
        "description": "PL Mix Test (Polish)",
        "lang": "pl",
    },
}

# Prefix-forcing, stimuli
PREFIX_TEMPLATES = {
    "en": "Consider the amount of harmfulness in the following prompt:\n{prompt}\nThe amount of harmfulness is",
    "pl": "Oceń stopień szkodliwości w poniższym zapytaniu:\n{prompt}\nStopień szkodliwości wynosi",
}

# CPU optimization: number of threads for PyTorch operations
# N_CPU_THREADS = 3

MAX_LENGTH = 256


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


def _log_gpu_memory(batch_idx: int, stage: str, memory_log: list) -> None:
    """Log GPU memory usage to tracking list."""
    if not torch.cuda.is_available():
        return

    mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    mem_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3

    memory_log.append(
        {
            "batch": batch_idx,
            "stage": stage,
            "allocated_gb": round(mem_allocated, 2),
            "reserved_gb": round(mem_reserved, 2),
            "free_gb": round(mem_free, 2),
        }
    )

    logger.info(
        "[Batch %d - %s] GPU Memory: %.2f GB allocated, %.2f GB reserved, %.2f GB free",
        batch_idx,
        stage,
        mem_allocated,
        mem_reserved,
        mem_free,
    )


def main() -> int:  # noqa: C901
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
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--store", type=str, default="store", help="LocalStore base path for saving results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--use-prefix",
        action="store_true",
        help="Apply language-specific prefix forcing templates to the input",
    )

    args = parser.parse_args()

    # CPU optimization: set number of threads for PyTorch operations
    # torch.set_num_threads(N_CPU_THREADS)
    # logger.info("Set PyTorch CPU threads to: %d", N_CPU_THREADS)

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
    category_field = dataset_config.get("category_field", None)

    dataset_store = LocalStore(base_path=dataset_store_path)
    dataset = ClassificationDataset.from_disk(store=dataset_store, text_field=text_field, category_field=category_field)

    dataset_load_s = perf_counter() - dataset_t0
    logger.info("✅ Dataset loaded: %d samples (%.2fs)", len(dataset), dataset_load_s)

    # Load model
    logger.info("Loading model...")
    model_t0 = perf_counter()

    results_store = LocalStore(args.store)

    # CPU optimization: use bfloat16 for 2x memory reduction and faster operations
    # model_params = {"torch_dtype": torch.bfloat16}
    lm = LanguageModel.from_huggingface(
        args.model,
        store=results_store,
        device=args.device,
        # model_params=model_params
    )

    # Get effective max length
    effective_max_length = _get_effective_max_length(lm)
    logger.info("Effective max input length: %d", effective_max_length)
    max_length = min(effective_max_length, MAX_LENGTH)
    logger.info("Using max length: %d", max_length)

    model_load_s = perf_counter() - model_t0
    logger.info("✅ Model loaded: %s (%.2fs)", lm.model_id, model_load_s)

    # Get layer signature
    layer_signature = _get_layer_signature(lm, layer_num)
    logger.info("Target layer: %s", layer_signature)

    # Generate run name
    model_short = args.model.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    dataset_short = args.dataset
    prefix_suffix = "_prefixed" if args.use_prefix else ""
    ts = _timestamp()
    run_name = f"activations_maxlen_{MAX_LENGTH}_{model_short}_{dataset_short}{prefix_suffix}_layer{layer_num}_{ts}"

    logger.info("Run name: %s", run_name)

    # Initialize memory tracking
    memory_log = []
    _log_gpu_memory(-1, "after_model_load", memory_log)

    # Setup activation hooks
    from mi_crow.hooks import HookType
    from mi_crow.hooks.implementations.layer_activation_detector import LayerActivationDetector
    from mi_crow.hooks.implementations.model_input_detector import ModelInputDetector

    # Register activation detector for target layer
    activation_detector = LayerActivationDetector(
        layer_signature=layer_signature, hook_id=f"detector_{run_name}_{layer_signature}"
    )
    activation_hook_id = lm.layers.register_hook(layer_signature, activation_detector, HookType.FORWARD)

    # Register attention mask detector
    attention_mask_layer_sig = "attention_masks"
    if attention_mask_layer_sig not in lm.layers.name_to_layer:
        lm.layers.name_to_layer[attention_mask_layer_sig] = lm.context.model

    attention_mask_detector = ModelInputDetector(
        layer_signature=attention_mask_layer_sig,
        hook_id=f"attention_mask_detector_{run_name}",
        save_input_ids=False,
        save_attention_mask=True,
    )
    attention_mask_hook_id = lm.layers.register_hook(
        attention_mask_layer_sig, attention_mask_detector, HookType.PRE_FORWARD
    )

    # Save activations with attention masks
    logger.info("Saving activations with attention masks...")
    save_t0 = perf_counter()

    # Process batches with memory monitoring
    num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    logger.info("Processing %d batches of size %d", num_batches, args.batch_size)

    # Use inference_mode for better performance and memory management (matches save_activations_dataset)
    with torch.inference_mode():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(dataset))
            batch_texts = dataset.get_all_texts()[start_idx:end_idx]

            if args.use_prefix:
                lang = dataset_config.get("lang")
                if lang and lang in PREFIX_TEMPLATES:
                    template = PREFIX_TEMPLATES[lang]
                    batch_texts = [template.format(prompt=text) for text in batch_texts]
                else:
                    logger.warning("No template found for language '%s', skipping prefix/stimuli", lang)

            _log_gpu_memory(batch_idx, "before_batch", memory_log)

            # Log batch sequence lengths
            if lm.context.tokenizer:
                batch_encodings = lm.context.tokenizer(
                    batch_texts,
                    padding=False,
                    truncation=True,
                    max_length=max_length,
                    return_tensors=None,
                )
                seq_lengths = [len(ids) for ids in batch_encodings["input_ids"]]
                logger.info(
                    "[Batch %d] Sequence lengths: min=%d, max=%d, mean=%.1f, samples=%d",
                    batch_idx,
                    min(seq_lengths),
                    max(seq_lengths),
                    sum(seq_lengths) / len(seq_lengths),
                    len(batch_texts),
                )

            try:
                lm.activations._process_batch(
                    batch_texts,
                    run_name=run_name,
                    batch_index=batch_idx,
                    max_length=max_length,
                    autocast=False,
                    autocast_dtype=None,
                    dtype=None,
                    verbose=True,
                    save_in_batches=True,
                    stop_after_layer=layer_signature,
                )
                _log_gpu_memory(batch_idx, "after_batch", memory_log)
            except torch.cuda.OutOfMemoryError as e:
                _log_gpu_memory(batch_idx, "OOM_error", memory_log)
                logger.error("OOM on batch %d: %s", batch_idx, str(e))
                # Save memory log before crashing
                memory_log_path = Path(results_store.base_path) / "runs" / f"{run_name}_memory_log.json"
                memory_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(memory_log_path, "w") as f:
                    json.dump(memory_log, f, indent=2)
                logger.info("Memory log saved to: %s", memory_log_path)
                raise

    # Cleanup hooks
    lm.layers.unregister_hook(activation_hook_id)
    lm.layers.unregister_hook(attention_mask_hook_id)

    actual_run_name = run_name

    save_s = perf_counter() - save_t0
    logger.info("✅ Saved activations with attention masks (%.2fs)", save_s)
    logger.info("Actual run name: %s", actual_run_name)

    # Save memory log
    memory_log_path = Path(results_store.base_path) / "runs" / f"{run_name}_memory_log.json"
    memory_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(memory_log_path, "w") as f:
        json.dump(memory_log, f, indent=2)
    logger.info("Memory log saved to: %s", memory_log_path)

    # Get save location
    run_dir = Path(results_store.base_path) / "runs" / actual_run_name
    logger.info("Save location: %s", run_dir)

    # Verify saved data
    logger.info("Verifying saved data...")
    batches = lm.context.store.list_run_batches(actual_run_name)
    logger.info("✅ Saved %d batches", len(batches))

    # Check first batch structure and validate data integrity
    if batches:
        _, tensors = lm.context.store.get_detector_metadata(actual_run_name, 0)
        logger.info("First batch contains layers: %s", list(tensors.keys()))

        if str(layer_signature) in tensors:
            activations = tensors[str(layer_signature)].get("activations")
            if activations is not None:
                logger.info("✅ Activations shape: %s", activations.shape)

                # Check for NaN values
                nan_count = torch.isnan(activations).sum().item()
                if nan_count > 0:
                    total_elements = activations.numel()
                    nan_percentage = (nan_count / total_elements) * 100
                    logger.error(
                        "❌ WARNING: First batch activations contain %d NaN values (%.2f%% of total)",
                        nan_count,
                        nan_percentage,
                    )
                else:
                    logger.info("✅ No NaN values detected in activations")
                    # Log basic statistics
                    logger.info(
                        "   Activation stats - mean: %.4f, std: %.4f, min: %.4f, max: %.4f",
                        activations.mean().item(),
                        activations.std().item(),
                        activations.min().item(),
                        activations.max().item(),
                    )

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
