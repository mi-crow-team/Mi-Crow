# ruff: noqa
"""
Inspect saved activation tensors to diagnose NaN issue.

This script loads saved activations directly from disk and checks for NaN patterns.
"""

import argparse
from pathlib import Path

import torch

from mi_crow.store import LocalStore
from mi_crow.utils import get_logger

logger = get_logger(__name__)


def inspect_activations(store_base: str, run_id: str, layer_signature: str):
    """
    Load and inspect saved activations for NaN patterns.

    Args:
        store_base: Base path to LocalStore
        run_id: Run ID containing saved activations
        layer_signature: Layer signature to inspect
    """
    store = LocalStore(base_path=store_base)
    batch_indices = store.list_run_batches(run_id)

    logger.info(f"Found {len(batch_indices)} batches in run: {run_id}")
    logger.info(f"Inspecting layer: {layer_signature}")
    logger.info("=" * 80)

    for batch_idx in batch_indices:
        try:
            # Load activations
            activations = store.get_detector_metadata_by_layer_by_key(run_id, batch_idx, layer_signature, "activations")

            # Load attention mask
            try:
                attention_mask = store.get_detector_metadata_by_layer_by_key(
                    run_id, batch_idx, "attention_masks", "attention_mask"
                )
                has_attention_mask = True
            except Exception:
                has_attention_mask = False
                attention_mask = None

            # Analyze NaN pattern
            nan_mask = torch.isnan(activations)
            inf_mask = torch.isinf(activations)

            total_elements = activations.numel()
            nan_count = nan_mask.sum().item()
            inf_count = inf_mask.sum().item()
            nan_percentage = (nan_count / total_elements) * 100

            logger.info(f"\nBatch {batch_idx}:")
            logger.info(f"  Shape: {tuple(activations.shape)}")
            logger.info(f"  Device: {activations.device}")
            logger.info(f"  Dtype: {activations.dtype}")
            logger.info(f"  NaNs: {nan_count}/{total_elements} ({nan_percentage:.2f}%)")
            logger.info(f"  Infs: {inf_count}/{total_elements}")

            if nan_count == total_elements:
                logger.warning(f"  ⚠️  ENTIRE batch is NaN!")
            elif nan_count > 0:
                logger.warning(f"  ⚠️  Partial NaN contamination")

                # Check NaN pattern per sample
                batch_size = activations.shape[0]
                all_nan_samples = []
                partial_nan_samples = []

                for sample_idx in range(batch_size):
                    sample = activations[sample_idx]
                    sample_nan_count = torch.isnan(sample).sum().item()
                    sample_total = sample.numel()

                    if sample_nan_count == sample_total:
                        all_nan_samples.append(sample_idx)
                    elif sample_nan_count > 0:
                        partial_nan_samples.append(sample_idx)

                if all_nan_samples:
                    logger.warning(
                        f"  Samples with ALL NaN: {all_nan_samples[:10]}"
                        + (f" ...and {len(all_nan_samples) - 10} more" if len(all_nan_samples) > 10 else "")
                    )
                if partial_nan_samples:
                    logger.warning(
                        f"  Samples with PARTIAL NaN: {partial_nan_samples[:10]}"
                        + (f" ...and {len(partial_nan_samples) - 10} more" if len(partial_nan_samples) > 10 else "")
                    )

                # Sample non-NaN values
                non_nan_mask = ~nan_mask
                non_nan_values = activations[non_nan_mask]
                if non_nan_values.numel() > 0:
                    logger.info(f"  Non-NaN sample (first 10): {non_nan_values[:10].tolist()}")
                    logger.info(
                        f"  Non-NaN stats - mean: {non_nan_values.mean().item():.4f}, std: {non_nan_values.std().item():.4f}"
                    )
            else:
                # Normal batch - show statistics
                logger.info(f"  ✅ No NaNs detected")
                logger.info(
                    f"  Stats - mean: {activations.mean().item():.4f}, std: {activations.std().item():.4f}, min: {activations.min().item():.4f}, max: {activations.max().item():.4f}"
                )

            if has_attention_mask:
                logger.info(f"  Attention mask shape: {tuple(attention_mask.shape)}")
                valid_tokens = attention_mask.sum().item()
                total_tokens = attention_mask.numel()
                logger.info(f"  Valid tokens: {valid_tokens}/{total_tokens}")

        except Exception as e:
            logger.error(f"Failed to load batch {batch_idx}: {e}")
            continue

    logger.info("=" * 80)
    logger.info("Inspection complete")


def main():
    parser = argparse.ArgumentParser(description="Inspect saved activations for NaN patterns")
    parser.add_argument("--store", type=str, default="store", help="Base path to LocalStore")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID to inspect")
    parser.add_argument(
        "--layer-signature",
        type=str,
        default=None,
        help="Layer signature (e.g., llamaforcausallm_model_layers_31). If not provided, will try to detect.",
    )

    args = parser.parse_args()

    # If layer signature not provided, try to detect from first batch
    layer_signature = args.layer_signature
    if layer_signature is None:
        logger.info("Layer signature not provided, attempting to detect...")
        store = LocalStore(base_path=args.store)
        batches = store.list_run_batches(args.run_id)
        if batches:
            _, tensors = store.get_detector_metadata(args.run_id, batches[0])
            # Find layer activation (skip attention_masks, input_ids, etc.)
            for key in tensors.keys():
                if "llamaforcausallm" in key.lower():
                    layer_signature = key
                    logger.info(f"Detected layer signature: {layer_signature}")
                    break

        if layer_signature is None:
            logger.error("Could not detect layer signature. Please provide it explicitly with --layer-signature")
            return 1

    inspect_activations(args.store, args.run_id, layer_signature)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
