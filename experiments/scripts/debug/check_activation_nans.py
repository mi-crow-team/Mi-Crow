# ruff: noqa
"""
Check safetensors activation files for NaN values.

This script loads activation tensors saved by save_activations.py and analyzes
them for NaN values, providing detailed statistics.

Usage:
    # Check a single batch activation file
    uv run python -m experiments.scripts.debug.check_activation_nans \
        --file /path/to/activations.safetensors

    # Check with verbose output
    uv run python -m experiments.scripts.debug.check_activation_nans \
        --file /path/to/activations.safetensors \
        --verbose

Example:
    uv run python -m experiments.scripts.debug.check_activation_nans \
        --file /mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store/runs/activations_maxlen_512_bielik_1_5b_v3_0_instruct_wgmix_train_layer31_20260126_044427/batch_0/llamaforcausallm_model_layers_31/activations.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from mi_crow.utils import get_logger

logger = get_logger(__name__)


def analyze_nan_values(tensor: torch.Tensor, name: str = "activations", verbose: bool = False) -> dict:
    """
    Analyze a tensor for NaN values and return statistics.

    Args:
        tensor: The tensor to analyze
        name: Name of the tensor for logging
        verbose: Whether to print detailed statistics

    Returns:
        Dictionary with NaN analysis results
    """
    nan_mask = torch.isnan(tensor)
    nan_count = nan_mask.sum().item()
    total_elements = tensor.numel()
    all_nan = nan_count == total_elements
    has_nan = nan_count > 0
    nan_percentage = (nan_count / total_elements * 100) if total_elements > 0 else 0

    results = {
        "name": name,
        "shape": tuple(tensor.shape),
        "total_elements": total_elements,
        "nan_count": nan_count,
        "nan_percentage": nan_percentage,
        "has_nan": has_nan,
        "all_nan": all_nan,
    }

    # Additional statistics for non-NaN values if tensor is not all NaN
    if not all_nan and total_elements > 0:
        valid_mask = ~nan_mask
        valid_values = tensor[valid_mask]

        if valid_values.numel() > 0:
            results["valid_count"] = valid_values.numel()
            results["min"] = valid_values.min().item()
            results["max"] = valid_values.max().item()
            results["mean"] = valid_values.mean().item()
            results["std"] = valid_values.std().item()

    # Print results
    logger.info("=" * 80)
    logger.info("Tensor: %s", name)
    logger.info("Shape: %s", results["shape"])
    logger.info("Total elements: %d", total_elements)
    logger.info("-" * 80)

    if all_nan:
        logger.error("❌ ALL VALUES ARE NaN!")
    elif has_nan:
        logger.warning("⚠️  Contains NaN values:")
        logger.warning("   NaN count: %d (%.2f%%)", nan_count, nan_percentage)
        logger.warning("   Valid values: %d (%.2f%%)", results.get("valid_count", 0), 100 - nan_percentage)
    else:
        logger.info("✅ No NaN values detected")

    if verbose and not all_nan and has_nan:
        # Show NaN distribution across dimensions
        logger.info("-" * 80)
        logger.info("NaN distribution:")

        if len(tensor.shape) == 3:  # [batch_size, seq_len, hidden_dim]
            # NaN count per sample
            nan_per_batch = nan_mask.sum(dim=(1, 2))
            logger.info("   NaN count per batch sample:")
            for i, count in enumerate(nan_per_batch):
                if count > 0:
                    logger.info("     Batch %d: %d NaN values", i, count.item())

            # NaN count per sequence position
            nan_per_seq = nan_mask.sum(dim=(0, 2))
            if nan_per_seq.sum() > 0:
                logger.info("   Sequence positions with NaN values:")
                for i, count in enumerate(nan_per_seq):
                    if count > 0:
                        logger.info("     Position %d: %d NaN values", i, count.item())

    if not all_nan and "mean" in results:
        logger.info("-" * 80)
        logger.info("Valid value statistics:")
        logger.info("   Min: %.6f", results["min"])
        logger.info("   Max: %.6f", results["max"])
        logger.info("   Mean: %.6f", results["mean"])
        logger.info("   Std: %.6f", results["std"])

    logger.info("=" * 80)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check safetensors activation files for NaN values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the safetensors file containing activations",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed NaN distribution statistics",
    )

    args = parser.parse_args()

    # Validate file path
    file_path = Path(args.file)
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return 1

    if not file_path.suffix == ".safetensors":
        logger.warning("File does not have .safetensors extension: %s", file_path)

    logger.info("Loading activations from: %s", file_path)

    # Load the safetensors file
    try:
        tensors = load_file(str(file_path))
    except Exception as e:
        logger.error("Failed to load safetensors file: %s", e)
        return 1

    logger.info("Loaded %d tensor(s) from file", len(tensors))
    logger.info("Available tensors: %s", list(tensors.keys()))
    logger.info("")

    # Analyze each tensor
    all_results = {}
    for tensor_name, tensor in tensors.items():
        results = analyze_nan_values(tensor, name=tensor_name, verbose=args.verbose)
        all_results[tensor_name] = results
        logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    for tensor_name, results in all_results.items():
        status = "❌ ALL NaN" if results["all_nan"] else ("⚠️  HAS NaN" if results["has_nan"] else "✅ OK")
        logger.info("%s: %s", tensor_name, status)
        if results["has_nan"]:
            logger.info(
                "   %.2f%% NaN (%d / %d values)",
                results["nan_percentage"],
                results["nan_count"],
                results["total_elements"],
            )

    logger.info("=" * 80)

    # Return error code if any tensor has NaN
    has_any_nan = any(r["has_nan"] for r in all_results.values())
    return 1 if has_any_nan else 0


if __name__ == "__main__":
    raise SystemExit(main())
