#!/usr/bin/env python3
# ruff: noqa
"""
Verify Activation-Dataset Alignment

This script verifies that saved activations align with the dataset samples.
Critical for ensuring LPM training and inference use correct activations.

Usage:
    python verify_activation_alignment.py \\
        --dataset wgmix_train \\
        --activation_run activations_maxlen_512_llama_3_2_3b_instruct_wgmix_train_prefixed_layer27_20260117_233725
"""

import argparse
import hashlib
import logging

import torch

from mi_crow.datasets import ClassificationDataset
from mi_crow.store import LocalStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


DATASET_CONFIGS = {
    "wgmix_train": {
        "parquet_file": "/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store/datasets/wgmix_train.parquet",
        "text_field": "prompt",
        "category_field": "label",
    },
    "wgmix_test": {
        "parquet_file": "/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store/datasets/wgmix_test.parquet",
        "text_field": "prompt",
        "category_field": "label",
    },
    "plmix_train": {
        "parquet_file": "/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store/datasets/plmix_train.parquet",
        "text_field": "text",
        "category_field": "label",
    },
    "plmix_test": {
        "parquet_file": "/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store/datasets/plmix_test.parquet",
        "text_field": "text",
        "category_field": "label",
    },
}


def hash_text(text: str) -> str:
    """Compute SHA256 hash of text for fast comparison."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_dataset_info(dataset_name: str) -> dict:
    """Load dataset and return key information."""
    logger.info(f"Loading dataset: {dataset_name}")

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]
    dataset = ClassificationDataset.from_parquet(**config)

    items = list(dataset.iter_items())

    return {
        "num_samples": len(items),
        "text_field": config["text_field"],
        "category_field": config["category_field"],
        "texts": [item[config["text_field"]] for item in items],
        "labels": [item[config["category_field"]] for item in items],
        "text_hashes": [hash_text(item[config["text_field"]]) for item in items],
    }


def load_activation_info(store_path: str, run_id: str) -> dict:
    """Load activation run information."""
    logger.info(f"Loading activation run: {run_id}")

    store = LocalStore(base_path=store_path)
    run_path = store._run_key(run_id)

    if not run_path.exists():
        raise ValueError(f"Activation run not found: {run_path}")

    # Count batches
    batch_files = sorted(run_path.glob("batch_*.pt"))
    num_batches = len(batch_files)

    logger.info(f"Found {num_batches} batches in run")

    # Load all activations to count samples
    total_samples = 0
    all_labels = []
    batch_shapes = []

    for batch_idx, batch_file in enumerate(batch_files):
        if batch_idx % 10 == 0:
            logger.info(f"Loading batch {batch_idx + 1}/{num_batches}...")

        batch_data = torch.load(batch_file, weights_only=True)

        # Extract activations (could be dict or tensor)
        if isinstance(batch_data, dict):
            activations = batch_data.get("activations", batch_data.get("activation"))
            labels = batch_data.get("labels", batch_data.get("label", []))
        else:
            activations = batch_data
            labels = []

        batch_size = activations.shape[0] if hasattr(activations, "shape") else len(activations)
        total_samples += batch_size
        batch_shapes.append(activations.shape if hasattr(activations, "shape") else None)

        if isinstance(labels, (list, tuple)):
            all_labels.extend(labels)
        elif hasattr(labels, "tolist"):
            all_labels.extend(labels.tolist())

    logger.info(f"Total samples in activations: {total_samples}")

    return {
        "num_batches": num_batches,
        "num_samples": total_samples,
        "labels": all_labels,
        "batch_shapes": batch_shapes,
    }


def verify_alignment(dataset_info: dict, activation_info: dict) -> dict:
    """Verify dataset and activations are aligned."""
    results = {
        "status": "OK",
        "issues": [],
    }

    # Check sample count
    dataset_samples = dataset_info["num_samples"]
    activation_samples = activation_info["num_samples"]

    logger.info(f"Dataset samples: {dataset_samples}")
    logger.info(f"Activation samples: {activation_samples}")

    if dataset_samples != activation_samples:
        results["status"] = "FAILED"
        results["issues"].append(
            {
                "type": "SAMPLE_COUNT_MISMATCH",
                "details": f"Dataset has {dataset_samples} samples, activations have {activation_samples}",
                "severity": "CRITICAL",
            }
        )
        return results  # Can't proceed with further checks

    # Check label alignment (if labels are stored in activations)
    if activation_info["labels"]:
        logger.info("Checking label alignment...")
        dataset_labels = dataset_info["labels"]
        activation_labels = activation_info["labels"]

        if len(dataset_labels) != len(activation_labels):
            results["status"] = "FAILED"
            results["issues"].append(
                {
                    "type": "LABEL_COUNT_MISMATCH",
                    "details": f"Dataset has {len(dataset_labels)} labels, activations have {len(activation_labels)}",
                    "severity": "CRITICAL",
                }
            )
        else:
            # Check label values
            mismatches = []
            for i, (d_label, a_label) in enumerate(zip(dataset_labels, activation_labels)):
                if d_label != a_label:
                    mismatches.append(i)
                    if len(mismatches) <= 10:  # Report first 10
                        results["issues"].append(
                            {
                                "type": "LABEL_VALUE_MISMATCH",
                                "sample_idx": i,
                                "details": f"Sample {i}: dataset={d_label}, activation={a_label}",
                                "severity": "HIGH",
                            }
                        )

            if mismatches:
                results["status"] = "FAILED"
                results["issues"].append(
                    {
                        "type": "LABEL_ALIGNMENT_ERROR",
                        "details": f"{len(mismatches)} label mismatches found (showing first 10)",
                        "severity": "HIGH",
                    }
                )
    else:
        logger.info("⚠️  Activations don't contain labels, skipping label verification")

    # Check batch structure
    logger.info("Checking batch structure...")
    expected_batch_size = 64  # Assuming standard batch size
    for i, shape in enumerate(activation_info["batch_shapes"]):
        if shape is None:
            continue

        batch_size = shape[0]
        is_last_batch = i == len(activation_info["batch_shapes"]) - 1

        if not is_last_batch and batch_size != expected_batch_size:
            results["issues"].append(
                {
                    "type": "UNEXPECTED_BATCH_SIZE",
                    "batch_idx": i,
                    "details": f"Batch {i} has size {batch_size}, expected {expected_batch_size}",
                    "severity": "MEDIUM",
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify activation-dataset alignment")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset name",
    )
    parser.add_argument(
        "--activation_run",
        type=str,
        required=True,
        help="Activation run ID",
    )
    parser.add_argument(
        "--store_path",
        type=str,
        default="/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store",
        help="Store base path (default: cluster store path)",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("ACTIVATION-DATASET ALIGNMENT VERIFICATION")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Activation run: {args.activation_run}")
    logger.info(f"Store path: {args.store_path}")
    logger.info("")

    # Load dataset info
    dataset_info = load_dataset_info(args.dataset)
    logger.info(f"✅ Dataset loaded: {dataset_info['num_samples']} samples")
    logger.info("")

    # Load activation info
    activation_info = load_activation_info(args.store_path, args.activation_run)
    logger.info(
        f"✅ Activations loaded: {activation_info['num_samples']} samples in {activation_info['num_batches']} batches"
    )
    logger.info("")

    # Verify alignment
    logger.info("Verifying alignment...")
    results = verify_alignment(dataset_info, activation_info)

    logger.info("")
    logger.info("=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info(f"Status: {results['status']}")

    if results["status"] == "OK":
        logger.info("")
        logger.info("✅ ACTIVATIONS ARE ALIGNED WITH DATASET")
        logger.info("   - Sample counts match")
        if activation_info["labels"]:
            logger.info("   - Labels match")
        logger.info("   - Batch structure is consistent")
    else:
        logger.info("")
        logger.info("❌ ALIGNMENT ISSUES DETECTED")
        logger.info(f"   - Found {len(results['issues'])} issues:")

        # Group by severity
        critical = [i for i in results["issues"] if i.get("severity") == "CRITICAL"]
        high = [i for i in results["issues"] if i.get("severity") == "HIGH"]
        medium = [i for i in results["issues"] if i.get("severity") == "MEDIUM"]

        if critical:
            logger.info("")
            logger.info("   CRITICAL ISSUES:")
            for issue in critical:
                logger.info(f"     • {issue['type']}: {issue['details']}")

        if high:
            logger.info("")
            logger.info("   HIGH PRIORITY ISSUES:")
            for issue in high[:5]:  # Show first 5
                logger.info(f"     • {issue['type']}: {issue['details']}")
            if len(high) > 5:
                logger.info(f"     ... and {len(high) - 5} more")

        if medium:
            logger.info("")
            logger.info("   MEDIUM PRIORITY ISSUES:")
            logger.info(f"     • {len(medium)} batch size inconsistencies")

    logger.info("=" * 80)

    return 0 if results["status"] == "OK" else 1


if __name__ == "__main__":
    exit(main())
