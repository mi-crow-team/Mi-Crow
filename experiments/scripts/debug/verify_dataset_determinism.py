#!/usr/bin/env python3
# ruff: noqa
"""
Verify Dataset Determinism

This script verifies that datasets load in the same order every time,
which is critical for ensuring dataset/activation alignment.

Usage:
    python verify_dataset_determinism.py --dataset wgmix_train --num_loads 10
"""

import argparse
import hashlib
import logging

from mi_crow.datasets import ClassificationDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def hash_text(text: str) -> str:
    """Compute SHA256 hash of text for fast comparison."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_dataset_snapshot(dataset_name: str) -> dict:
    """Load dataset and return snapshot of key properties."""
    logger.info(f"Loading dataset: {dataset_name}")

    dataset_configs = {
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

    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    config = dataset_configs[dataset_name]
    dataset = ClassificationDataset.from_parquet(**config)

    # Extract key properties
    items = list(dataset.iter_items())

    snapshot = {
        "num_samples": len(items),
        "text_field": config["text_field"],
        "category_field": config["category_field"],
        "sample_hashes": [hash_text(item[config["text_field"]]) for item in items],
        "labels": [item[config["category_field"]] for item in items],
        "first_text": items[0][config["text_field"]][:100] if items else "",
        "last_text": items[-1][config["text_field"]][:100] if items else "",
    }

    return snapshot


def compare_snapshots(snapshots: list[dict]) -> dict:
    """Compare multiple dataset snapshots for consistency."""
    if not snapshots:
        return {"status": "ERROR", "message": "No snapshots to compare"}

    results = {
        "status": "OK",
        "num_loads": len(snapshots),
        "num_samples": snapshots[0]["num_samples"],
        "issues": [],
    }

    # Check sample count consistency
    sample_counts = [s["num_samples"] for s in snapshots]
    if len(set(sample_counts)) > 1:
        results["status"] = "FAILED"
        results["issues"].append(
            {
                "type": "SAMPLE_COUNT_MISMATCH",
                "details": f"Sample counts vary: {set(sample_counts)}",
            }
        )
        return results

    # Check hash consistency (sample order)
    num_samples = snapshots[0]["num_samples"]
    for sample_idx in range(num_samples):
        hashes = [s["sample_hashes"][sample_idx] for s in snapshots]
        if len(set(hashes)) > 1:
            results["status"] = "FAILED"
            results["issues"].append(
                {
                    "type": "SAMPLE_ORDER_MISMATCH",
                    "sample_idx": sample_idx,
                    "details": f"Sample {sample_idx} has different text across loads",
                    "hashes": list(set(hashes)),
                }
            )
            # Only report first 10 mismatches
            if len([i for i in results["issues"] if i["type"] == "SAMPLE_ORDER_MISMATCH"]) >= 10:
                break

    # Check label consistency
    for sample_idx in range(num_samples):
        labels = [s["labels"][sample_idx] for s in snapshots]
        if len(set(labels)) > 1:
            results["status"] = "FAILED"
            results["issues"].append(
                {
                    "type": "LABEL_MISMATCH",
                    "sample_idx": sample_idx,
                    "details": f"Sample {sample_idx} has different labels across loads",
                    "labels": list(set(labels)),
                }
            )
            # Only report first 10 mismatches
            if len([i for i in results["issues"] if i["type"] == "LABEL_MISMATCH"]) >= 10:
                break

    # Check first/last text consistency
    first_texts = [s["first_text"] for s in snapshots]
    if len(set(first_texts)) > 1:
        results["status"] = "FAILED"
        results["issues"].append(
            {
                "type": "FIRST_TEXT_MISMATCH",
                "details": "First sample text differs across loads",
            }
        )

    last_texts = [s["last_text"] for s in snapshots]
    if len(set(last_texts)) > 1:
        results["status"] = "FAILED"
        results["issues"].append(
            {
                "type": "LAST_TEXT_MISMATCH",
                "details": "Last sample text differs across loads",
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify dataset loading determinism")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["wgmix_train", "wgmix_test", "plmix_train", "plmix_test"],
        help="Dataset to verify",
    )
    parser.add_argument(
        "--num_loads",
        type=int,
        default=10,
        help="Number of times to load dataset (default: 10)",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("DATASET DETERMINISM VERIFICATION")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Number of loads: {args.num_loads}")
    logger.info("")

    # Load dataset multiple times
    snapshots = []
    for i in range(args.num_loads):
        logger.info(f"Load {i + 1}/{args.num_loads}...")
        snapshot = load_dataset_snapshot(args.dataset)
        snapshots.append(snapshot)

    logger.info("")
    logger.info("Comparing snapshots...")
    results = compare_snapshots(snapshots)

    logger.info("")
    logger.info("=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info(f"Status: {results['status']}")
    logger.info(f"Number of loads: {results['num_loads']}")
    logger.info(f"Number of samples: {results['num_samples']}")

    if results["status"] == "OK":
        logger.info("")
        logger.info("✅ DATASET IS DETERMINISTIC")
        logger.info("   - Sample order is identical across all loads")
        logger.info("   - Text content matches character-by-character")
        logger.info("   - Labels are consistent")
    else:
        logger.info("")
        logger.info("❌ DATASET IS NON-DETERMINISTIC")
        logger.info(f"   - Found {len(results['issues'])} issues:")
        for issue in results["issues"]:
            logger.info(f"     • {issue['type']}: {issue['details']}")

    logger.info("=" * 80)

    return 0 if results["status"] == "OK" else 1


if __name__ == "__main__":
    exit(main())
