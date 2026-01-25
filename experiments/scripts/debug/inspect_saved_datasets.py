#!/usr/bin/env python3
# ruff: noqa
"""
Inspect Saved Datasets

This script inspects all saved datasets to verify their structure and contents.
It checks what columns are actually stored in the Arrow files and displays
sample data to diagnose any potential data corruption or overwrites.

Usage:
    python inspect_saved_datasets.py --store /path/to/store
"""

import argparse
import logging
from pathlib import Path

from mi_crow.datasets import ClassificationDataset
from mi_crow.store import LocalStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default store path (can be overridden via command line)
DEFAULT_STORE_PATH = "/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store"

# Dataset configurations matching prepare_datasets.py
DATASET_CONFIGS = {
    "wgmix_train": {
        "store_path": "datasets/wgmix_train",
        "text_field": "prompt",
        "category_field": ["prompt_harm_label", "subcategory"],
    },
    "wgmix_test": {
        "store_path": "datasets/wgmix_test",
        "text_field": "prompt",
        "category_field": ["prompt_harm_label", "subcategory"],
    },
    "plmix_train": {
        "store_path": "datasets/plmix_train",
        "text_field": "text",
        "category_field": "text_harm_label",
    },
    "plmix_test": {
        "store_path": "datasets/plmix_test",
        "text_field": "text",
        "category_field": "text_harm_label",
    },
}


def inspect_dataset(dataset_name: str, store_base_path: str) -> dict:
    """Inspect a single dataset and return its properties.

    Args:
        dataset_name: Name of the dataset (e.g., 'wgmix_train')
        store_base_path: Base path to the store directory

    Returns:
        Dictionary with dataset properties
    """
    if dataset_name not in DATASET_CONFIGS:
        logger.error(f"❌ Unknown dataset: {dataset_name}")
        return {"status": "UNKNOWN_DATASET", "dataset_name": dataset_name}

    config = DATASET_CONFIGS[dataset_name]
    store = LocalStore(base_path=str(Path(store_base_path) / config["store_path"]))
    dataset_path = Path(store.base_path) / "datasets"

    logger.info("=" * 80)
    logger.info(f"Inspecting: {dataset_name}")
    logger.info("=" * 80)
    logger.info(f"Store path: {store.base_path}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Text field: {config['text_field']}")
    logger.info(f"Category field: {config['category_field']}")
    logger.info("")

    # Check if dataset exists
    if not dataset_path.exists():
        logger.error("❌ Dataset directory not found!")
        return {"status": "NOT_FOUND", "path": str(dataset_path)}

    # Check for Arrow files
    arrow_files = list(dataset_path.glob("*.arrow"))
    if not arrow_files:
        logger.error("❌ No Arrow files found in directory!")
        return {"status": "NO_ARROW_FILES", "path": str(dataset_path)}

    logger.info(f"Found {len(arrow_files)} Arrow file(s)")

    try:
        # Load using ClassificationDataset.from_disk (same as actual usage)
        dataset = ClassificationDataset.from_disk(
            store=store,
            text_field=config["text_field"],
            category_field=config["category_field"],
        )

        # Get basic info
        num_rows = len(dataset)

        logger.info(f"✅ Successfully loaded ClassificationDataset")
        logger.info(f"   Rows: {num_rows}")
        logger.info("")

        # Display first 3 rows using ClassificationDataset API
        logger.info("First 3 rows (via ClassificationDataset API):")
        for i in range(min(3, num_rows)):
            logger.info(f"   Row {i}:")
            item = dataset[i]
            logger.info(f"     Item keys: {list(item.keys())}")
            for key, value in item.items():
                # Truncate long values
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                logger.info(f"     {key}: {value!r}")
            logger.info("")

        # Get item structure from first row
        sample_item = dataset[0] if num_rows > 0 else {}
        item_keys = list(sample_item.keys())

        logger.info("Item structure returned by ClassificationDataset:")
        logger.info(f"   Keys: {item_keys}")
        logger.info(f"   Note: 'text' is normalized from '{config['text_field']}'")
        logger.info("")

        return {
            "status": "OK",
            "path": str(dataset_path),
            "num_rows": num_rows,
            "item_keys": item_keys,
            "text_field_original": config["text_field"],
            "category_field": config["category_field"],
        }

    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}", exc_info=True)
        return {"status": "LOAD_FAILED", "path": str(dataset_path), "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Inspect saved datasets")
    parser.add_argument(
        "--store",
        type=str,
        default=DEFAULT_STORE_PATH,
        help=f"Store base path (default: {DEFAULT_STORE_PATH})",
    )
    args = parser.parse_args()

    store_path = Path(args.store)

    logger.info("🔍 Dataset Inspector (using ClassificationDataset.from_disk)")
    logger.info(f"Store base path: {store_path}")
    logger.info("")

    # Inspect all configured datasets
    dataset_names = list(DATASET_CONFIGS.keys())

    results = {}
    for dataset_name in dataset_names:
        results[dataset_name] = inspect_dataset(dataset_name, str(store_path))
        logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    for dataset_name, result in results.items():
        status = result["status"]
        if status == "OK":
            item_keys = result["item_keys"]
            logger.info(f"✅ {dataset_name}: {result['num_rows']} rows")
            logger.info(f"     Item keys: {item_keys}")
            logger.info(f"     Text field '{result['text_field_original']}' → 'text' (normalized)")
            logger.info(f"     Category field: {result['category_field']}")
        elif status == "NOT_FOUND":
            logger.info(f"❌ {dataset_name}: Directory not found")
        elif status == "NO_ARROW_FILES":
            logger.info(f"❌ {dataset_name}: No Arrow files")
        elif status == "UNKNOWN_DATASET":
            logger.info(f"❌ {dataset_name}: Unknown dataset configuration")
        else:
            logger.info(f"❌ {dataset_name}: Load failed - {result.get('error', 'unknown error')}")

    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
