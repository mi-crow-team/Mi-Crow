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

from datasets import load_from_disk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default store path (can be overridden via command line)
DEFAULT_STORE_PATH = "/mnt/evafs/groups/mi2lab/hkowalski/Mi-Crow/store"


def inspect_dataset(dataset_path: Path, dataset_name: str) -> dict:
    """Inspect a single dataset and return its properties.

    Args:
        dataset_path: Path to the dataset directory
        dataset_name: Name of the dataset for logging

    Returns:
        Dictionary with dataset properties
    """
    logger.info("=" * 80)
    logger.info(f"Inspecting: {dataset_name}")
    logger.info("=" * 80)
    logger.info(f"Path: {dataset_path}")

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
        # Load the dataset
        ds = load_from_disk(str(dataset_path))

        # Get basic info
        num_rows = ds.num_rows
        columns = ds.column_names
        features = ds.features

        logger.info(f"✅ Successfully loaded dataset")
        logger.info(f"   Rows: {num_rows}")
        logger.info(f"   Columns: {columns}")
        logger.info("")

        # Display feature types
        logger.info("Column Types:")
        for col_name, feature in features.items():
            logger.info(f"   {col_name}: {feature}")
        logger.info("")

        # Display first 3 rows
        logger.info("First 3 rows:")
        for i in range(min(3, num_rows)):
            logger.info(f"   Row {i}:")
            row = ds[i]
            for col_name in columns:
                value = row[col_name]
                # Truncate long values
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                logger.info(f"     {col_name}: {value!r}")
            logger.info("")

        return {
            "status": "OK",
            "path": str(dataset_path),
            "num_rows": num_rows,
            "columns": columns,
            "features": {k: str(v) for k, v in features.items()},
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

    logger.info("🔍 Dataset Inspector")
    logger.info(f"Store path: {store_path}")
    logger.info("")

    # Define datasets to inspect
    datasets = {
        "wgmix_train": store_path / "datasets/wgmix_train/datasets",
        "wgmix_test": store_path / "datasets/wgmix_test/datasets",
        "plmix_train": store_path / "datasets/plmix_train/datasets",
        "plmix_test": store_path / "datasets/plmix_test/datasets",
    }

    results = {}
    for dataset_name, dataset_path in datasets.items():
        results[dataset_name] = inspect_dataset(dataset_path, dataset_name)
        logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    for dataset_name, result in results.items():
        status = result["status"]
        if status == "OK":
            logger.info(f"✅ {dataset_name}: {result['num_rows']} rows, columns: {result['columns']}")
        elif status == "NOT_FOUND":
            logger.info(f"❌ {dataset_name}: Directory not found")
        elif status == "NO_ARROW_FILES":
            logger.info(f"❌ {dataset_name}: No Arrow files")
        else:
            logger.info(f"❌ {dataset_name}: Load failed - {result.get('error', 'unknown error')}")

    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
