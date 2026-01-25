"""
Prepare and cache all datasets for experiments (NEW VERSIONS for testing).

This script is identical to prepare_datasets.py but saves to different paths
(new_wgmix_train, new_wgmix_test, etc.) for comparison and debugging purposes.

Usage:
    uv run python -m experiments.scripts.debug.new_prepare_datasets --seed 42
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

from mi_crow.datasets import ClassificationDataset
from mi_crow.store import LocalStore
from mi_crow.utils import get_logger, set_seed

logger = get_logger(__name__)


def log_label_distribution(dataset: ClassificationDataset, label_field: str, dataset_name: str) -> None:
    """Log the distribution of labels in the dataset."""
    logger.info("   Label distribution (%s):", label_field)

    # Get all labels from the dataset
    labels = [item[label_field] for item in dataset.iter_items()]
    counter = Counter(labels)

    total = len(labels)
    for label, count in sorted(counter.items()):
        percentage = (count / total * 100) if total > 0 else 0
        logger.info("     %s: %d (%.2f%%)", label, count, percentage)


def log_subcategory_distribution(dataset: ClassificationDataset, dataset_name: str) -> None:
    """Log the distribution of subcategories in the dataset."""
    logger.info("   Subcategory distribution:")

    # Access underlying HuggingFace dataset directly to get subcategory column
    try:
        if hasattr(dataset, "_ds"):
            # Check if column exists in the dataset
            if "subcategory" not in dataset._ds.column_names:
                logger.warning("   Subcategory column not available (only text and label columns were saved)")
                return

            subcategories = dataset._ds["subcategory"]
            counter = Counter(subcategories)

            total = len(subcategories)
            # Sort by count (descending) then by name
            for subcategory, count in sorted(counter.items(), key=lambda x: (-x[1], str(x[0]))):
                percentage = (count / total * 100) if total > 0 else 0
                logger.info("     %s: %d (%.2f%%)", subcategory, count, percentage)
        else:
            logger.warning("   Cannot access underlying dataset")
    except Exception as e:
        logger.warning("   Failed to get subcategory distribution: %s", e)


def prepare_wgmix_train(seed: int) -> None:
    """Prepare WildGuardMix Train dataset (5000 samples, stratified)."""
    logger.info("=" * 80)
    logger.info("Preparing NEW WildGuardMix Train (stratified, 5000 samples)")
    logger.info("=" * 80)

    t0 = perf_counter()
    store = LocalStore(base_path="store/datasets/new_wgmix_train")

    dataset = ClassificationDataset.from_huggingface(
        repo_id="allenai/wildguardmix",
        store=store,
        name="wildguardtrain",
        split="train",
        text_field="prompt",
        category_field=["prompt_harm_label", "subcategory"],  # Include subcategory to preserve it
        limit=5_000,
        stratify_by="prompt_harm_label",
        stratify_seed=seed,
        drop_na=True,
    )

    elapsed = perf_counter() - t0
    logger.info("✅ NEW WildGuardMix Train ready: %d samples (%.2fs)", len(dataset), elapsed)
    logger.info("   Location: %s/datasets/", store.base_path)

    log_label_distribution(dataset, "prompt_harm_label", "NEW WildGuardMix Train")
    log_subcategory_distribution(dataset, "NEW WildGuardMix Train")
    logger.info("")


def prepare_wgmix_test(seed: int) -> None:
    """Prepare WildGuardMix Test dataset (full)."""
    logger.info("=" * 80)
    logger.info("Preparing NEW WildGuardMix Test (full dataset)")
    logger.info("=" * 80)

    t0 = perf_counter()
    store = LocalStore(base_path="store/datasets/new_wgmix_test")

    dataset = ClassificationDataset.from_huggingface(
        repo_id="allenai/wildguardmix",
        store=store,
        name="wildguardtest",
        split="test",
        text_field="prompt",
        category_field=["prompt_harm_label", "subcategory"],
        drop_na=True,
    )

    elapsed = perf_counter() - t0
    logger.info("✅ NEW WildGuardMix Test ready: %d samples (%.2fs)", len(dataset), elapsed)
    logger.info("   Location: %s/datasets/", store.base_path)

    log_label_distribution(dataset, "prompt_harm_label", "NEW WildGuardMix Test")
    logger.info("")


def prepare_plmix_train(seed: int) -> None:
    """Prepare PL Mix Train dataset from CSV."""
    logger.info("=" * 80)
    logger.info("Preparing NEW PL Mix Train (from CSV)")
    logger.info("=" * 80)

    t0 = perf_counter()
    store = LocalStore(base_path="store/datasets/new_plmix_train")
    csv_path = Path("data/pl_mix_train.csv")

    if not csv_path.exists():
        logger.error("❌ CSV file not found: %s", csv_path)
        raise FileNotFoundError(f"PL Mix Train CSV not found: {csv_path}")

    dataset = ClassificationDataset.from_csv(
        source=csv_path,
        store=store,
        text_field="text",
        category_field="text_harm_label",
        stratify_by="text_harm_label",
        stratify_seed=seed,
        drop_na=True,
    )

    elapsed = perf_counter() - t0
    logger.info("✅ NEW PL Mix Train ready: %d samples (%.2fs)", len(dataset), elapsed)
    logger.info("   Location: %s/datasets/", store.base_path)

    log_label_distribution(dataset, "text_harm_label", "NEW PL Mix Train")
    logger.info("")


def prepare_plmix_test(seed: int) -> None:
    """Prepare PL Mix Test dataset from CSV."""
    logger.info("=" * 80)
    logger.info("Preparing NEW PL Mix Test (from CSV)")
    logger.info("=" * 80)

    t0 = perf_counter()
    store = LocalStore(base_path="store/datasets/new_plmix_test")
    csv_path = Path("data/pl_mix_test.csv")

    if not csv_path.exists():
        logger.error("❌ CSV file not found: %s", csv_path)
        raise FileNotFoundError(f"PL Mix Test CSV not found: {csv_path}")

    dataset = ClassificationDataset.from_csv(
        source=csv_path,
        store=store,
        text_field="text",
        category_field="text_harm_label",
        drop_na=True,
    )

    elapsed = perf_counter() - t0
    logger.info("✅ NEW PL Mix Test ready: %d samples (%.2fs)", len(dataset), elapsed)
    logger.info("   Location: %s/datasets/", store.base_path)

    log_label_distribution(dataset, "text_harm_label", "NEW PL Mix Test")
    logger.info("")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and cache all datasets (NEW versions)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratification")
    parser.add_argument(
        "--skip-wgmix-train",
        action="store_true",
        help="Skip WildGuardMix Train preparation",
    )
    parser.add_argument(
        "--skip-wgmix-test",
        action="store_true",
        help="Skip WildGuardMix Test preparation",
    )
    parser.add_argument(
        "--skip-plmix-train",
        action="store_true",
        help="Skip PL Mix Train preparation",
    )
    parser.add_argument(
        "--skip-plmix-test",
        action="store_true",
        help="Skip PL Mix Test preparation",
    )

    args = parser.parse_args()

    script_t0 = perf_counter()
    set_seed(args.seed)

    logger.info("🚀 Starting NEW dataset preparation (for debugging/comparison)")
    logger.info("   Seed: %d", args.seed)
    logger.info("")

    try:
        if not args.skip_wgmix_train:
            prepare_wgmix_train(args.seed)

        if not args.skip_wgmix_test:
            prepare_wgmix_test(args.seed)

        if not args.skip_plmix_train:
            prepare_plmix_train(args.seed)

        if not args.skip_plmix_test:
            prepare_plmix_test(args.seed)

        total_elapsed = perf_counter() - script_t0
        logger.info("=" * 80)
        logger.info("✅ All NEW datasets prepared successfully!")
        logger.info("   Total time: %.2f seconds", total_elapsed)
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error("❌ NEW dataset preparation failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
