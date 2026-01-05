"""
Prepare and cache all datasets for experiments.

This script downloads and saves the following datasets to local store:
- WildGuardMix Train (5000 samples, stratified): For English LPM prototypes and probe training
- WildGuardMix Test (full): For English evaluation (In-Distribution)
- PL Mix Train: For Polish LPM prototypes and probe training
- PL Mix Test: For Polish evaluation

All datasets are saved to store/datasets/ with proper stratification and preprocessing.

Usage:
    uv run python -m experiments.scripts.prepare_datasets --seed 42

SLURM:
    sbatch slurm/prepare_datasets.sh
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
    if hasattr(dataset, "_ds") and "subcategory" in dataset._ds.column_names:
        subcategories = dataset._ds["subcategory"]
        counter = Counter(subcategories)

        total = len(subcategories)
        # Sort by count (descending) then by name
        for subcategory, count in sorted(counter.items(), key=lambda x: (-x[1], str(x[0]))):
            percentage = (count / total * 100) if total > 0 else 0
            logger.info("     %s: %d (%.2f%%)", subcategory, count, percentage)
    else:
        logger.warning("   Subcategory column not found in dataset")


def prepare_wgmix_train(seed: int) -> None:
    """Prepare WildGuardMix Train dataset (5000 samples, stratified)."""
    logger.info("=" * 80)
    logger.info("Preparing WildGuardMix Train (stratified, 5000 samples)")
    logger.info("=" * 80)

    t0 = perf_counter()
    store = LocalStore(base_path="store/datasets/wgmix_train")

    dataset = ClassificationDataset.from_huggingface(
        repo_id="allenai/wildguardmix",
        store=store,
        name="wildguardtrain",
        split="train",
        text_field="prompt",
        category_field="prompt_harm_label",
        limit=5_000,
        stratify_by="prompt_harm_label",
        stratify_seed=seed,
        drop_na=True,
    )

    elapsed = perf_counter() - t0
    logger.info("✅ WildGuardMix Train ready: %d samples (%.2fs)", len(dataset), elapsed)
    logger.info("   Location: %s/datasets/", store.base_path)

    log_label_distribution(dataset, "prompt_harm_label", "WildGuardMix Train")
    log_subcategory_distribution(dataset, "WildGuardMix Train")
    logger.info("")


def prepare_wgmix_test(seed: int) -> None:
    """Prepare WildGuardMix Test dataset (full)."""
    logger.info("=" * 80)
    logger.info("Preparing WildGuardMix Test (full dataset)")
    logger.info("=" * 80)

    t0 = perf_counter()
    store = LocalStore(base_path="store/datasets/wgmix_test")

    dataset = ClassificationDataset.from_huggingface(
        repo_id="allenai/wildguardmix",
        store=store,
        name="wildguardtest",
        split="test",
        text_field="prompt",
        category_field="prompt_harm_label",
        drop_na=True,
    )

    elapsed = perf_counter() - t0
    logger.info("✅ WildGuardMix Test ready: %d samples (%.2fs)", len(dataset), elapsed)
    logger.info("   Location: %s/datasets/", store.base_path)

    log_label_distribution(dataset, "prompt_harm_label", "WildGuardMix Test")
    logger.info("")


def prepare_plmix_train(seed: int) -> None:
    """Prepare PL Mix Train dataset from CSV."""
    logger.info("=" * 80)
    logger.info("Preparing PL Mix Train (from CSV)")
    logger.info("=" * 80)

    t0 = perf_counter()
    store = LocalStore(base_path="store/datasets/plmix_train")
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
    logger.info("✅ PL Mix Train ready: %d samples (%.2fs)", len(dataset), elapsed)
    logger.info("   Location: %s/datasets/", store.base_path)

    log_label_distribution(dataset, "text_harm_label", "PL Mix Train")
    logger.info("")


def prepare_plmix_test(seed: int) -> None:
    """Prepare PL Mix Test dataset from CSV."""
    logger.info("=" * 80)
    logger.info("Preparing PL Mix Test (from CSV)")
    logger.info("=" * 80)

    t0 = perf_counter()
    store = LocalStore(base_path="store/datasets/plmix_test")
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
    logger.info("✅ PL Mix Test ready: %d samples (%.2fs)", len(dataset), elapsed)
    logger.info("   Location: %s/datasets/", store.base_path)

    log_label_distribution(dataset, "text_harm_label", "PL Mix Test")
    logger.info("")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and cache all datasets for experiments")
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

    logger.info("🚀 Starting dataset preparation")
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
        logger.info("✅ All datasets prepared successfully!")
        logger.info("   Total time: %.2f seconds", total_elapsed)
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error("❌ Dataset preparation failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
