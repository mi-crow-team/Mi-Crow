"""
Test dataset reproducibility with stratification and seeding.

This script loads the same dataset twice with identical parameters
and verifies that the results are identical (same rows, same order).

Usage:
    uv run python -m experiments.scripts.debug.test_dataset_reproducibility
"""

from __future__ import annotations

from mi_crow.datasets import ClassificationDataset
from mi_crow.store import LocalStore
from mi_crow.utils import get_logger

logger = get_logger(__name__)


def main() -> int:
    """Load dataset twice with same seed and verify they're identical."""

    store = LocalStore("store")
    seed = 42
    limit = 100

    logger.info("=" * 80)
    logger.info("Testing dataset reproducibility with stratification")
    logger.info("=" * 80)

    # Load dataset first time
    logger.info("\n[1/2] Loading dataset (first time)...")
    dataset1 = ClassificationDataset.from_huggingface(
        repo_id="allenai/wildguardmix",
        store=store,
        name="wildguardtrain",
        split="train",
        text_field="prompt",
        category_field="prompt_harm_label",
        limit=limit,
        stratify_by="prompt_harm_label",
        stratify_seed=seed,
        drop_na=True,
    )
    logger.info(f"Dataset 1: loaded {len(dataset1)} samples")

    # Load dataset second time with same parameters
    logger.info("\n[2/2] Loading dataset (second time)...")
    dataset2 = ClassificationDataset.from_huggingface(
        repo_id="allenai/wildguardmix",
        store=store,
        name="wildguardtrain",
        split="train",
        text_field="prompt",
        category_field="prompt_harm_label",
        limit=limit,
        stratify_by="prompt_harm_label",
        stratify_seed=seed,
        drop_na=True,
    )
    logger.info(f"Dataset 2: loaded {len(dataset2)} samples")

    # Verify they have the same length
    logger.info("\n" + "=" * 80)
    logger.info("Verification Results")
    logger.info("=" * 80)

    if len(dataset1) != len(dataset2):
        logger.error(f"❌ FAILED: Different lengths! {len(dataset1)} vs {len(dataset2)}")
        return 1

    logger.info(f"✓ Lengths match: {len(dataset1)} samples")

    # Verify row-by-row that content is identical
    mismatches = 0
    for i in range(len(dataset1)):
        item1 = dataset1[i]
        item2 = dataset2[i]

        if item1["text"] != item2["text"]:
            logger.error(f"❌ Row {i}: text mismatch!")
            logger.error(f"  Dataset1: {item1['text'][:100]}...")
            logger.error(f"  Dataset2: {item2['text'][:100]}...")
            mismatches += 1

        if item1.get("prompt_harm_label") != item2.get("prompt_harm_label"):
            logger.error(f"❌ Row {i}: label mismatch!")
            logger.error(f"  Dataset1: {item1.get('prompt_harm_label')}")
            logger.error(f"  Dataset2: {item2.get('prompt_harm_label')}")
            mismatches += 1

    if mismatches > 0:
        logger.error(f"\n❌ FAILED: Found {mismatches} mismatches!")
        return 1

    logger.info("✓ All rows match (text and labels)")

    # Verify order is identical by checking first and last few samples
    logger.info("\nFirst 3 samples (labels):")
    for i in range(min(3, len(dataset1))):
        label = dataset1[i].get("prompt_harm_label")
        logger.info(f"  [{i}] {label}")

    logger.info("\nLast 3 samples (labels):")
    for i in range(max(0, len(dataset1) - 3), len(dataset1)):
        label = dataset1[i].get("prompt_harm_label")
        logger.info(f"  [{i}] {label}")

    # Check label distribution
    categories1 = dataset1.get_categories()
    logger.info(f"\nCategories found: {categories1}")

    # Count distribution
    label_counts = {}
    for i in range(len(dataset1)):
        label = dataset1[i].get("prompt_harm_label")
        label_counts[label] = label_counts.get(label, 0) + 1

    logger.info("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        percentage = (count / len(dataset1)) * 100
        logger.info(f"  {label}: {count} ({percentage:.1f}%)")

    logger.info("\n" + "=" * 80)
    logger.info("✅ SUCCESS: Datasets are IDENTICAL!")
    logger.info("=" * 80)
    logger.info("\nConclusion:")
    logger.info("  - Same seed produces same rows in same order")
    logger.info("  - Stratification is reproducible")
    logger.info("  - Drop NA is deterministic")
    logger.info("  - Dataset creation is fully reproducible ✓")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
