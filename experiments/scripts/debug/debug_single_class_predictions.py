# ruff: noqa
"""
Debug Single-Class Predictions

This script investigates why LPM experiments classify all samples to a single class.
It analyzes prototypes, distances, and prediction logic to identify the root cause.

Usage:
    python debug_single_class_predictions.py \\
        --experiment_dir store/lpm_llama_3b_wgmix_train_wgmix_test_last_token_prefix_layer27_euclidean \\
        --dataset wgmix_test \\
        --num_samples 50
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.models.lpm.lpm import LPM
from mi_crow.datasets import ClassificationDataset

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


def analyze_prototypes(lpm: LPM) -> dict:
    """Analyze LPM prototypes for anomalies."""
    logger.info("=" * 80)
    logger.info("PROTOTYPE ANALYSIS")
    logger.info("=" * 80)

    results = {
        "num_classes": len(lpm.class_labels_),
        "class_labels": lpm.class_labels_,
        "prototype_stats": {},
        "issues": [],
    }

    for i, label in enumerate(lpm.class_labels_):
        proto = lpm.prototypes[i]

        stats = {
            "shape": list(proto.shape),
            "mean": float(proto.mean().item()),
            "std": float(proto.std().item()),
            "min": float(proto.min().item()),
            "max": float(proto.max().item()),
            "norm": float(proto.norm().item()),
        }

        results["prototype_stats"][label] = stats

        logger.info(f"\nPrototype '{label}':")
        logger.info(f"  Shape: {stats['shape']}")
        logger.info(f"  Mean: {stats['mean']:.6f}")
        logger.info(f"  Std: {stats['std']:.6f}")
        logger.info(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        logger.info(f"  L2 Norm: {stats['norm']:.6f}")

        # Check for anomalies
        if stats["std"] < 1e-6:
            results["issues"].append(
                {
                    "type": "ZERO_VARIANCE",
                    "label": label,
                    "details": f"Prototype has near-zero variance (std={stats['std']:.2e})",
                    "severity": "CRITICAL",
                }
            )
            logger.warning("  ⚠️  Near-zero variance detected!")

        if abs(stats["mean"]) > 100:
            results["issues"].append(
                {
                    "type": "EXTREME_VALUES",
                    "label": label,
                    "details": f"Prototype has extreme mean value ({stats['mean']:.2f})",
                    "severity": "HIGH",
                }
            )

    # Check prototype separation
    if len(lpm.prototypes) == 2:
        proto1, proto2 = lpm.prototypes
        distance = torch.dist(proto1, proto2).item()
        relative_distance = distance / ((proto1.norm() + proto2.norm()) / 2).item()

        logger.info("\nPrototype Separation:")
        logger.info(f"  L2 Distance: {distance:.6f}")
        logger.info(f"  Relative Distance: {relative_distance:.6f}")

        results["prototype_separation"] = {
            "l2_distance": float(distance),
            "relative_distance": float(relative_distance),
        }

        if relative_distance < 0.01:
            results["issues"].append(
                {
                    "type": "PROTOTYPES_TOO_CLOSE",
                    "details": f"Prototypes are very close (relative distance={relative_distance:.2e})",
                    "severity": "CRITICAL",
                }
            )
            logger.warning("  ⚠️  Prototypes are extremely close - classification will be poor!")
        elif relative_distance < 0.1:
            results["issues"].append(
                {
                    "type": "PROTOTYPES_CLOSE",
                    "details": f"Prototypes are somewhat close (relative distance={relative_distance:.4f})",
                    "severity": "MEDIUM",
                }
            )
            logger.warning("  ⚠️  Prototypes are close - might affect classification")

    return results


def analyze_predictions(  # noqa: C901
    lpm: LPM,
    predictions_df: pd.DataFrame,
    ground_truth: list[str],
    num_samples: int = 50,
) -> dict:
    """Analyze prediction behavior."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("PREDICTION ANALYSIS")
    logger.info("=" * 80)

    predicted_labels = predictions_df["predicted_class"].tolist()[:num_samples]
    ground_truth = ground_truth[:num_samples]

    # Get distances if available
    has_distances = "distance_unharmful" in predictions_df.columns

    results = {
        "num_samples_analyzed": len(predicted_labels),
        "prediction_distribution": {},
        "accuracy": 0.0,
        "issues": [],
        "sample_analysis": [],
    }

    # Prediction distribution
    unique, counts = np.unique(predicted_labels, return_counts=True)
    for label, count in zip(unique, counts):
        results["prediction_distribution"][label] = int(count)

    logger.info(f"\nPrediction Distribution (first {num_samples} samples):")
    for label, count in results["prediction_distribution"].items():
        pct = 100 * count / len(predicted_labels)
        logger.info(f"  {label}: {count} ({pct:.1f}%)")

    # Check for single-class prediction
    if len(results["prediction_distribution"]) == 1:
        single_class = list(results["prediction_distribution"].keys())[0]
        results["issues"].append(
            {
                "type": "SINGLE_CLASS_PREDICTION",
                "class": single_class,
                "details": f"ALL predictions are '{single_class}'",
                "severity": "CRITICAL",
            }
        )
        logger.warning(f"\n⚠️  ALL PREDICTIONS ARE '{single_class}'!")

    # Accuracy
    correct = sum(1 for pred, true in zip(predicted_labels, ground_truth) if pred == true)
    results["accuracy"] = correct / len(predicted_labels)
    logger.info(f"\nAccuracy (first {num_samples}): {results['accuracy']:.4f}")

    # Analyze individual samples
    logger.info("\nSample-by-sample Analysis (first 10):")
    for i in range(min(10, len(predicted_labels))):
        sample = {
            "idx": i,
            "true_label": ground_truth[i],
            "predicted_label": predicted_labels[i],
            "correct": predicted_labels[i] == ground_truth[i],
        }

        if has_distances:
            sample["distance_unharmful"] = float(predictions_df.iloc[i]["distance_unharmful"])
            sample["distance_harmful"] = float(predictions_df.iloc[i]["distance_harmful"])
            sample["distance_diff"] = sample["distance_unharmful"] - sample["distance_harmful"]

            logger.info(f"\n  Sample {i} (true={ground_truth[i]}):")
            logger.info(f"    Distance to 'unharmful': {sample['distance_unharmful']:.6f}")
            logger.info(f"    Distance to 'harmful': {sample['distance_harmful']:.6f}")
            logger.info(
                f"    Difference: {sample['distance_diff']:.6f} ({'unharmful closer' if sample['distance_diff'] < 0 else 'harmful closer'})"  # noqa: E501
            )
            logger.info(f"    Predicted: {predicted_labels[i]} {'✓' if sample['correct'] else '✗'}")

            # Check for very small differences
            if abs(sample["distance_diff"]) < 1e-3:
                logger.warning("      ⚠️  Very small distance difference!")
        else:
            logger.info(
                f"  Sample {i}: true={ground_truth[i]}, pred={predicted_labels[i]} {'✓' if sample['correct'] else '✗'}"
            )

        results["sample_analysis"].append(sample)

    # Analyze distance distribution if available
    if has_distances:
        logger.info("\nDistance Statistics:")

        dist_unharmful = predictions_df["distance_unharmful"][:num_samples].values
        dist_harmful = predictions_df["distance_harmful"][:num_samples].values
        dist_diff = dist_unharmful - dist_harmful

        logger.info("  Distance to 'unharmful':")
        logger.info(f"    Mean: {dist_unharmful.mean():.6f}")
        logger.info(f"    Std: {dist_unharmful.std():.6f}")
        logger.info(f"    Range: [{dist_unharmful.min():.6f}, {dist_unharmful.max():.6f}]")

        logger.info("  Distance to 'harmful':")
        logger.info(f"    Mean: {dist_harmful.mean():.6f}")
        logger.info(f"    Std: {dist_harmful.std():.6f}")
        logger.info(f"    Range: [{dist_harmful.min():.6f}, {dist_harmful.max():.6f}]")

        logger.info("  Distance difference (unharmful - harmful):")
        logger.info(f"    Mean: {dist_diff.mean():.6f}")
        logger.info(f"    Std: {dist_diff.std():.6f}")
        logger.info(f"    Range: [{dist_diff.min():.6f}, {dist_diff.max():.6f}]")

        # Check if all differences have same sign
        if (dist_diff < 0).all():
            logger.warning("\n  ⚠️  ALL samples are closer to 'unharmful'!")
            results["issues"].append(
                {
                    "type": "ALL_CLOSER_TO_UNHARMFUL",
                    "details": "All distance differences are negative",
                    "severity": "CRITICAL",
                }
            )
        elif (dist_diff > 0).all():
            logger.warning("\n  ⚠️  ALL samples are closer to 'harmful'!")
            results["issues"].append(
                {
                    "type": "ALL_CLOSER_TO_HARMFUL",
                    "details": "All distance differences are positive",
                    "severity": "CRITICAL",
                }
            )

        # Check if differences are very small
        if dist_diff.std() < 1e-3:
            logger.warning("\n  ⚠️  Distance differences have very low variance!")
            results["issues"].append(
                {
                    "type": "LOW_DISTANCE_VARIANCE",
                    "details": f"Distance difference std={dist_diff.std():.2e}",
                    "severity": "HIGH",
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Debug single-class prediction issue")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Experiment directory (e.g., store/lpm_llama_3b_wgmix_train_wgmix_test_...)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Test dataset name",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to analyze (default: 50)",
    )
    args = parser.parse_args()

    experiment_path = Path(args.experiment_dir)
    if not experiment_path.exists():
        logger.error(f"Experiment directory not found: {experiment_path}")
        return 1

    logger.info("=" * 80)
    logger.info("SINGLE-CLASS PREDICTION DEBUG")
    logger.info("=" * 80)
    logger.info(f"Experiment: {experiment_path.name}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples to analyze: {args.num_samples}")
    logger.info("")

    # Load LPM model
    logger.info("Loading LPM model...")
    lpm_models = list((experiment_path / "models").glob("lpm_*.pt"))
    if not lpm_models:
        logger.error("No LPM model found in experiment directory")
        return 1

    lpm_model_path = lpm_models[0]
    logger.info(f"Found model: {lpm_model_path.name}")

    lpm = LPM.load(lpm_model_path)
    logger.info(f"✅ LPM loaded: {len(lpm.class_labels_)} classes")
    logger.info("")

    # Analyze prototypes
    proto_results = analyze_prototypes(lpm)

    # Load predictions
    logger.info("")
    logger.info("Loading predictions...")
    inference_runs = list((experiment_path / "runs").glob("inference_*"))
    if not inference_runs:
        logger.error("No inference run found in experiment directory")
        return 1

    inference_run = inference_runs[0]
    pred_file = inference_run / "predictions.parquet"

    if not pred_file.exists():
        logger.error(f"Predictions file not found: {pred_file}")
        return 1

    predictions_df = pd.read_parquet(pred_file)
    logger.info(f"✅ Loaded {len(predictions_df)} predictions")
    logger.info("")

    # Load ground truth
    logger.info(f"Loading ground truth from {args.dataset}...")
    config = DATASET_CONFIGS[args.dataset]
    dataset = ClassificationDataset.from_parquet(**config)
    ground_truth = [item[config["category_field"]] for item in dataset.iter_items()]
    logger.info(f"✅ Loaded {len(ground_truth)} ground truth labels")

    # Analyze predictions
    pred_results = analyze_predictions(lpm, predictions_df, ground_truth, args.num_samples)

    # Final diagnosis
    logger.info("")
    logger.info("=" * 80)
    logger.info("DIAGNOSIS")
    logger.info("=" * 80)

    all_issues = proto_results["issues"] + pred_results["issues"]
    critical_issues = [i for i in all_issues if i["severity"] == "CRITICAL"]
    high_issues = [i for i in all_issues if i["severity"] == "HIGH"]

    if not all_issues:
        logger.info("✅ No obvious issues detected")
        logger.info("   The model appears to be functioning normally.")
        logger.info("   Low accuracy might be due to:")
        logger.info("   - Insufficient training data")
        logger.info("   - Poor layer selection")
        logger.info("   - Dataset mismatch or corruption")
    else:
        logger.info(f"Found {len(all_issues)} issues:")

        if critical_issues:
            logger.info("\n❌ CRITICAL ISSUES:")
            for issue in critical_issues:
                logger.info(f"  • {issue['type']}: {issue['details']}")

        if high_issues:
            logger.info("\n⚠️  HIGH PRIORITY ISSUES:")
            for issue in high_issues:
                logger.info(f"  • {issue['type']}: {issue['details']}")

        logger.info("\nRECOMMENDATIONS:")

        if any(i["type"] == "PROTOTYPES_TOO_CLOSE" for i in all_issues):
            logger.info("  1. Prototypes are nearly identical:")
            logger.info("     → Check if activations were loaded correctly during training")
            logger.info("     → Verify aggregation method is working properly")
            logger.info("     → Try different layer or aggregation method")

        if any(i["type"] == "SINGLE_CLASS_PREDICTION" for i in all_issues):
            logger.info("  2. All predictions go to single class:")
            logger.info("     → Verify dataset/activation alignment")
            logger.info("     → Check if attention masks match actual sequences")
            logger.info("     → Run verify_activation_alignment.py script")

        if any(i["type"] in ["ALL_CLOSER_TO_UNHARMFUL", "ALL_CLOSER_TO_HARMFUL"] for i in all_issues):
            logger.info("  3. All samples closer to one prototype:")
            logger.info("     → Indicates systematic bias in activations or distances")
            logger.info("     → Check if prefix/tokenization differs between train and test")
            logger.info("     → Verify distance metric implementation")

        if any(i["type"] == "LOW_DISTANCE_VARIANCE" for i in all_issues):
            logger.info("  4. Low variance in distance differences:")
            logger.info("     → Model has low discriminative power")
            logger.info("     → Consider using different layer or aggregation")
            logger.info("     → Check if activations are being normalized incorrectly")

    logger.info("=" * 80)

    return 0 if not critical_issues else 1


if __name__ == "__main__":
    exit(main())
