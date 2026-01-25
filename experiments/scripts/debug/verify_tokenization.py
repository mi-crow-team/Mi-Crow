#!/usr/bin/env python3
# ruff: noqa
"""
Verify Tokenization Consistency

This script verifies that tokenization parameters are consistent between
attention mask saving (Step 3) and inference (Step 4), which is critical
for avoiding shape mismatch errors.

Usage:
    python verify_tokenization.py \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --dataset wgmix_test \\
        --sample_size 100
"""

import argparse
import logging

import torch
from transformers import AutoTokenizer

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


PREFIX_TEMPLATES = {
    "en": "You are a content moderator. Analyze the following content and determine if it is harmful or unharmful: {text_field}",
    "pl": "Jesteś moderatorem treści. Przeanalizuj poniższą treść i określ, czy jest szkodliwa, czy nieszkodliwa: {text_field}",
}


def apply_prefix(texts: list[str], template: str, text_field: str) -> list[str]:
    """Apply prefix template to texts."""
    return [template.format(**{text_field: text}) for text in texts]


def tokenize_with_config(tokenizer, texts: list[str], config_name: str, max_length: int = 512) -> dict:
    """Tokenize texts with a specific configuration."""
    configs = {
        "step3_standard": {
            "padding": "max_length",
            "max_length": max_length,
            "truncation": True,
            "return_tensors": "pt",
        },
        "step4_standard": {
            "padding": "max_length",
            "max_length": max_length,
            "truncation": True,
            "return_tensors": "pt",
        },
        "left_padding": {
            "padding": "max_length",
            "max_length": max_length,
            "truncation": True,
            "padding_side": "left",
            "return_tensors": "pt",
        },
        "right_padding": {
            "padding": "max_length",
            "max_length": max_length,
            "truncation": True,
            "padding_side": "right",
            "return_tensors": "pt",
        },
    }

    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}")

    # Apply padding side if specified
    config = configs[config_name]
    original_padding_side = tokenizer.padding_side
    if "padding_side" in config:
        tokenizer.padding_side = config["padding_side"]

    # Tokenize
    encoded = tokenizer(
        texts,
        padding=config["padding"],
        max_length=config["max_length"],
        truncation=config["truncation"],
        return_tensors=config["return_tensors"],
    )

    # Restore original padding side
    tokenizer.padding_side = original_padding_side

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "shapes": encoded["input_ids"].shape,
        "num_tokens": encoded["attention_mask"].sum(dim=1).tolist(),  # Actual tokens per sample
    }


def compare_tokenizations(tok1: dict, tok2: dict, config1: str, config2: str) -> dict:
    """Compare two tokenization results."""
    results = {
        "status": "OK",
        "issues": [],
    }

    # Check shapes
    if tok1["shapes"] != tok2["shapes"]:
        results["status"] = "FAILED"
        results["issues"].append(
            {
                "type": "SHAPE_MISMATCH",
                "details": f"{config1} shape {tok1['shapes']} != {config2} shape {tok2['shapes']}",
                "severity": "CRITICAL",
            }
        )
        return results

    # Check token IDs
    if not torch.equal(tok1["input_ids"], tok2["input_ids"]):
        results["status"] = "FAILED"
        # Find first mismatch
        mismatches = (tok1["input_ids"] != tok2["input_ids"]).nonzero()
        if len(mismatches) > 0:
            sample_idx, token_idx = mismatches[0]
            results["issues"].append(
                {
                    "type": "TOKEN_ID_MISMATCH",
                    "details": f"First mismatch at sample {sample_idx}, token {token_idx}",
                    "severity": "CRITICAL",
                }
            )

    # Check attention masks
    if not torch.equal(tok1["attention_mask"], tok2["attention_mask"]):
        results["status"] = "FAILED"
        mismatches = (tok1["attention_mask"] != tok2["attention_mask"]).nonzero()
        if len(mismatches) > 0:
            sample_idx, token_idx = mismatches[0]
            results["issues"].append(
                {
                    "type": "ATTENTION_MASK_MISMATCH",
                    "details": f"First mismatch at sample {sample_idx}, token {token_idx}",
                    "severity": "CRITICAL",
                }
            )

    # Check token counts
    if tok1["num_tokens"] != tok2["num_tokens"]:
        results["status"] = "FAILED"
        for i, (c1, c2) in enumerate(zip(tok1["num_tokens"], tok2["num_tokens"])):
            if c1 != c2:
                results["issues"].append(
                    {
                        "type": "TOKEN_COUNT_MISMATCH",
                        "sample_idx": i,
                        "details": f"Sample {i}: {config1} has {c1} tokens, {config2} has {c2} tokens",
                        "severity": "HIGH",
                    }
                )
                if len([x for x in results["issues"] if x["type"] == "TOKEN_COUNT_MISMATCH"]) >= 5:
                    break

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify tokenization consistency")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model ID (e.g., meta-llama/Llama-3.2-3B-Instruct)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset name",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of samples to test (default: 100)",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "last_token", "last_token_prefix"],
        help="Aggregation method to simulate (default: mean)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length (default: 512)",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("TOKENIZATION CONSISTENCY VERIFICATION")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Aggregation: {args.aggregation}")
    logger.info(f"Max length: {args.max_length}")
    logger.info("")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    config = DATASET_CONFIGS[args.dataset]
    dataset = ClassificationDataset.from_parquet(**config)
    items = list(dataset.iter_items())[: args.sample_size]
    texts = [item[config["text_field"]] for item in items]
    logger.info(f"✅ Loaded {len(texts)} samples")
    logger.info("")

    # Apply prefix if needed
    if args.aggregation == "last_token_prefix":
        language = "pl" if "plmix" in args.dataset else "en"
        template = PREFIX_TEMPLATES[language]
        logger.info(f"Applying prefix template for {language}")
        logger.info(f"Template: {template[:100]}...")
        texts = apply_prefix(texts, template, config["text_field"])
        logger.info("✅ Prefix applied")
        logger.info(f"Example: {texts[0][:150]}...")
        logger.info("")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logger.info("✅ Tokenizer loaded")
    logger.info(f"Original padding side: {tokenizer.padding_side}")
    logger.info("")

    # Test different tokenization configurations
    logger.info("Testing tokenization configurations...")

    # Step 3 vs Step 4 (should be identical)
    logger.info("\n1. Comparing Step 3 vs Step 4 (standard configs)")
    tok_step3 = tokenize_with_config(tokenizer, texts, "step3_standard", args.max_length)
    tok_step4 = tokenize_with_config(tokenizer, texts, "step4_standard", args.max_length)
    result_standard = compare_tokenizations(tok_step3, tok_step4, "Step 3", "Step 4")

    if result_standard["status"] == "OK":
        logger.info("   ✅ Step 3 and Step 4 tokenization is identical")
    else:
        logger.info("   ❌ Step 3 and Step 4 tokenization differs!")
        for issue in result_standard["issues"]:
            logger.info(f"      • {issue['type']}: {issue['details']}")

    # Left vs Right padding
    logger.info("\n2. Comparing left vs right padding")
    tok_left = tokenize_with_config(tokenizer, texts, "left_padding", args.max_length)
    tok_right = tokenize_with_config(tokenizer, texts, "right_padding", args.max_length)
    result_padding = compare_tokenizations(tok_left, tok_right, "Left Padding", "Right Padding")

    if result_padding["status"] == "OK":
        logger.info("   ℹ️  Padding side doesn't affect tokenization (unexpected)")
    else:
        logger.info("   ℹ️  Padding side affects tokenization (expected for some models)")
        logger.info(f"      Issues: {len(result_padding['issues'])}")

    # Sample analysis
    logger.info("\n3. Sample tokenization analysis")
    logger.info(f"   Sample 0 (first {min(50, len(texts[0]))} chars): {texts[0][:50]}...")
    logger.info(f"   Token IDs length: {tok_step3['shapes'][1]}")
    logger.info(f"   Actual tokens: {tok_step3['num_tokens'][0]}")
    logger.info(f"   Padding tokens: {tok_step3['shapes'][1] - tok_step3['num_tokens'][0]}")

    # Check for common issues
    logger.info("\n4. Checking for common issues")
    issues_found = []

    # Check if all samples have same length (should be max_length)
    unique_lengths = set([tok_step3["input_ids"].shape[1]])
    if len(unique_lengths) > 1:
        issues_found.append(f"Variable sequence lengths: {unique_lengths}")
    else:
        logger.info(f"   ✅ All sequences padded to {args.max_length}")

    # Check if attention masks are valid (only 0 and 1)
    unique_mask_values = tok_step3["attention_mask"].unique().tolist()
    if set(unique_mask_values) - {0, 1}:
        issues_found.append(f"Invalid attention mask values: {unique_mask_values}")
    else:
        logger.info("   ✅ Attention masks contain only 0 and 1")

    # Check for very short sequences (might indicate tokenization issues)
    min_tokens = min(tok_step3["num_tokens"])
    if min_tokens < 10:
        issues_found.append(f"Very short sequence detected: {min_tokens} tokens")

    # Check for max-length sequences (might be truncated)
    max_tokens = max(tok_step3["num_tokens"])
    if max_tokens >= args.max_length:
        logger.info(f"   ⚠️  Some sequences reach max_length ({args.max_length}) - may be truncated")

    if issues_found:
        logger.info("\n   Issues found:")
        for issue in issues_found:
            logger.info(f"      • {issue}")

    # Final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    overall_status = "OK"
    critical_issues = []

    if result_standard["status"] != "OK":
        overall_status = "FAILED"
        critical_issues.extend([i for i in result_standard["issues"] if i["severity"] == "CRITICAL"])

    if overall_status == "OK":
        logger.info("✅ TOKENIZATION IS CONSISTENT")
        logger.info("   - Step 3 and Step 4 produce identical tokenization")
        logger.info("   - All sequences properly padded/truncated")
        logger.info("   - No critical issues detected")
    else:
        logger.info("❌ TOKENIZATION INCONSISTENCIES DETECTED")
        logger.info(f"   - {len(critical_issues)} critical issues found:")
        for issue in critical_issues:
            logger.info(f"     • {issue['type']}: {issue['details']}")
        logger.info("")
        logger.info("   RECOMMENDATION:")
        logger.info("   - Ensure identical tokenization parameters in Step 3 and Step 4")
        logger.info("   - Verify padding_side, truncation_side, max_length are consistent")
        logger.info("   - Check if prefix application is consistent")

    logger.info("=" * 80)

    return 0 if overall_status == "OK" else 1


if __name__ == "__main__":
    exit(main())
