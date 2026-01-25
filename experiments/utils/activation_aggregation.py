"""
Activation aggregation helpers for LLM experiments

This module provides utilities for aggregating the sequence dimension of LLM activations using attention masks.

Aggregation types:
- 'last': Selects the last non-special token (where attention_mask == 1) for each sequence.
- 'mean': Computes the mean of activations over non-special tokens (where attention_mask == 1) for each sequence.
- 'max': Computes the maximum value across the sequence for each dimension over non-special tokens (where attention_mask == 1).

Functions are provided for both single activation (2D) and batch activations (3D).

Args:
    activations: torch.Tensor
        - For batch: shape [batch_size, seq_len, d_model]
        - For single: shape [seq_len, d_model]
    attention_mask: torch.Tensor
        - For batch: shape [batch_size, seq_len]
        - For single: shape [seq_len]

Returns:
    torch.Tensor: Aggregated activations
        - For batch: [batch_size, d_model]
        - For single: [d_model]
"""

from typing import Literal

import torch

from mi_crow.utils import get_logger

logger = get_logger(__name__)

AggregationType = Literal["last", "mean", "max"]


def aggregate_activations_single(
    activations: torch.Tensor, attention_mask: torch.Tensor, agg: AggregationType = "last"
) -> torch.Tensor:
    """
    Aggregate activations for a single sequence.
    Args:
        activations: [seq_len, d_model]
        attention_mask: [seq_len]
        agg: 'last', 'mean', or 'max'
    Returns:
        [d_model]
    """
    if activations.ndim != 2 or attention_mask.ndim != 1:
        raise ValueError("Expected activations [seq_len, d_model] and attention_mask [seq_len]")
    valid_idx = attention_mask.nonzero(as_tuple=True)[0]
    if len(valid_idx) == 0:
        logger.warning(
            "All tokens are special (attention_mask==0 everywhere). "
            f"Fallback for agg='{agg}': using last token (for 'last'), mean over all (for 'mean'), or max over all (for 'max')."
        )
        if agg == "last":
            return activations[-1]
        elif agg == "mean":
            return activations.mean(dim=0)
        elif agg == "max":
            return activations.max(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation type: {agg}")
    if agg == "last":
        return activations[valid_idx[-1]]
    elif agg == "mean":
        return activations[valid_idx].mean(dim=0)
    elif agg == "max":
        return activations[valid_idx].max(dim=0)[0]
    else:
        raise ValueError(f"Unknown aggregation type: {agg}")


def aggregate_activations_batch(
    activations: torch.Tensor, attention_mask: torch.Tensor, agg: AggregationType = "last"
) -> torch.Tensor:
    """
    Aggregate activations for a batch of sequences.
    Args:
        activations: [batch_size, seq_len, d_model]
        attention_mask: [batch_size, seq_len]
        agg: 'last', 'mean', or 'max'
    Returns:
        [batch_size, d_model]
    """
    if activations.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Expected activations [batch_size, seq_len, d_model] and attention_mask [batch_size, seq_len]")
    batch_size, seq_len, d_model = activations.shape
    result = []
    for i in range(batch_size):
        result.append(aggregate_activations_single(activations[i], attention_mask[i], agg=agg))
    return torch.stack(result, dim=0)
