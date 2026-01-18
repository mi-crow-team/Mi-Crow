"""
Behavior with Large CUDA Activations:

When `aggregate_activations_batch` is called with large activation tensors
residing on CUDA (GPU), the function processes each sequence in the batch
individually by calling `aggregate_activations_single` in a Python loop.
For each sequence, a much smaller aggregated activation (of shape `[d_model]`)
is produced, and the results are stacked into a final tensor of shape
`[batch_size, d_model]`.

Memory Usage and CUDA Space:

- The original large activation tensor (`activations`) remains in CUDA memory
for the duration of the function call and is not automatically freed by this
function.
- The function does not modify or delete the input tensor; it only reads from
it and constructs a new, smaller tensor for the output.
- The memory occupied by the original activations will only be freed when there
are no more references to the tensor in your Python program and after Python's
garbage collector releases it. If you want to explicitly free CUDA memory, you
can use `del activations` followed by `torch.cuda.empty_cache()`, but this is
not handled by the aggregation function itself.
- The output tensor (aggregated activations) will also reside on CUDA if
the input was on CUDA.

In summary, the original large activations will continue to occupy CUDA
memory until you explicitly delete them or they go out of scope and are
garbage collected. The aggregation function itself does not free or move the
original activations.
---
Activation aggregation helpers for LLM experiments

This module provides utilities for aggregating the sequence dimension of LLM activations using attention masks.

Aggregation types:
- 'last': Selects the last non-special token (where attention_mask == 1) for each sequence.
- 'mean': Computes the mean of activations over non-special tokens (where attention_mask == 1) for each sequence.

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

AggregationType = Literal["last", "mean"]


def aggregate_activations_single(
    activations: torch.Tensor, attention_mask: torch.Tensor, agg: AggregationType = "last"
) -> torch.Tensor:
    """
    Aggregate activations for a single sequence.
    Args:
        activations: [seq_len, d_model]
        attention_mask: [seq_len]
        agg: 'last' or 'mean'
    Returns:
        [d_model]
    """
    if activations.ndim != 2 or attention_mask.ndim != 1:
        raise ValueError("Expected activations [seq_len, d_model] and attention_mask [seq_len]")
    valid_idx = attention_mask.nonzero(as_tuple=True)[0]
    if len(valid_idx) == 0:
        # All tokens are special
        logger.warning(
            "All tokens are special (attention_mask==0 everywhere). "
            f"Fallback for agg='{agg}': using last token (for 'last') or mean over all (for 'mean')."
        )
        if agg == "last":
            # Use the very last token
            return activations[-1]
        elif agg == "mean":
            # Mean over all tokens
            return activations.mean(dim=0)
        else:
            raise ValueError(f"Unknown aggregation type: {agg}")
    if agg == "last":
        return activations[valid_idx[-1]]
    elif agg == "mean":
        return activations[valid_idx].mean(dim=0)
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
        agg: 'last' or 'mean'
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
