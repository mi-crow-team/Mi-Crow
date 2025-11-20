"""Utility functions for SAE tensor handling and hook operations."""

from __future__ import annotations

from typing import Any, Tuple

import torch
from torch import nn

from amber.hooks.hook import HookType, HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT
from amber.hooks.utils import extract_tensor_from_input, extract_tensor_from_output


def extract_activation_tensor(
    hook_type: HookType,
    inputs: HOOK_FUNCTION_INPUT,
    output: HOOK_FUNCTION_OUTPUT
) -> tuple[torch.Tensor | None, Any]:
    """
    Extract activation tensor from hook inputs or output based on hook type.
    
    Handles various formats:
    - FORWARD hooks: extract from output (tensor, tuple, list, or object with attributes)
    - PRE_FORWARD hooks: extract from inputs (first element or nested structure)
    
    Args:
        hook_type: Type of hook (FORWARD or PRE_FORWARD)
        inputs: Input sequence to the module
        output: Output from the module (None for PRE_FORWARD hooks)
        
    Returns:
        Tuple of (extracted_tensor, original_value)
        - extracted_tensor: The tensor if found, None otherwise
        - original_value: The original input/output value for reconstruction
    """
    if hook_type == HookType.FORWARD:
        tensor = extract_tensor_from_output(output)
        return tensor, output
    else:  # PRE_FORWARD
        tensor = extract_tensor_from_input(inputs)
        return tensor, inputs


def reshape_for_sae(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], bool]:
    """
    Reshape tensor to 2D format required by SAE (batch * seq_len, hidden_dim).
    
    Args:
        tensor: Input tensor of shape [batch, seq_len, hidden] or [batch, hidden]
        
    Returns:
        Tuple of (reshaped_tensor, original_shape, needs_reshape)
        - reshaped_tensor: 2D tensor [batch * seq_len, hidden] or [batch, hidden]
        - original_shape: Original tensor shape
        - needs_reshape: Whether reshaping was needed
    """
    original_shape = tensor.shape
    needs_reshape = len(original_shape) > 2
    
    if needs_reshape:
        # Flatten: (batch, seq_len, hidden) -> (batch * seq_len, hidden)
        tensor = tensor.reshape(-1, original_shape[-1])
    
    return tensor, original_shape, needs_reshape


def reshape_from_sae(
    tensor: torch.Tensor,
    original_shape: tuple[int, ...],
    needs_reshape: bool
) -> torch.Tensor:
    """
    Reshape tensor back to original shape after SAE processing.
    
    Args:
        tensor: 2D tensor from SAE [batch * seq_len, hidden] or [batch, hidden]
        original_shape: Original shape before SAE processing
        needs_reshape: Whether reshaping is needed
        
    Returns:
        Tensor reshaped to original_shape if needed, otherwise unchanged
    """
    if needs_reshape:
        return tensor.reshape(original_shape)
    return tensor


def reconstruct_hook_output(
    hook_type: HookType,
    reconstructed_tensor: torch.Tensor,
    original_value: Any
) -> Any:
    """
    Reconstruct hook output/input in the same format as the original.
    
    For FORWARD hooks: replaces tensor in output (tensor, tuple, list, or object)
    For PRE_FORWARD hooks: replaces tensor in inputs tuple
    
    Args:
        hook_type: Type of hook (FORWARD or PRE_FORWARD)
        reconstructed_tensor: The reconstructed tensor from SAE
        original_value: Original output (FORWARD) or inputs (PRE_FORWARD)
        
    Returns:
        Reconstructed value in the same format as original_value
    """
    if hook_type == HookType.FORWARD:
        return _reconstruct_forward_output(reconstructed_tensor, original_value)
    else:  # PRE_FORWARD
        return _reconstruct_pre_forward_inputs(reconstructed_tensor, original_value)


def _reconstruct_forward_output(
    reconstructed: torch.Tensor,
    output: Any
) -> Any:
    """Reconstruct FORWARD hook output in original format."""
    if isinstance(output, torch.Tensor):
        return reconstructed
    
    if isinstance(output, (tuple, list)):
        # Replace first tensor in tuple/list
        result = list(output)
        for i, item in enumerate(result):
            if isinstance(item, torch.Tensor):
                result[i] = reconstructed
                break
        return tuple(result) if isinstance(output, tuple) else result
    
    # For objects with attributes, try to set last_hidden_state
    if hasattr(output, "last_hidden_state"):
        output.last_hidden_state = reconstructed
    return output


def _reconstruct_pre_forward_inputs(
    reconstructed: torch.Tensor,
    inputs: HOOK_FUNCTION_INPUT
) -> tuple:
    """Reconstruct PRE_FORWARD hook inputs tuple."""
    result = list(inputs)
    if len(result) > 0:
        result[0] = reconstructed
    return tuple(result)

