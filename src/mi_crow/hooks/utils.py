"""Utility functions for hook implementations."""

from __future__ import annotations

from typing import Any

import torch

from mi_crow.hooks.hook import HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT


def extract_tensor_from_input(input: HOOK_FUNCTION_INPUT) -> torch.Tensor | None:
    """
    Extract the first tensor from input sequence.
    
    Handles various input formats:
    - Direct tensor in first position
    - Tuple/list of tensors in first position
    - Empty or None inputs
    
    Args:
        input: Input sequence (tuple/list of tensors)
        
    Returns:
        First tensor found, or None if no tensor found
    """
    if not input or len(input) == 0:
        return None
    
    first_item = input[0]
    if isinstance(first_item, torch.Tensor):
        return first_item
    
    if isinstance(first_item, (tuple, list)):
        for item in first_item:
            if isinstance(item, torch.Tensor):
                return item
    
    return None


def extract_tensor_from_output(output: HOOK_FUNCTION_OUTPUT) -> torch.Tensor | None:
    """
    Extract tensor from output (handles various output types).
    
    Handles various output formats:
    - Plain tensors
    - Tuples/lists of tensors (takes first tensor)
    - Objects with last_hidden_state attribute (e.g., HuggingFace outputs)
    - None outputs
    
    Args:
        output: Output from module (tensor, tuple, or object with attributes)
        
    Returns:
        First tensor found, or None if no tensor found
    """
    if output is None:
        return None
    
    if isinstance(output, torch.Tensor):
        return output
    
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    
    # Try common HuggingFace output objects
    if hasattr(output, "last_hidden_state"):
        maybe = getattr(output, "last_hidden_state")
        if isinstance(maybe, torch.Tensor):
            return maybe
    
    return None


def apply_modification_to_output(
    output: HOOK_FUNCTION_OUTPUT,
    modified_tensor: torch.Tensor
) -> None:
    """
    Apply a modified tensor to an output object in-place.
    
    Handles various output formats:
    - Plain tensors: modifies the tensor directly (in-place)
    - Tuples/lists of tensors: replaces first tensor
    - Objects with last_hidden_state attribute: sets last_hidden_state
    
    Args:
        output: Output object to modify
        modified_tensor: Modified tensor to apply
    """
    if output is None:
        return
    
    if isinstance(output, torch.Tensor):
        # Ensure modified_tensor is on the same device as output
        if modified_tensor.device != output.device:
            modified_tensor = modified_tensor.to(output.device)
        output.data.copy_(modified_tensor.data)
        return
    
    if isinstance(output, (tuple, list)):
        for i, item in enumerate(output):
            if isinstance(item, torch.Tensor):
                # Ensure modified_tensor is on the same device and dtype as item
                if modified_tensor.device != item.device or modified_tensor.dtype != item.dtype:
                    modified_tensor = modified_tensor.to(device=item.device, dtype=item.dtype)
                if isinstance(output, tuple):
                    # For tuples, we need to modify in-place, so copy the data
                    item.data.copy_(modified_tensor.data)
                else:
                    # For lists, we can replace the item
                    output[i] = modified_tensor
                break
        return
    
    if hasattr(output, "last_hidden_state"):
        original_tensor = output.last_hidden_state
        if isinstance(original_tensor, torch.Tensor):
            # Ensure modified_tensor is on the same device as original
            if modified_tensor.device != original_tensor.device:
                modified_tensor = modified_tensor.to(original_tensor.device)
        output.last_hidden_state = modified_tensor
        return

