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

