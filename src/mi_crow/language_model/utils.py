"""Utility functions for language model operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def extract_model_id(model: nn.Module, provided_model_id: str | None = None) -> str:
    """
    Extract model ID from model or use provided one.
    
    Args:
        model: PyTorch model module
        provided_model_id: Optional model ID provided by user
        
    Returns:
        Model ID string
        
    Raises:
        ValueError: If model_id cannot be determined
    """
    if provided_model_id is not None:
        return provided_model_id
    
    if hasattr(model, 'config') and hasattr(model.config, 'name_or_path'):
        return model.config.name_or_path.replace("/", "_")
    
    return model.__class__.__name__


def get_device_from_model(model: nn.Module) -> torch.device:
    """
    Get the device from model parameters.
    
    Args:
        model: PyTorch model module
        
    Returns:
        Device where model parameters are located, or CPU if no parameters
    """
    try:
        first_param = next(model.parameters(), None)
        return first_param.device if first_param is not None else torch.device("cpu")
    except (TypeError, AttributeError):
        return torch.device("cpu")


def move_tensors_to_device(
        tensors: dict[str, torch.Tensor],
        device: torch.device
) -> dict[str, torch.Tensor]:
    """
    Move dictionary of tensors to specified device.
    
    Args:
        tensors: Dictionary of tensor name to tensor
        device: Target device
        
    Returns:
        Dictionary with tensors moved to device
    """
    return {k: v.to(device) for k, v in tensors.items()}


def extract_logits_from_output(output: any) -> torch.Tensor:
    """
    Extract logits tensor from model output.
    
    Handles various output formats:
    - Objects with 'logits' attribute (e.g., HuggingFace model outputs)
    - Tuples (takes first element)
    - Direct tensors
    
    Args:
        output: Model output (various formats)
        
    Returns:
        Logits tensor
        
    Raises:
        ValueError: If logits cannot be extracted from output
    """
    if hasattr(output, 'logits'):
        return output.logits
    elif isinstance(output, tuple) and len(output) > 0:
        return output[0]
    elif isinstance(output, torch.Tensor):
        return output
    else:
        raise ValueError(f"Unable to extract logits from output type: {type(output)}")

