"""Centralized device management utilities for LanguageModel operations.

This module provides shared device handling logic to ensure consistent
device management across the codebase.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mi_crow.language_model.language_model import LanguageModel

logger = logging.getLogger(__name__)


def normalize_device(device: str | torch.device | None) -> str:
    """
    Normalize and validate device specification.
    
    Ensures the device is available and normalizes generic device strings.
    - None → "cpu"
    - "cuda" → "cuda:0" (if available)
    - Validates CUDA/MPS availability
    
    Args:
        device: Device specification as string, torch.device, or None
        
    Returns:
        Normalized device string such as "cpu", "cuda:0", or "mps"
        
    Raises:
        ValueError: If requested device is not available
    """
    if device is None:
        return "cpu"
    
    if isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = str(device)
    
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                "Requested device 'cuda' but CUDA is not available. "
                "Install a CUDA-enabled PyTorch build or use device='cpu'."
            )
        if device_str == "cuda":
            device_str = "cuda:0"
    
    if device_str == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())
        if not mps_available:
            raise ValueError(
                "Requested device 'mps' but MPS is not available. "
                "Ensure PyTorch is built with MPS support or use device='cpu'."
            )
    
    return device_str


def ensure_context_device(lm: LanguageModel) -> torch.device:
    """
    Ensure LanguageModel has valid context.device and return it.
    
    Args:
        lm: LanguageModel instance
        
    Returns:
        torch.device from context
        
    Raises:
        ValueError: If context.device is not properly set
    """
    if not hasattr(lm, "context") or not hasattr(lm.context, "device") or lm.context.device is None:
        raise ValueError(
            "LanguageModel must have context.device set. "
            "Ensure LanguageModel is properly initialized with a device."
        )
    return torch.device(lm.context.device)


def sync_model_to_context_device(lm: LanguageModel) -> None:
    """
    Ensure model is on the device specified by context.device.
    
    Moves the model if there's a mismatch between current location
    and context.device. This is the primary device synchronization
    function that should be called before any model operations.
    
    Args:
        lm: LanguageModel instance with context.device set
        
    Raises:
        ValueError: If context.device is not set
        RuntimeError: If model cannot be moved to target device
    """
    from mi_crow.language_model.utils import get_device_from_model
    
    target_device = ensure_context_device(lm)
    model_device = get_device_from_model(lm.context.model)
    
    if model_device != target_device:
        try:
            lm.context.model = lm.context.model.to(target_device)
            logger.debug(
                "Moved model from %s to %s to match context.device",
                model_device,
                target_device,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to move model from {model_device} to {target_device}: {e}"
            ) from e
