"""Model initialization and factory methods."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase

from mi_crow.store.store import Store
from mi_crow.language_model.utils import extract_model_id

if TYPE_CHECKING:
    from mi_crow.language_model.language_model import LanguageModel


def initialize_model_id(
        model: nn.Module,
        provided_model_id: str | None = None
) -> str:
    """
    Initialize model ID for LanguageModel.
    
    Args:
        model: PyTorch model module
        provided_model_id: Optional model ID provided by user
        
    Returns:
        Model ID string
    """
    return extract_model_id(model, provided_model_id)


def create_from_huggingface(
        cls: type["LanguageModel"],
        model_name: str,
        store: Store,
        tokenizer_params: dict | None = None,
        model_params: dict | None = None,
        device: str | torch.device | None = None,
) -> "LanguageModel":
    """
    Load a language model from HuggingFace Hub.
    
    Args:
        cls: LanguageModel class
        model_name: HuggingFace model identifier
        store: Store instance for persistence
        tokenizer_params: Optional tokenizer parameters
        model_params: Optional model parameters
        device: Target device ("cuda", "cpu", "mps"). Model will be moved to this device
            after loading.
    Returns:
        LanguageModel instance
    
    Raises:
        ValueError: If model_name is invalid
        RuntimeError: If model loading fails
    """
    if not model_name or not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(f"model_name must be a non-empty string, got: {model_name!r}")
    
    if store is None:
        raise ValueError("store cannot be None")
    
    if tokenizer_params is None:
        tokenizer_params = {}
    if model_params is None:
        model_params = {}
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_params)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_params)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{model_name}' from HuggingFace. Error: {e}"
        ) from e

    return cls(model, tokenizer, store, device=device)


def create_from_local_torch(
        cls: type["LanguageModel"],
        model_path: str,
        tokenizer_path: str,
        store: Store,
        device: str | torch.device | None = None,
) -> "LanguageModel":
    """
    Load a language model from local HuggingFace paths.
    
    Args:
        cls: LanguageModel class
        model_path: Path to the model directory or file
        tokenizer_path: Path to the tokenizer directory or file
        store: Store instance for persistence
        device: Optional device string or torch.device (defaults to 'cpu' if None)
        
    Returns:
        LanguageModel instance
        
    Raises:
        FileNotFoundError: If model or tokenizer paths don't exist
        RuntimeError: If model loading fails
    """
    if store is None:
        raise ValueError("store cannot be None")
    
    model_path_obj = Path(model_path)
    tokenizer_path_obj = Path(tokenizer_path)
    
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    if not tokenizer_path_obj.exists():
        raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from local paths. "
            f"model_path={model_path!r}, tokenizer_path={tokenizer_path!r}. Error: {e}"
        ) from e

    return cls(model, tokenizer, store, device=device)

