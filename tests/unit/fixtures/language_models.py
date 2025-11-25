"""LanguageModel fixtures for testing."""

from __future__ import annotations

from typing import Optional
import torch

from amber.language_model.language_model import LanguageModel
from amber.store.store import Store
from .models import create_mock_model
from .tokenizers import create_mock_tokenizer


def create_language_model_from_mock(
    store: Store,
    model_type: str = "simple",
    vocab_size: int = 1000,
    hidden_size: int = 128,
    model_id: Optional[str] = None,
) -> LanguageModel:
    """
    Create a LanguageModel from mock model and tokenizer.
    
    Args:
        store: Store instance
        model_type: Type of model to create
        vocab_size: Vocabulary size
        hidden_size: Hidden layer size
        model_id: Optional model ID
        
    Returns:
        LanguageModel instance
    """
    model = create_mock_model(model_type=model_type, vocab_size=vocab_size, hidden_size=hidden_size)
    tokenizer = create_mock_tokenizer(vocab_size=vocab_size)
    
    return LanguageModel(
        model=model,
        tokenizer=tokenizer,
        store=store,
        model_id=model_id,
    )


def create_language_model(
    store: Store,
    model: Optional[torch.nn.Module] = None,
    tokenizer: Optional = None,
    model_id: Optional[str] = None,
) -> LanguageModel:
    """
    Create a LanguageModel with provided or default model/tokenizer.
    
    Args:
        store: Store instance
        model: Optional PyTorch model (defaults to mock)
        tokenizer: Optional tokenizer (defaults to mock)
        model_id: Optional model ID
        
    Returns:
        LanguageModel instance
    """
    if model is None:
        model = create_mock_model()
    if tokenizer is None:
        tokenizer = create_mock_tokenizer()
    
    return LanguageModel(
        model=model,
        tokenizer=tokenizer,
        store=store,
        model_id=model_id,
    )

