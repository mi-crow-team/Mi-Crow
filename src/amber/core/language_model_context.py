from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING, List
from pathlib import Path

if TYPE_CHECKING:
    from transformers import AutoTokenizer
    from torch import nn
    from amber.store import Store
    from amber.core.language_model import LanguageModel


@dataclass
class LanguageModelContext:
    """Shared context for LanguageModel and its components."""

    language_model: "LanguageModel"
    model_id: Optional[str] = None

    # Tokenizer parameters
    tokenizer_params: Optional[Dict[str, Any]] = None
    model_params: Optional[Dict[str, Any]] = None

    # Device and computation
    device: str = 'cpu'
    dtype: Optional[str] = None

    # Model references (set after initialization)
    model: Optional["nn.Module"] = None
    tokenizer: Optional["AutoTokenizer"] = None

    # Store and activation tracking
    store: Optional["Store"] = None
