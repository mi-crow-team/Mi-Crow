from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING, List, Set

if TYPE_CHECKING:
    from mi_crow.language_model.language_model import LanguageModel
    from torch import nn
    from transformers import PreTrainedTokenizerBase
    from mi_crow.hooks.hook import Hook
    from mi_crow.store.store import Store


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

    model: Optional["nn.Module"] = None
    tokenizer: Optional["PreTrainedTokenizerBase"] = None
    store: Optional["Store"] = None
    special_token_ids: Optional[Set[int]] = None

    _hook_registry: Dict[str | int, Dict[str, List[tuple["Hook", Any]]]] = field(default_factory=dict)
    _hook_id_map: Dict[str, tuple[str | int, str, "Hook"]] = field(default_factory=dict)
