from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING, List

if TYPE_CHECKING:
    pass


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
    tokenizer: Optional["PreTrainedTokenizerBase"] = None

    # Store
    store: Optional["Store"] = None
    
    # Hook registry: layer_signature -> hook_type -> list of (hook, handle)
    _hook_registry: Dict[str | int, Dict[str, List[tuple["Hook", Any]]]] = field(default_factory=dict)
    # Map hook_id -> (layer_signature, hook_type, hook) for fast lookup
    _hook_id_map: Dict[str, tuple[str | int, str, "Hook"]] = field(default_factory=dict)
