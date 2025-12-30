from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from mi_crow.store.store import Store

if TYPE_CHECKING:
    pass


@dataclass
class AutoencoderContext:
    """Shared context for Autoencoder and its nested components."""

    autoencoder: "Sae"

    # Core SAE parameters
    n_latents: int
    n_inputs: int

    # Language model parameters (shared across hierarchy)
    lm: Optional["LanguageModel"] = None
    lm_layer_signature: Optional[int | str] = None
    model_id: Optional[str] = None

    # Training/experiment metadata
    device: str = 'cpu'
    experiment_name: Optional[str] = None
    run_id: Optional[str] = None

    # Text tracking parameters
    text_tracking_enabled: bool = False
    text_tracking_k: int = 5
    text_tracking_negative: bool = False

    store: Optional[Store] = None

    # Training parameters
    tied: bool = False
    bias_init: float = 0.0
    init_method: str = "kaiming"
