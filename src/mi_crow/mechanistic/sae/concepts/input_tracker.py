from typing import TYPE_CHECKING, Sequence

from mi_crow.utils import get_logger

if TYPE_CHECKING:
    from mi_crow.language_model.language_model import LanguageModel

logger = get_logger(__name__)


class InputTracker:
    """
    Simple listener that saves input texts before tokenization.
    
    This is a singleton per LanguageModel instance. It's used as a listener
    during inference to capture texts before they are tokenized. SAE hooks
    can then access these texts to track top activating texts for their neurons.
    """

    def __init__(
            self,
            language_model: "LanguageModel",
    ) -> None:
        """
        Initialize InputTracker.
        
        Args:
            language_model: Language model instance
        """
        self.language_model = language_model
        
        # Flag to control whether to save inputs
        self._enabled: bool = False
        
        # Runtime state - only stores texts
        self._current_texts: list[str] = []

    @property
    def enabled(self) -> bool:
        """Whether input tracking is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable input tracking."""
        self._enabled = True

    def disable(self) -> None:
        """Disable input tracking."""
        self._enabled = False

    def reset(self) -> None:
        """Reset stored texts."""
        self._current_texts.clear()

    def set_current_texts(self, texts: Sequence[str]) -> None:
        """
        Set the current batch of texts being processed.
        
        This is called by LanguageModel._inference() before tokenization
        if tracking is enabled.
        """
        if self._enabled:
            self._current_texts = list(texts)

    def get_current_texts(self) -> list[str]:
        """Get the current batch of texts."""
        return self._current_texts.copy()

