import abc
from typing import Any, TYPE_CHECKING, List, Dict
import torch

from amber.hooks.hook import Hook, HookType, HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT
from amber.store import Store

if TYPE_CHECKING:
    pass


class Detector(Hook):
    """
    Abstract base class for detector hooks that collect metadata during inference.
    
    Detectors can accumulate data across batches and optionally save it to a Store.
    """

    def __init__(
            self,
            hook_type: HookType | str = HookType.FORWARD,
            hook_id: str | None = None,
            store: Store | None = None,
            layer_signature: str | int | None = None
    ):
        """
        Initialize a detector hook.
        
        Args:
            hook_type: Type of hook (HookType.FORWARD or HookType.PRE_FORWARD)
            hook_id: Unique identifier
            store: Optional Store for saving metadata
            layer_signature: Layer to attach to (optional, for compatibility)
        """
        super().__init__(layer_signature=layer_signature, hook_type=hook_type, hook_id=hook_id)
        self.store = store
        self._metadata: Dict[str, Any] = {}
        self._tensor_metadata: Dict[str, torch.Tensor] = {}
        # Internal accumulator for saving multiple batches
        self._tensor_batches: Dict[str, List[torch.Tensor]] = {}

    def _hook_fn(
            self,
            module: torch.nn.Module,
            input: HOOK_FUNCTION_INPUT,
            output: HOOK_FUNCTION_OUTPUT
    ) -> None:
        """
        Internal hook function that collects metadata.
        
        This calls process_activations and collect_metadata.
        """
        if not self._enabled:
            return None
        self.process_activations(module, input, output)
        return None

    @abc.abstractmethod
    def process_activations(
            self,
            module: torch.nn.Module,
            input: HOOK_FUNCTION_INPUT,
            output: HOOK_FUNCTION_OUTPUT
    ) -> None:
        """
        Process activations from the hooked layer.
        
        This is where detector-specific logic goes (e.g., tracking top activations,
        computing statistics, etc.).
        
        Args:
            module: The PyTorch module being hooked
            input: Tuple of input tensors to the module
            output: Output tensor(s) from the module
        """
        pass
