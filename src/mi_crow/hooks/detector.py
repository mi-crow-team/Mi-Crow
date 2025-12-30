from __future__ import annotations

import abc
from typing import Any, TYPE_CHECKING, Dict

import torch

from mi_crow.hooks.hook import Hook, HookType, HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT
from mi_crow.store.store import Store

if TYPE_CHECKING:
    pass


class Detector(Hook):
    """
    Abstract base class for detector hooks that collect metadata during inference.
    
    Detectors can accumulate data across batches and optionally save it to a Store.
    They are designed to observe and record information without modifying activations.
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
        self.metadata: Dict[str, Any] = {}
        self.tensor_metadata: Dict[str, torch.Tensor] = {}

    def _hook_fn(
            self,
            module: torch.nn.Module,
            input: HOOK_FUNCTION_INPUT,
            output: HOOK_FUNCTION_OUTPUT
    ) -> None:
        """
        Internal hook function that collects metadata.
        
        This calls process_activations to allow subclasses to implement
        their specific detection logic.
        
        Args:
            module: The PyTorch module being hooked
            input: Tuple of input tensors to the module
            output: Output tensor(s) from the module
            
        Raises:
            Exception: If process_activations raises an exception
        """
        if not self._enabled:
            return None
        try:
            self.process_activations(module, input, output)
        except Exception as e:
            raise RuntimeError(
                f"Error in detector {self.id} process_activations: {e}"
            ) from e
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
            
        Raises:
            Exception: Subclasses may raise exceptions for invalid inputs or processing errors
        """
        raise NotImplementedError("process_activations must be implemented by subclasses")
