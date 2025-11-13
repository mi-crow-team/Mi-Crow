from typing import Any, TYPE_CHECKING
import torch

from amber.hooks.detector import Detector
from amber.hooks.hook import HookType, HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT

if TYPE_CHECKING:
    from torch import nn


class LayerActivationDetector(Detector):
    """
    Detector hook that captures and saves activations during inference.
    
    This detector extracts activations from layer outputs and stores them
    for later use (e.g., saving to disk, further analysis).
    """

    def __init__(
            self,
            layer_signature: str | int,
            hook_id: str | None = None
    ):
        """
        Initialize the activation saver detector.
        
        Args:
            layer_signature: Layer to capture activations from
            hook_id: Unique identifier for this hook
        """
        super().__init__(
            hook_type=HookType.FORWARD,
            hook_id=hook_id,
            store=None
        )
        # Set layer_signature attribute
        self.layer_signature = layer_signature
        self.captured_activations: torch.Tensor | None = None

    def process_activations(
            self,
            module: torch.nn.Module,
            input: HOOK_FUNCTION_INPUT,
            output: HOOK_FUNCTION_OUTPUT
    ) -> None:
        """
        Extract and store activations from output.
        
        Handles various output types:
        - Plain tensors
        - Tuples/lists of tensors (takes first tensor)
        - Objects with last_hidden_state attribute (e.g., HuggingFace outputs)
        """
        tensor = None
        if isinstance(output, torch.Tensor):
            tensor = output
        elif isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    tensor = item
                    break
        else:
            # Try common HF output objects
            if hasattr(output, "last_hidden_state"):
                maybe = getattr(output, "last_hidden_state")
                if isinstance(maybe, torch.Tensor):
                    tensor = maybe

        if tensor is not None:
            tensor_cpu = tensor.detach().to("cpu")
            # Store current batch's tensor (overwrites previous)
            self._tensor_metadata['activations'] = tensor_cpu
            # Accumulate for saving multiple batches
            if 'activations' not in self._tensor_batches:
                self._tensor_batches['activations'] = []
            self._tensor_batches['activations'].append(tensor_cpu)

    def get_captured(self) -> torch.Tensor | None:
        """
        Get the captured activations from the current batch.
        
        Returns:
            The captured activation tensor from the current batch or None if no activations captured yet
        """
        return self._tensor_metadata.get('activations')

    def clear_captured(self) -> None:
        """Clear captured activations for current batch."""
        self._tensor_metadata.pop('activations', None)
