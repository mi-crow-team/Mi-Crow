from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mi_crow.hooks.detector import Detector
from mi_crow.hooks.hook import HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT, HookType
from mi_crow.hooks.utils import extract_tensor_from_output

if TYPE_CHECKING:
    pass


class LayerActivationDetector(Detector):
    """
    Detector hook that captures and saves activations during inference.

    This detector extracts activations from layer outputs and stores them
    for later use (e.g., saving to disk, further analysis).
    """

    def __init__(self, layer_signature: str | int, hook_id: str | None = None, target_dtype: torch.dtype | None = None):
        """
        Initialize the activation saver detector.

        Args:
            layer_signature: Layer to capture activations from
            hook_id: Unique identifier for this hook
            target_dtype: Optional dtype to convert activations to before storing

        Raises:
            ValueError: If layer_signature is None
        """
        if layer_signature is None:
            raise ValueError("layer_signature cannot be None for LayerActivationDetector")

        super().__init__(hook_type=HookType.FORWARD, hook_id=hook_id, store=None, layer_signature=layer_signature)
        self.target_dtype = target_dtype

    def process_activations(
        self, module: torch.nn.Module, input: HOOK_FUNCTION_INPUT, output: HOOK_FUNCTION_OUTPUT
    ) -> None:
        """
        Extract and store activations from output.

        Handles various output types:
        - Plain tensors
        - Tuples/lists of tensors (takes first tensor)
        - Objects with last_hidden_state attribute (e.g., HuggingFace outputs)

        Args:
            module: The PyTorch module being hooked
            input: Tuple of input tensors to the module
            output: Output tensor(s) from the module

        Raises:
            RuntimeError: If tensor extraction or storage fails
        """
        try:
            tensor = extract_tensor_from_output(output)

            if tensor is not None:
                if tensor.is_cuda:
                    tensor_cpu = tensor.detach().to("cpu", non_blocking=True)
                else:
                    tensor_cpu = tensor.detach()

                if self.target_dtype is not None:
                    tensor_cpu = tensor_cpu.to(self.target_dtype)

                self.tensor_metadata["activations"] = tensor_cpu
                self.metadata["activations_shape"] = tuple(tensor_cpu.shape)
        except Exception as e:
            layer_sig = str(self.layer_signature) if self.layer_signature is not None else "unknown"
            raise RuntimeError(
                f"Error extracting activations in LayerActivationDetector {self.id} (layer={layer_sig}): {e}"
            ) from e

    def get_captured(self) -> torch.Tensor | None:
        """
        Get the captured activations from the current batch.

        Returns:
            The captured activation tensor from the current batch or None if no activations captured yet
        """
        return self.tensor_metadata.get("activations")

    def clear_captured(self) -> None:
        """Clear captured activations for current batch."""
        self.tensor_metadata.pop("activations", None)
        self.metadata.pop("activations_shape", None)
