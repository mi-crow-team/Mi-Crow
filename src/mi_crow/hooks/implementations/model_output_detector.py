from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mi_crow.hooks.detector import Detector
from mi_crow.hooks.hook import HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT, HookType

if TYPE_CHECKING:
    pass


class ModelOutputDetector(Detector):
    """
    Detector hook that captures and saves model outputs.

    This detector is designed to be attached to the root model module and captures:
    - Model outputs (logits) from the model's forward pass
    - Hidden states (optional) from the model's forward pass

    Uses FORWARD hook to capture outputs after they are computed.
    Useful for saving model outputs for analysis or training.
    """

    def __init__(
        self,
        layer_signature: str | int | None = None,
        hook_id: str | None = None,
        save_output_logits: bool = True,
        save_output_hidden_state: bool = False,
    ):
        """
        Initialize the model output detector.

        Args:
            layer_signature: Layer to capture from (typically the root model, can be None)
            hook_id: Unique identifier for this hook
            save_output_logits: Whether to save output logits (if available)
            save_output_hidden_state: Whether to save last_hidden_state (if available)
        """
        super().__init__(hook_type=HookType.FORWARD, hook_id=hook_id, store=None, layer_signature=layer_signature)
        self.save_output_logits = save_output_logits
        self.save_output_hidden_state = save_output_hidden_state

    def _extract_output_tensor(self, output: HOOK_FUNCTION_OUTPUT) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Extract logits and last_hidden_state from model output.

        Args:
            output: Output from the model forward pass

        Returns:
            Tuple of (logits, last_hidden_state), either can be None
        """
        logits = None
        hidden_state = None

        if output is None:
            return None, None

        # Handle HuggingFace output objects
        if hasattr(output, "logits"):
            logits = output.logits
        if hasattr(output, "last_hidden_state"):
            hidden_state = output.last_hidden_state

        # Handle tuple output (logits might be first element)
        if isinstance(output, (tuple, list)) and len(output) > 0:
            first_item = output[0]
            if isinstance(first_item, torch.Tensor) and logits is None:
                logits = first_item

        # Handle direct tensor output
        if isinstance(output, torch.Tensor) and logits is None:
            logits = output

        return logits, hidden_state

    def process_activations(
        self, module: torch.nn.Module, input: HOOK_FUNCTION_INPUT, output: HOOK_FUNCTION_OUTPUT
    ) -> None:
        """
        Extract and store model outputs.

        Args:
            module: The PyTorch module being hooked (typically the root model)
            input: Tuple of input tensors/dicts to the module
            output: Output from the module

        Raises:
            RuntimeError: If tensor extraction or storage fails
        """
        try:
            # Extract and save outputs
            logits, hidden_state = self._extract_output_tensor(output)

            if self.save_output_logits and logits is not None:
                if logits.is_cuda:
                    logits_cpu = logits.detach().to("cpu", non_blocking=True)
                else:
                    logits_cpu = logits.detach()
                self.tensor_metadata["output_logits"] = logits_cpu
                self.metadata["output_logits_shape"] = tuple(logits_cpu.shape)

            if self.save_output_hidden_state and hidden_state is not None:
                if hidden_state.is_cuda:
                    hidden_state_cpu = hidden_state.detach().to("cpu", non_blocking=True)
                else:
                    hidden_state_cpu = hidden_state.detach()
                self.tensor_metadata["output_hidden_state"] = hidden_state_cpu
                self.metadata["output_hidden_state_shape"] = tuple(hidden_state_cpu.shape)

        except Exception as e:
            layer_sig = str(self.layer_signature) if self.layer_signature is not None else "unknown"
            raise RuntimeError(
                f"Error extracting outputs in ModelOutputDetector {self.id} (layer={layer_sig}): {e}"
            ) from e

    def get_captured_output_logits(self) -> torch.Tensor | None:
        """Get the captured output logits from the current batch."""
        return self.tensor_metadata.get("output_logits")

    def get_captured_output_hidden_state(self) -> torch.Tensor | None:
        """Get the captured output hidden state from the current batch."""
        return self.tensor_metadata.get("output_hidden_state")

    def clear_captured(self) -> None:
        """Clear all captured outputs for current batch."""
        keys_to_remove = ["output_logits", "output_hidden_state"]
        for key in keys_to_remove:
            self.tensor_metadata.pop(key, None)
            self.metadata.pop(f"{key}_shape", None)
