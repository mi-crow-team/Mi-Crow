from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any
import torch

from amber.hooks.detector import Detector
from amber.hooks.hook import HookType, HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT
from amber.hooks.utils import extract_tensor_from_output, extract_tensor_from_input

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
            
        Raises:
            ValueError: If layer_signature is None
        """
        if layer_signature is None:
            raise ValueError("layer_signature cannot be None for LayerActivationDetector")

        super().__init__(
            hook_type=HookType.FORWARD,
            hook_id=hook_id,
            store=None,
            layer_signature=layer_signature
        )

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
                tensor_cpu = tensor.detach().to("cpu")
                # Store current batch's tensor (overwrites previous)
                self.tensor_metadata['activations'] = tensor_cpu
                # Store activations shape to metadata
                self.metadata['activations_shape'] = tuple(tensor_cpu.shape)
        except Exception as e:
            raise RuntimeError(
                f"Error extracting activations in LayerActivationDetector {self.id}: {e}"
            ) from e

    def get_captured(self) -> torch.Tensor | None:
        """
        Get the captured activations from the current batch.
        
        Returns:
            The captured activation tensor from the current batch or None if no activations captured yet
        """
        return self.tensor_metadata.get('activations')

    def clear_captured(self) -> None:
        """Clear captured activations for current batch."""
        self.tensor_metadata.pop('activations', None)
        self.metadata.pop('activations_shape', None)


class ModelInputDetector(Detector):
    """
    Detector hook that captures and saves tokenized inputs from model forward pass.
    
    This detector is designed to be attached to the root model module and captures:
    - Tokenized inputs (input_ids) from the model's forward pass
    - Attention masks (optional) from the model's forward pass
    
    Uses PRE_FORWARD hook to capture inputs before they are processed.
    Useful for saving tokenized inputs for analysis or training.
    """

    def __init__(
            self,
            layer_signature: str | int | None = None,
            hook_id: str | None = None,
            save_input_ids: bool = True,
            save_attention_mask: bool = False
    ):
        """
        Initialize the model input detector.
        
        Args:
            layer_signature: Layer to capture from (typically the root model, can be None)
            hook_id: Unique identifier for this hook
            save_input_ids: Whether to save input_ids tensor
            save_attention_mask: Whether to save attention_mask tensor
        """
        super().__init__(
            hook_type=HookType.PRE_FORWARD,
            hook_id=hook_id,
            store=None,
            layer_signature=layer_signature
        )
        self.save_input_ids = save_input_ids
        self.save_attention_mask = save_attention_mask

    def _extract_input_ids(self, input: HOOK_FUNCTION_INPUT) -> torch.Tensor | None:
        """
        Extract input_ids from model input.
        
        Handles various input formats:
        - Dict with 'input_ids' key (most common for HuggingFace models)
        - Tuple with dict as first element
        - Tuple with tensor as first element
        
        Args:
            input: Input to the model forward pass
            
        Returns:
            input_ids tensor or None if not found
        """
        if not input or len(input) == 0:
            return None
        
        first_item = input[0]
        
        # Handle dict input (most common case for HuggingFace models)
        if isinstance(first_item, dict):
            if 'input_ids' in first_item:
                return first_item['input_ids']
            return None
        
        # Handle tensor input (direct input_ids)
        if isinstance(first_item, torch.Tensor):
            return first_item
        
        return None

    def _extract_attention_mask(self, input: HOOK_FUNCTION_INPUT) -> torch.Tensor | None:
        """
        Extract attention_mask from model input.
        
        Args:
            input: Input to the model forward pass
            
        Returns:
            attention_mask tensor or None if not found
        """
        if not input or len(input) == 0:
            return None
        
        first_item = input[0]
        
        if isinstance(first_item, dict):
            if 'attention_mask' in first_item:
                return first_item['attention_mask']
        
        return None

    def set_inputs_from_encodings(self, encodings: Dict[str, torch.Tensor]) -> None:
        """
        Manually set inputs from encodings dictionary.
        
        This is useful when the model is called with keyword arguments,
        as PyTorch's pre_forward hook doesn't receive kwargs.
        
        Args:
            encodings: Dictionary of encoded inputs (e.g., from lm.forwards() or lm.tokenize())
            
        Raises:
            RuntimeError: If tensor extraction or storage fails
        """
        try:
            if self.save_input_ids and 'input_ids' in encodings:
                input_ids = encodings['input_ids']
                self.tensor_metadata['input_ids'] = input_ids.detach().to("cpu")
                self.metadata['input_ids_shape'] = tuple(input_ids.shape)
            
            if self.save_attention_mask and 'attention_mask' in encodings:
                attention_mask = encodings['attention_mask']
                self.tensor_metadata['attention_mask'] = attention_mask.detach().to("cpu")
                self.metadata['attention_mask_shape'] = tuple(attention_mask.shape)
        except Exception as e:
            raise RuntimeError(
                f"Error setting inputs from encodings in ModelInputDetector {self.id}: {e}"
            ) from e

    def process_activations(
            self,
            module: torch.nn.Module,
            input: HOOK_FUNCTION_INPUT,
            output: HOOK_FUNCTION_OUTPUT
    ) -> None:
        """
        Extract and store tokenized inputs.
        
        Note: For HuggingFace models called with **kwargs, the input tuple may be empty.
        In such cases, use set_inputs_from_encodings() to manually set inputs from
        the encodings dictionary returned by lm.forwards().
        
        Args:
            module: The PyTorch module being hooked (typically the root model)
            input: Tuple of input tensors/dicts to the module
            output: Output from the module (None for PRE_FORWARD hooks)
            
        Raises:
            RuntimeError: If tensor extraction or storage fails
        """
        try:
            # Extract and save inputs
            if self.save_input_ids:
                input_ids = self._extract_input_ids(input)
                if input_ids is not None:
                    self.tensor_metadata['input_ids'] = input_ids.detach().to("cpu")
                    self.metadata['input_ids_shape'] = tuple(input_ids.shape)
            
            if self.save_attention_mask:
                attention_mask = self._extract_attention_mask(input)
                if attention_mask is not None:
                    self.tensor_metadata['attention_mask'] = attention_mask.detach().to("cpu")
                    self.metadata['attention_mask_shape'] = tuple(attention_mask.shape)
                
        except Exception as e:
            raise RuntimeError(
                f"Error extracting inputs in ModelInputDetector {self.id}: {e}"
            ) from e

    def get_captured_input_ids(self) -> torch.Tensor | None:
        """Get the captured input_ids from the current batch."""
        return self.tensor_metadata.get('input_ids')

    def get_captured_attention_mask(self) -> torch.Tensor | None:
        """Get the captured attention_mask from the current batch."""
        return self.tensor_metadata.get('attention_mask')

    def clear_captured(self) -> None:
        """Clear all captured inputs for current batch."""
        keys_to_remove = ['input_ids', 'attention_mask']
        for key in keys_to_remove:
            self.tensor_metadata.pop(key, None)
            self.metadata.pop(f'{key}_shape', None)


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
            save_output_hidden_state: bool = False
    ):
        """
        Initialize the model output detector.
        
        Args:
            layer_signature: Layer to capture from (typically the root model, can be None)
            hook_id: Unique identifier for this hook
            save_output_logits: Whether to save output logits (if available)
            save_output_hidden_state: Whether to save last_hidden_state (if available)
        """
        super().__init__(
            hook_type=HookType.FORWARD,
            hook_id=hook_id,
            store=None,
            layer_signature=layer_signature
        )
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
            self,
            module: torch.nn.Module,
            input: HOOK_FUNCTION_INPUT,
            output: HOOK_FUNCTION_OUTPUT
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
                self.tensor_metadata['output_logits'] = logits.detach().to("cpu")
                self.metadata['output_logits_shape'] = tuple(logits.shape)
            
            if self.save_output_hidden_state and hidden_state is not None:
                self.tensor_metadata['output_hidden_state'] = hidden_state.detach().to("cpu")
                self.metadata['output_hidden_state_shape'] = tuple(hidden_state.shape)
                
        except Exception as e:
            raise RuntimeError(
                f"Error extracting outputs in ModelOutputDetector {self.id}: {e}"
            ) from e

    def get_captured_output_logits(self) -> torch.Tensor | None:
        """Get the captured output logits from the current batch."""
        return self.tensor_metadata.get('output_logits')

    def get_captured_output_hidden_state(self) -> torch.Tensor | None:
        """Get the captured output hidden state from the current batch."""
        return self.tensor_metadata.get('output_hidden_state')

    def clear_captured(self) -> None:
        """Clear all captured outputs for current batch."""
        keys_to_remove = ['output_logits', 'output_hidden_state']
        for key in keys_to_remove:
            self.tensor_metadata.pop(key, None)
            self.metadata.pop(f'{key}_shape', None)
