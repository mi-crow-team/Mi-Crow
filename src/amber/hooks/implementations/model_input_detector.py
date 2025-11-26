from __future__ import annotations

from typing import TYPE_CHECKING, Dict
import torch

from amber.hooks.detector import Detector
from amber.hooks.hook import HookType, HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT

if TYPE_CHECKING:
    from torch import nn


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

