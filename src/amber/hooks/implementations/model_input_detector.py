from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Set, List, Optional
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
    - Special token masks (optional) indicating positions of special tokens
    
    Uses PRE_FORWARD hook to capture inputs before they are processed.
    Useful for saving tokenized inputs for analysis or training.
    """

    def __init__(
            self,
            layer_signature: str | int | None = None,
            hook_id: str | None = None,
            save_input_ids: bool = True,
            save_attention_mask: bool = False,
            save_special_token_mask: bool = False,
            special_token_ids: Optional[List[int] | Set[int]] = None
    ):
        """
        Initialize the model input detector.
        
        Args:
            layer_signature: Layer to capture from (typically the root model, can be None)
            hook_id: Unique identifier for this hook
            save_input_ids: Whether to save input_ids tensor
            save_attention_mask: Whether to save attention_mask tensor
            save_special_token_mask: Whether to save special token mask tensor
            special_token_ids: Optional list/set of special token IDs. If None, will try to
                             extract from model/tokenizer if available.
        """
        super().__init__(
            hook_type=HookType.PRE_FORWARD,
            hook_id=hook_id,
            store=None,
            layer_signature=layer_signature
        )
        self.save_input_ids = save_input_ids
        self.save_attention_mask = save_attention_mask
        self.save_special_token_mask = save_special_token_mask
        self.special_token_ids = set(special_token_ids) if special_token_ids is not None else None

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
        
        if isinstance(first_item, dict):
            if 'input_ids' in first_item:
                return first_item['input_ids']
            return None
        
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

    def _get_special_token_ids(self, module: torch.nn.Module) -> Set[int]:
        """
        Get special token IDs from user-provided list or extract from model/tokenizer.
        
        Args:
            module: The PyTorch module being hooked (may have tokenizer/config attributes)
            
        Returns:
            Set of special token IDs
        """
        if self.special_token_ids is not None:
            return self.special_token_ids
        
        special_ids = set()
        
        if hasattr(module, 'tokenizer'):
            tokenizer = module.tokenizer
            token_id_attrs = ['pad_token_id', 'eos_token_id', 'bos_token_id', 'unk_token_id', 
                            'cls_token_id', 'sep_token_id', 'mask_token_id']
            for attr in token_id_attrs:
                token_id = getattr(tokenizer, attr, None)
                if token_id is not None:
                    special_ids.add(token_id)
        
        if hasattr(module, 'config'):
            config = module.config
            token_id_attrs = ['pad_token_id', 'eos_token_id', 'bos_token_id', 'unk_token_id',
                            'cls_token_id', 'sep_token_id', 'mask_token_id']
            for attr in token_id_attrs:
                token_id = getattr(config, attr, None)
                if token_id is not None:
                    special_ids.add(token_id)
        
        if hasattr(module, 'tokenizer') and hasattr(module.tokenizer, 'all_special_ids'):
            all_special_ids = module.tokenizer.all_special_ids
            if all_special_ids:
                special_ids.update(all_special_ids)
        
        return special_ids

    def _create_special_token_mask(self, input_ids: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
        """
        Create a binary mask indicating special token positions.
        
        Args:
            input_ids: Input token IDs tensor (batch_size × sequence_length)
            module: The PyTorch module being hooked
            
        Returns:
            Binary mask tensor with same shape as input_ids (1 for special tokens, 0 otherwise)
        """
        special_token_ids = self._get_special_token_ids(module)
        
        if not special_token_ids:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        
        special_ids_tensor = torch.tensor(list(special_token_ids), device=input_ids.device, dtype=input_ids.dtype)
        
        expanded_input = input_ids.unsqueeze(-1)
        expanded_special = special_ids_tensor.unsqueeze(0).unsqueeze(0)
        
        mask = (expanded_input == expanded_special).any(dim=-1)
        
        return mask.to(torch.bool)

    def set_inputs_from_encodings(self, encodings: Dict[str, torch.Tensor], module: Optional[torch.nn.Module] = None) -> None:
        """
        Manually set inputs from encodings dictionary.
        
        This is useful when the model is called with keyword arguments,
        as PyTorch's pre_forward hook doesn't receive kwargs.
        
        Args:
            encodings: Dictionary of encoded inputs (e.g., from lm.forwards() or lm.tokenize())
            module: Optional module for extracting special token IDs. If None, special token
                   mask will use previously extracted special token IDs or return all zeros.
            
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
            
            if self.save_special_token_mask and 'input_ids' in encodings:
                input_ids = encodings['input_ids']
                if module is None:
                    class DummyModule:
                        pass
                    module = DummyModule()
                
                special_token_mask = self._create_special_token_mask(input_ids, module)
                self.tensor_metadata['special_token_mask'] = special_token_mask.detach().to("cpu")
                self.metadata['special_token_mask_shape'] = tuple(special_token_mask.shape)
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
            
            if self.save_special_token_mask:
                input_ids = self._extract_input_ids(input)
                if input_ids is not None:
                    special_token_mask = self._create_special_token_mask(input_ids, module)
                    self.tensor_metadata['special_token_mask'] = special_token_mask.detach().to("cpu")
                    self.metadata['special_token_mask_shape'] = tuple(special_token_mask.shape)
                
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

    def get_captured_special_token_mask(self) -> torch.Tensor | None:
        """Get the captured special_token_mask from the current batch."""
        return self.tensor_metadata.get('special_token_mask')

    def clear_captured(self) -> None:
        """Clear all captured inputs for current batch."""
        keys_to_remove = ['input_ids', 'attention_mask', 'special_token_mask']
        for key in keys_to_remove:
            self.tensor_metadata.pop(key, None)
            self.metadata.pop(f'{key}_shape', None)

