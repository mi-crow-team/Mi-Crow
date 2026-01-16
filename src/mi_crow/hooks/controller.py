from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from mi_crow.hooks.hook import Hook, HookType, HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT
from mi_crow.hooks.utils import (
    extract_tensor_from_input,
    extract_tensor_from_output,
    apply_modification_to_output
)
from mi_crow.utils import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class Controller(Hook):
    """
    Abstract base class for controller hooks that modify activations during inference.
    
    Controllers can modify inputs (pre_forward) or outputs (forward) of layers.
    They are designed to actively change the behavior of the model during inference.
    """

    def __init__(
            self,
            hook_type: HookType | str = HookType.FORWARD,
            hook_id: str | None = None,
            layer_signature: str | int | None = None
    ):
        """
        Initialize a controller hook.
        
        Args:
            hook_type: Type of hook (HookType.FORWARD or HookType.PRE_FORWARD)
            hook_id: Unique identifier
            layer_signature: Layer to attach to (optional, for compatibility)
        """
        super().__init__(layer_signature=layer_signature, hook_type=hook_type, hook_id=hook_id)

    def _handle_pre_forward(
            self,
            module: torch.nn.Module,
            input: HOOK_FUNCTION_INPUT
    ) -> HOOK_FUNCTION_INPUT | None:
        """Handle pre-forward hook execution.
        
        Args:
            module: The PyTorch module being hooked
            input: Tuple of input tensors to the module
            
        Returns:
            Modified input tuple or None to keep original
        """
        input_tensor = extract_tensor_from_input(input)

        if input_tensor is None:
            return None

        modified_tensor = self.modify_activations(module, input_tensor, input_tensor)

        if modified_tensor is not None and isinstance(modified_tensor, torch.Tensor):
            result = list(input)
            if len(result) > 0:
                result[0] = modified_tensor
            return tuple(result)
        return None

    def _handle_forward(
            self,
            module: torch.nn.Module,
            input: HOOK_FUNCTION_INPUT,
            output: HOOK_FUNCTION_OUTPUT
    ) -> None:
        """Handle forward hook execution.

        Args:
            module: The PyTorch module being hooked
            input: Tuple of input tensors to the module
            output: Output tensor(s) from the module
        """
        output_tensor = extract_tensor_from_output(output)

        if output_tensor is None:
            return

        input_tensor = extract_tensor_from_input(input)
        modified_tensor = self.modify_activations(module, input_tensor, output_tensor)
        
        if modified_tensor is not None and isinstance(modified_tensor, torch.Tensor):
            target_device = None
            if self.context is not None and hasattr(self.context, 'device') and self.context.device:
                target_device = torch.device(self.context.device)
            apply_modification_to_output(output, modified_tensor, target_device=target_device)

    def _hook_fn(
            self,
            module: torch.nn.Module,
            input: HOOK_FUNCTION_INPUT,
            output: HOOK_FUNCTION_OUTPUT
    ) -> None | HOOK_FUNCTION_INPUT:
        """
        Internal hook function that modifies activations.
        
        If the instance also inherits from Detector, first processes activations
        as a Detector (saves metadata), then modifies activations as a Controller.
        
        Args:
            module: The PyTorch module being hooked
            input: Tuple of input tensors to the module
            output: Output tensor(s) from the module
            
        Returns:
            For pre_forward hooks: modified inputs (tuple) or None to keep original
            For forward hooks: None (forward hooks cannot modify output in PyTorch)
            
        Raises:
            RuntimeError: If modify_activations raises an exception
        """
        if not self._enabled:
            return None

        # Check if this instance also inherits from Detector
        if self._is_both_controller_and_detector():
            # First, process activations as a Detector (save metadata)
            try:
                self.process_activations(module, input, output)
            except Exception as e:
                logger.warning(
                    f"Error in {self.__class__.__name__} detector process_activations: {e}",
                    exc_info=True
                )

        try:
            if self.hook_type == HookType.PRE_FORWARD:
                return self._handle_pre_forward(module, input)
            else:
                self._handle_forward(module, input, output)
                return None
        except Exception as e:
            raise RuntimeError(
                f"Error in controller {self.id} modify_activations: {e}"
            ) from e

    @abc.abstractmethod
    def modify_activations(
            self,
            module: nn.Module,
            inputs: torch.Tensor | None,
            output: torch.Tensor | None
    ) -> torch.Tensor | None:
        """
        Modify activations from the hooked layer.
        
        For pre_forward hooks: receives input tensor, should return modified input tensor.
        For forward hooks: receives input and output tensors, should return modified output tensor.
        
        Args:
            module: The PyTorch module being hooked
            inputs: Input tensor (None for forward hooks if not available)
            output: Output tensor (None for pre_forward hooks)
            
        Returns:
            Modified input tensor (for pre_forward) or modified output tensor (for forward).
            Return None to keep original tensor unchanged.
            
        Raises:
            Exception: Subclasses may raise exceptions for invalid inputs or modification errors
        """
        raise NotImplementedError("modify_activations must be implemented by subclasses")
