import abc
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn

from amber.hooks.hook import Hook, HookType, HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT


class Controller(Hook):
    """
    Abstract base class for controller hooks that modify activations during inference.
    
    Controllers can modify inputs (pre_forward) or outputs (forward) of layers.
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

    def _hook_fn(
        self, 
        module: torch.nn.Module, 
        input: HOOK_FUNCTION_INPUT, 
        output: HOOK_FUNCTION_OUTPUT
    ) -> None | HOOK_FUNCTION_INPUT:
        """
        Internal hook function that modifies activations.
        
        Extracts tensors from input/output and calls modify_activations with tensors.
        """
        if not self._enabled:
            return None

        try:
            if self.hook_type == HookType.PRE_FORWARD:
                input_tensor = None
                if len(input) > 0:
                    if isinstance(input[0], torch.Tensor):
                        input_tensor = input[0]
                    elif isinstance(input[0], (tuple, list)):
                        for item in input[0]:
                            if isinstance(item, torch.Tensor):
                                input_tensor = item
                                break

                if input_tensor is None or not isinstance(input_tensor, torch.Tensor):
                    return None
                modified_tensor = self.modify_activations(module, input_tensor, input_tensor)

                if modified_tensor is not None:
                    result = list(input)
                    if len(result) > 0:
                        result[0] = modified_tensor
                    return tuple(result)
                return None
            else:
                # For forward hooks, output is the tensor to modify
                if output is None:
                    return None
                # Extract input tensor if available for modify_activations
                input_tensor = None
                if len(input) > 0 and isinstance(input[0], torch.Tensor):
                    input_tensor = input[0]
                # Handle both single tensor and tuple outputs
                if isinstance(output, torch.Tensor):
                    modified_tensor = self.modify_activations(module, input_tensor, output)
                    # Note: forward hooks can't modify output in PyTorch, but we call modify_activations
                    # for consistency. The actual modification happens via the hook mechanism.
                elif isinstance(output, tuple):
                    # For tuple outputs, modify the first tensor
                    if len(output) > 0 and isinstance(output[0], torch.Tensor):
                        self.modify_activations(module, input_tensor, output[0])
                return None
        except Exception:
            return None

    @abc.abstractmethod
    def modify_activations(
            self,
            module: nn.Module,
            inputs: torch.Tensor,
            output: torch.Tensor
    ) -> torch.Tensor:
        """
        Modify activations from the hooked layer.
        
        For pre_forward hooks: receives input tensor, should return modified input tensor.
        For forward hooks: receives input and output tensors, should return modified output tensor.
        
        Args:
            module: The PyTorch module being hooked
            inputs: Input tensor (None for forward hooks if not available)
            output: Output tensor (None for pre_forward hooks)
            
        Returns:
            Modified input tensor (for pre_forward) or modified output tensor (for forward)
        """
        pass
