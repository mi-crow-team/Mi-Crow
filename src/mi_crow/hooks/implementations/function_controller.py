from __future__ import annotations

from typing import Callable, TYPE_CHECKING
import torch

from mi_crow.hooks.controller import Controller
from mi_crow.hooks.hook import HookType

if TYPE_CHECKING:
    from torch import nn


class FunctionController(Controller):
    """
    A controller that applies a user-provided function to tensors during inference.
    
    This controller allows users to pass any function and apply it to activations.
    The function will be applied to:
    - Single tensors directly
    - All tensors in tuples/lists (default behavior)
    
    Example:
        >>> # Scale activations by 2
        >>> controller = FunctionController(
        ...     layer_signature="layer_0",
        ...     function=lambda x: x * 2.0
        ... )
    """
    
    def __init__(
        self,
        layer_signature: str | int,
        function: Callable[[torch.Tensor], torch.Tensor],
        hook_type: HookType | str = HookType.FORWARD,
        hook_id: str | None = None,
    ):
        """
        Initialize a function controller.
        
        Args:
            layer_signature: Layer to attach to
            function: Function to apply to tensors. Must take a torch.Tensor and return a torch.Tensor
            hook_type: Type of hook (HookType.FORWARD or HookType.PRE_FORWARD)
            hook_id: Unique identifier
            
        Raises:
            ValueError: If function is None or not callable
        """
        if function is None:
            raise ValueError("function cannot be None")
        
        if not callable(function):
            raise ValueError(f"function must be callable, got: {type(function)}")
        
        super().__init__(hook_type=hook_type, hook_id=hook_id, layer_signature=layer_signature)
        self.function = function
    
    def modify_activations(
        self,
        module: "nn.Module",
        inputs: torch.Tensor | None,
        output: torch.Tensor | None
    ) -> torch.Tensor | None:
        """
        Apply the user-provided function to activations.
        
        Args:
            module: The PyTorch module being hooked
            inputs: Input tensor (None for forward hooks)
            output: Output tensor (None for pre_forward hooks)
            
        Returns:
            Modified tensor with function applied, or None if target tensor is None
            
        Raises:
            RuntimeError: If function raises an exception when applied to tensor
        """
        target = output if self.hook_type == HookType.FORWARD else inputs
        
        if target is None or not isinstance(target, torch.Tensor):
            return target
        
        try:
            result = self.function(target)
            if not isinstance(result, torch.Tensor):
                raise TypeError(
                    f"Function must return a torch.Tensor, got: {type(result)}"
                )
            return result
        except Exception as e:
            raise RuntimeError(
                f"Error applying function in FunctionController {self.id}: {e}"
            ) from e
