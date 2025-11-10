from typing import Any, Callable, TYPE_CHECKING
import torch

from amber.hooks.controller import Controller
from amber.hooks.hook import HookType

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
        """
        super().__init__(hook_type=hook_type, hook_id=hook_id, layer_signature=layer_signature)
        self.function = function
    
    def modify_activations(
        self,
        module: "nn.Module",
        inputs: tuple,
        output: Any
    ) -> Any:
        """
        Apply the user-provided function to activations.
        
        Args:
            module: The PyTorch module being hooked
            inputs: Tuple of input tensors to the module
            output: Output tensor or tuple/list of tensors from the module (None for pre_forward hooks)
            
        Returns:
            Modified activations with function applied
        """
        target = output if self.hook_type == HookType.FORWARD else inputs
        
        if isinstance(target, torch.Tensor):
            return self.function(target)
        
        
        return target

