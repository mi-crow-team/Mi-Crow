import abc
from typing import Any, TYPE_CHECKING

from amber.hooks.hook import Hook, HookType

if TYPE_CHECKING:
    from torch import nn


class Controller(Hook):
    """
    Abstract base class for controller hooks that modify activations during inference.
    
    Controllers can modify inputs (pre_forward) or outputs (forward) of layers.
    """
    
    def __init__(
        self,
        layer_signature: str | int,
        hook_type: HookType | str = HookType.FORWARD,
        hook_id: str | None = None
    ):
        """
        Initialize a controller hook.
        
        Args:
            layer_signature: Layer to attach to
            hook_type: Type of hook (HookType.FORWARD or HookType.PRE_FORWARD)
            hook_id: Unique identifier
        """
        super().__init__(layer_signature, hook_type, hook_id)
    
    def _hook_fn(self, module: "nn.Module", inputs: tuple, output: Any) -> Any:
        """
        Internal hook function that modifies activations.
        
        Calls modify_activations to get the modified tensor.
        """
        if not self._enabled:
            return None
        
        try:
            if self.hook_type == HookType.PRE_FORWARD:
                # Modify inputs - inputs is a tuple, return modified version
                modified = self.modify_activations(module, inputs, None)
                return modified
            else:  # forward
                # Modify output
                modified = self.modify_activations(module, inputs, output)
                return modified
        except Exception:
            # Don't let hook errors crash inference
            return None
    
    @abc.abstractmethod
    def modify_activations(
        self,
        module: "nn.Module",
        inputs: tuple,
        output: Any
    ) -> Any:
        """
        Modify activations from the hooked layer.
        
        For pre_forward hooks: receives inputs tuple, should return modified inputs.
        For forward hooks: receives output, should return modified output.
        
        Args:
            module: The PyTorch module being hooked
            inputs: Tuple of inputs to the module
            output: Output from the module (None for pre_forward hooks)
            
        Returns:
            Modified inputs (for pre_forward) or modified output (for forward)
        """
        pass
