import abc
import uuid
from enum import Enum
from typing import Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


class HookType(str, Enum):
    """Enum for hook types."""
    FORWARD = "forward"
    PRE_FORWARD = "pre_forward"


class Hook(abc.ABC):
    """
    Abstract base class for hooks that can be registered on language model layers.
    
    Hooks provide a way to intercept and process activations during model inference.
    They expose PyTorch-compatible callables via get_torch_hook() while providing
    additional functionality like enable/disable and unique identification.
    """
    
    def __init__(
        self,
        layer_signature: str | int,
        hook_type: HookType | str = HookType.FORWARD,
        hook_id: str | None = None
    ):
        """
        Initialize a hook.
        
        Args:
            layer_signature: Layer name or index to attach hook to
            hook_type: Type of hook - HookType.FORWARD or HookType.PRE_FORWARD
            hook_id: Unique identifier (auto-generated if not provided)
        """
        self.layer_signature = layer_signature
        # Convert string to enum if needed for backward compatibility
        if isinstance(hook_type, str):
            self.hook_type = HookType(hook_type)
        else:
            self.hook_type = hook_type
        self.id = hook_id if hook_id is not None else str(uuid.uuid4())
        self._enabled = True
        self._torch_hook_handle = None
        
    @property
    def enabled(self) -> bool:
        """Whether this hook is currently enabled."""
        return self._enabled
    
    def enable(self) -> None:
        """Enable this hook."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable this hook."""
        self._enabled = False
    
    def get_torch_hook(self) -> Callable:
        """
        Return a PyTorch-compatible hook function.
        
        The returned callable will check the enabled flag before executing
        and call the abstract _hook_fn method.
        
        Returns:
            A callable compatible with PyTorch's register_forward_hook or
            register_forward_pre_hook APIs.
        """
        if self.hook_type == HookType.PRE_FORWARD:
            def pre_forward_wrapper(module: "nn.Module", inputs: tuple) -> Any:
                if not self._enabled:
                    return None  # PyTorch pre-hooks return None to not modify inputs
                return self._hook_fn(module, inputs, None)
            return pre_forward_wrapper
        else:  # forward
            def forward_wrapper(module: "nn.Module", inputs: tuple, output: Any) -> Any:
                if not self._enabled:
                    return None  # PyTorch forward hooks return None to not modify output
                return self._hook_fn(module, inputs, output)
            return forward_wrapper
    
    @abc.abstractmethod
    def _hook_fn(self, module: "nn.Module", inputs: tuple, output: Any) -> Any:
        """
        Internal hook function to be implemented by subclasses.
        
        Args:
            module: The PyTorch module being hooked
            inputs: Tuple of inputs to the module
            output: Output from the module (None for pre_forward hooks)
            
        Returns:
            For pre_forward hooks: modified inputs or None
            For forward hooks: modified output or None
        """
        pass
