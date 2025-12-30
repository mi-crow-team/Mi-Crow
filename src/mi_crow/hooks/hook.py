from __future__ import annotations

import abc
import uuid
from enum import Enum
from typing import Callable, TypeAlias, Sequence, Optional, TYPE_CHECKING

import torch
from torch import nn, Tensor
from torch.types import _TensorOrTensors

if TYPE_CHECKING:
    from mi_crow.language_model.context import LanguageModelContext


class HookType(str, Enum):
    """Type of hook to register on a layer."""
    FORWARD = "forward"
    PRE_FORWARD = "pre_forward"


HOOK_FUNCTION_INPUT: TypeAlias = Sequence[Tensor]
HOOK_FUNCTION_OUTPUT: TypeAlias = _TensorOrTensors | None


class HookError(Exception):
    """Exception raised when a hook encounters an error during execution."""
    
    def __init__(self, hook_id: str, hook_type: str, original_error: Exception):
        """
        Initialize HookError.
        
        Args:
            hook_id: Unique identifier of the hook that raised the error
            hook_type: Type of hook (e.g., "forward", "pre_forward")
            original_error: The original exception that was raised
        """
        self.hook_id = hook_id
        self.hook_type = hook_type
        self.original_error = original_error
        message = f"Hook {hook_id} (type={hook_type}) raised exception: {original_error}"
        super().__init__(message)


class Hook(abc.ABC):
    """
    Abstract base class for hooks that can be registered on language model layers.
    
    Hooks provide a way to intercept and process activations during model inference.
    They expose PyTorch-compatible callables via get_torch_hook() while providing
    additional functionality like enable/disable and unique identification.
    """

    def __init__(
            self,
            layer_signature: str | int | None = None,
            hook_type: HookType | str = HookType.FORWARD,
            hook_id: str | None = None
    ):
        """
        Initialize a hook.
        
        Args:
            layer_signature: Layer name or index to attach hook to
            hook_type: Type of hook - HookType.FORWARD or HookType.PRE_FORWARD
            hook_id: Unique identifier (auto-generated if not provided)
            
        Raises:
            ValueError: If hook_type string is invalid
        """
        self.layer_signature = layer_signature
        self.hook_type = self._normalize_hook_type(hook_type)
        self.id = hook_id if hook_id is not None else str(uuid.uuid4())
        self._enabled = True
        self._torch_hook_handle = None
        self._context: Optional["LanguageModelContext"] = None

    def _normalize_hook_type(self, hook_type: HookType | str) -> HookType:
        """Normalize hook_type to HookType enum.
        
        Args:
            hook_type: HookType enum or string value
            
        Returns:
            HookType enum value
            
        Raises:
            ValueError: If hook_type string is not a valid HookType value
        """
        if isinstance(hook_type, HookType):
            return hook_type
        
        if isinstance(hook_type, str):
            try:
                return HookType(hook_type)
            except ValueError:
                valid_values = [ht.value for ht in HookType]
                raise ValueError(
                    f"Invalid hook_type string '{hook_type}'. "
                    f"Must be one of: {valid_values}"
                ) from None
        
        raise ValueError(
            f"hook_type must be HookType enum or string, got: {type(hook_type)}"
        )

    def _create_pre_forward_wrapper(self) -> Callable:
        """Create a pre-forward hook wrapper function.
        
        Returns:
            Wrapper function for pre-forward hooks
        """
        def pre_forward_wrapper(module: nn.Module, input: HOOK_FUNCTION_INPUT) -> None | HOOK_FUNCTION_INPUT:
            if not self._enabled:
                return None
            try:
                result = self._hook_fn(module, input, None)
                return result if result is not None else None
            except Exception as e:
                raise HookError(self.id, self.hook_type.value, e) from e

        return pre_forward_wrapper

    def _create_forward_wrapper(self) -> Callable:
        """Create a forward hook wrapper function.
        
        Returns:
            Wrapper function for forward hooks
        """
        def forward_wrapper(module: nn.Module, input: HOOK_FUNCTION_INPUT, output: HOOK_FUNCTION_OUTPUT) -> None:
            if not self._enabled:
                return None
            try:
                self._hook_fn(module, input, output)
                return None
            except Exception as e:
                raise HookError(self.id, self.hook_type.value, e) from e

        return forward_wrapper

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

    @property
    def context(self) -> Optional["LanguageModelContext"]:
        """Get the LanguageModelContext associated with this hook."""
        return self._context

    def set_context(self, context: "LanguageModelContext") -> None:
        """Set the LanguageModelContext for this hook.
        
        Args:
            context: The LanguageModelContext instance
        """
        self._context = context

    def _is_both_controller_and_detector(self) -> bool:
        """
        Check if this hook instance inherits from both Controller and Detector.
        
        Uses MRO (Method Resolution Order) to check for both class names
        without requiring imports, avoiding circular dependencies.
        
        Returns:
            True if the instance inherits from both Controller and Detector, False otherwise
        """
        mro_class_names = [cls.__name__ for cls in type(self).__mro__]
        return 'Controller' in mro_class_names and 'Detector' in mro_class_names

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
            return self._create_pre_forward_wrapper()
        else:
            return self._create_forward_wrapper()

    @abc.abstractmethod
    def _hook_fn(
            self,
            module: torch.nn.Module,
            input: HOOK_FUNCTION_INPUT,
            output: HOOK_FUNCTION_OUTPUT
    ) -> None | HOOK_FUNCTION_INPUT:
        """
        Internal hook function to be implemented by subclasses.
        
        Args:
            module: The PyTorch module being hooked
            input: Tuple of input tensors to the module
            output: Output tensor(s) from the module (None for pre_forward hooks)
            
        Returns:
            For pre_forward hooks: modified inputs (tuple) or None to keep original
            For forward hooks: None (forward hooks cannot modify output in PyTorch)
            
        Raises:
            Exception: Subclasses may raise exceptions which will be caught by the wrapper
        """
        raise NotImplementedError("_hook_fn must be implemented by subclasses")
