from __future__ import annotations

import abc
import uuid
from enum import Enum
from typing import Callable, TypeAlias, Sequence

import torch
from torch import nn, Tensor
from torch.types import _TensorOrTensors


class HookType(str, Enum):
    FORWARD = "forward"
    PRE_FORWARD = "pre_forward"


HOOK_FUNCTION_INPUT: TypeAlias = Sequence[Tensor]
HOOK_FUNCTION_OUTPUT: TypeAlias = _TensorOrTensors | None

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
        """
        self.layer_signature = layer_signature
        if type(hook_type) == str:
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
            def pre_forward_wrapper(module: nn.Module, input: HOOK_FUNCTION_INPUT) -> None | HOOK_FUNCTION_INPUT:
                if not self._enabled:
                    return None
                result = self._hook_fn(module, input, None)
                return result if result is not None else None

            return pre_forward_wrapper
        else:
            def forward_wrapper(module: nn.Module, input: HOOK_FUNCTION_INPUT, output: HOOK_FUNCTION_OUTPUT) -> None:
                if not self._enabled:
                    return None
                self._hook_fn(module, input, output)
                return None

            return forward_wrapper

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
        """
        raise NotImplementedError("_hook_fn must be implemented by subclasses")
