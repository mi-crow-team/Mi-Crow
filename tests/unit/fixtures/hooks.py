"""Hook fixtures for testing."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from mi_crow.hooks.controller import Controller
from mi_crow.hooks.detector import Detector
from mi_crow.hooks.hook import HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT, HookType
from mi_crow.hooks.implementations.function_controller import FunctionController
from mi_crow.hooks.implementations.layer_activation_detector import LayerActivationDetector
from mi_crow.store.store import Store


class MockDetector(Detector):
    """Mock detector for testing."""

    def __init__(
        self,
        hook_type: HookType | str = HookType.FORWARD,
        hook_id: Optional[str] = None,
        store: Optional[Store] = None,
        layer_signature: Optional[str | int] = None,
    ):
        super().__init__(
            hook_type=hook_type,
            hook_id=hook_id,
            store=store,
            layer_signature=layer_signature,
        )
        self.processed_count = 0

    def process_activations(
        self,
        module: torch.nn.Module,
        input: HOOK_FUNCTION_INPUT,
        output: HOOK_FUNCTION_OUTPUT,
    ) -> None:
        """Process activations and increment counter."""
        self.processed_count += 1
        self.metadata["count"] = self.processed_count


class MockController(Controller):
    """Mock controller for testing."""

    def __init__(
        self,
        hook_type: HookType | str = HookType.FORWARD,
        hook_id: Optional[str] = None,
        layer_signature: Optional[str | int] = None,
        modification_factor: float = 2.0,
    ):
        super().__init__(
            hook_type=hook_type,
            hook_id=hook_id,
            layer_signature=layer_signature,
        )
        self.modification_factor = modification_factor
        self.modified_count = 0

    def modify_activations(
        self,
        module: torch.nn.Module,
        inputs: Optional[torch.Tensor],
        output: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Modify activations by multiplying by factor."""
        target = output if self.hook_type == HookType.FORWARD else inputs
        if target is not None and isinstance(target, torch.Tensor):
            self.modified_count += 1
            return target * self.modification_factor
        return target


def create_mock_detector(
    hook_type: HookType | str = HookType.FORWARD,
    hook_id: Optional[str] = None,
    store: Optional[Store] = None,
    layer_signature: Optional[str | int] = None,
) -> MockDetector:
    """
    Create a mock detector for testing.
    Args:
        hook_type: Type of hook
        hook_id: Optional hook ID
        store: Optional store instance
        layer_signature: Optional layer signature

    Returns:
        MockDetector instance
    """
    return MockDetector(
        hook_type=hook_type,
        hook_id=hook_id,
        store=store,
        layer_signature=layer_signature,
    )


def create_mock_controller(
    hook_type: HookType | str = HookType.FORWARD,
    hook_id: Optional[str] = None,
    layer_signature: Optional[str | int] = None,
    modification_factor: float = 2.0,
) -> MockController:
    """
    Create a mock controller for testing.
    Args:
        hook_type: Type of hook
        hook_id: Optional hook ID
        layer_signature: Optional layer signature
        modification_factor: Factor to multiply activations by

    Returns:
        MockController instance
    """
    return MockController(
        hook_type=hook_type,
        hook_id=hook_id,
        layer_signature=layer_signature,
        modification_factor=modification_factor,
    )


def create_activation_detector(
    layer_signature: str | int,
    hook_id: Optional[str] = None,
) -> LayerActivationDetector:
    """
    Create a LayerActivationDetector for testing.
    Args:
        layer_signature: Layer to attach to
        hook_id: Optional hook ID

    Returns:
        LayerActivationDetector instance
    """
    return LayerActivationDetector(
        layer_signature=layer_signature,
        hook_id=hook_id,
    )


def create_function_controller(
    layer_signature: str | int,
    function: Callable[[torch.Tensor], torch.Tensor],
    hook_type: HookType | str = HookType.FORWARD,
    hook_id: Optional[str] = None,
) -> FunctionController:
    """
    Create a FunctionController for testing.
    Args:
        layer_signature: Layer to attach to
        function: Function to apply to tensors
        hook_type: Type of hook
        hook_id: Optional hook ID

    Returns:
        FunctionController instance
    """
    return FunctionController(
        layer_signature=layer_signature,
        function=function,
        hook_type=hook_type,
        hook_id=hook_id,
    )
