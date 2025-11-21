"""Tests for Controller base class."""

import pytest
import torch
from torch import nn

from amber.hooks.controller import Controller
from amber.hooks.hook import HookType
from tests.unit.fixtures.hooks import MockController


class ConcreteController(Controller):
    """Concrete implementation of Controller for testing."""

    def modify_activations(self, module, inputs, output):
        """Implementation that doubles the tensor."""
        target = output if self.hook_type == HookType.FORWARD else inputs
        if target is not None and isinstance(target, torch.Tensor):
            return target * 2.0
        return target


class TestControllerInitialization:
    """Tests for Controller initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        controller = ConcreteController()
        assert controller.hook_type == HookType.FORWARD

    def test_init_with_hook_type(self):
        """Test initialization with hook type."""
        controller = ConcreteController(hook_type=HookType.PRE_FORWARD)
        assert controller.hook_type == HookType.PRE_FORWARD

    def test_init_with_layer_signature(self):
        """Test initialization with layer signature."""
        controller = ConcreteController(layer_signature="layer_0")
        assert controller.layer_signature == "layer_0"


class TestControllerModifyActivations:
    """Tests for modify_activations method."""

    def test_modify_activations_forward_hook(self):
        """Test modify_activations with forward hook."""
        controller = ConcreteController(hook_type=HookType.FORWARD)
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5)
        
        result = controller.modify_activations(module, input_tensor, output_tensor)
        assert result is not None
        assert torch.allclose(result, output_tensor * 2.0)

    def test_modify_activations_pre_forward_hook(self):
        """Test modify_activations with pre-forward hook."""
        controller = ConcreteController(hook_type=HookType.PRE_FORWARD)
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        
        result = controller.modify_activations(module, input_tensor, None)
        assert result is not None
        assert torch.allclose(result, input_tensor * 2.0)

    def test_modify_activations_none_tensor(self):
        """Test modify_activations with None tensor."""
        controller = ConcreteController()
        module = nn.Linear(10, 5)
        
        result = controller.modify_activations(module, None, None)
        assert result is None

    def test_modify_activations_when_disabled(self):
        """Test that modify_activations is not called when disabled."""
        controller = ConcreteController()
        controller.disable()
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5)
        
        # Should return None when disabled
        result = controller._hook_fn(module, (input_tensor,), output_tensor)
        assert result is None

    def test_modify_activations_exception_handling(self):
        """Test that exceptions in modify_activations are handled."""
        class FailingController(Controller):
            def modify_activations(self, module, inputs, output):
                raise ValueError("Test error")
        
        controller = FailingController()
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5)
        
        with pytest.raises(RuntimeError, match="Error in controller"):
            controller._hook_fn(module, (input_tensor,), output_tensor)


class TestControllerHandlePreForward:
    """Tests for _handle_pre_forward method."""

    def test_handle_pre_forward_modifies_input(self):
        """Test that _handle_pre_forward modifies input."""
        controller = ConcreteController(hook_type=HookType.PRE_FORWARD)
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        
        result = controller._handle_pre_forward(module, (input_tensor,))
        assert result is not None
        assert isinstance(result, tuple)
        assert torch.allclose(result[0], input_tensor * 2.0)

    def test_handle_pre_forward_no_tensor_returns_none(self):
        """Test that _handle_pre_forward returns None when no tensor."""
        controller = ConcreteController(hook_type=HookType.PRE_FORWARD)
        module = nn.Linear(10, 5)
        
        result = controller._handle_pre_forward(module, ())
        assert result is None


class TestControllerHandleForward:
    """Tests for _handle_forward method."""

    def test_handle_forward_calls_modify_activations(self):
        """Test that _handle_forward calls modify_activations."""
        controller = ConcreteController(hook_type=HookType.FORWARD)
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5)
        
        # Should not raise
        controller._handle_forward(module, (input_tensor,), output_tensor)

    def test_handle_forward_no_output_returns_early(self):
        """Test that _handle_forward returns early when no output."""
        controller = ConcreteController(hook_type=HookType.FORWARD)
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        
        # Should not raise
        controller._handle_forward(module, (input_tensor,), None)


class TestMockController:
    """Tests for MockController utility."""

    def test_mock_controller_modifies_activations(self):
        """Test that MockController modifies activations."""
        controller = MockController(modification_factor=3.0)
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5)
        
        result = controller.modify_activations(module, input_tensor, output_tensor)
        assert result is not None
        assert torch.allclose(result, output_tensor * 3.0)

    def test_mock_controller_tracks_modifications(self):
        """Test that MockController tracks modification count."""
        controller = MockController()
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5)
        
        controller.modify_activations(module, input_tensor, output_tensor)
        controller.modify_activations(module, input_tensor, output_tensor)
        
        assert controller.modified_count == 2

