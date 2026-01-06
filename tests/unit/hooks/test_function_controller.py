"""Tests for FunctionController."""

import pytest
import torch
from torch import nn

from mi_crow.hooks.implementations.function_controller import FunctionController
from mi_crow.hooks.hook import HookType


class TestFunctionControllerInitialization:
    """Tests for FunctionController initialization."""

    def test_init_with_function(self):
        """Test initialization with function."""
        def scale_by_two(x):
            return x * 2.0
        
        controller = FunctionController(
            layer_signature="layer_0",
            function=scale_by_two
        )
        assert controller.function == scale_by_two
        assert controller.layer_signature == "layer_0"

    def test_init_with_hook_type(self):
        """Test initialization with hook type."""
        def identity(x):
            return x
        
        controller = FunctionController(
            layer_signature="layer_0",
            function=identity,
            hook_type=HookType.PRE_FORWARD
        )
        assert controller.hook_type == HookType.PRE_FORWARD

    def test_init_with_custom_hook_id(self):
        """Test initialization with custom hook ID."""
        def identity(x):
            return x
        
        controller = FunctionController(
            layer_signature="layer_0",
            function=identity,
            hook_id="custom_id"
        )
        assert controller.id == "custom_id"

    def test_init_with_none_function_raises_error(self):
        """Test that None function raises ValueError."""
        with pytest.raises(ValueError, match="function cannot be None"):
            FunctionController(layer_signature="layer_0", function=None)

    def test_init_with_non_callable_raises_error(self):
        """Test that non-callable function raises ValueError."""
        with pytest.raises(ValueError, match="function must be callable"):
            FunctionController(layer_signature="layer_0", function="not a function")


class TestFunctionControllerModifyActivations:
    """Tests for modify_activations method."""

    def test_modify_activations_forward_hook(self):
        """Test modify_activations with forward hook."""
        def scale_by_two(x):
            return x * 2.0
        
        controller = FunctionController(
            layer_signature="layer_0",
            function=scale_by_two,
            hook_type=HookType.FORWARD
        )
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5)
        
        result = controller.modify_activations(module, input_tensor, output_tensor)
        assert result is not None
        assert torch.allclose(result, output_tensor * 2.0)

    def test_modify_activations_pre_forward_hook(self):
        """Test modify_activations with pre-forward hook."""
        def add_one(x):
            return x + 1.0
        
        controller = FunctionController(
            layer_signature="layer_0",
            function=add_one,
            hook_type=HookType.PRE_FORWARD
        )
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        
        result = controller.modify_activations(module, input_tensor, None)
        assert result is not None
        assert torch.allclose(result, input_tensor + 1.0)

    def test_modify_activations_none_tensor(self):
        """Test modify_activations with None tensor."""
        def identity(x):
            return x
        
        controller = FunctionController(
            layer_signature="layer_0",
            function=identity
        )
        module = nn.Linear(10, 5)
        
        result = controller.modify_activations(module, None, None)
        assert result is None

    def test_modify_activations_non_tensor(self):
        """Test modify_activations with non-tensor."""
        def identity(x):
            return x
        
        controller = FunctionController(
            layer_signature="layer_0",
            function=identity
        )
        module = nn.Linear(10, 5)
        
        # When target is not a tensor, function is not called, None is returned
        result = controller.modify_activations(module, "not a tensor", None)
        assert result is None

    def test_modify_activations_function_returns_non_tensor_raises_error(self):
        """Test that function returning non-tensor raises RuntimeError."""
        def return_string(x):
            return "not a tensor"
        
        controller = FunctionController(
            layer_signature="layer_0",
            function=return_string
        )
        module = nn.Linear(10, 5)
        output_tensor = torch.randn(2, 5)
        
        with pytest.raises(RuntimeError, match="Function must return a torch.Tensor"):
            controller.modify_activations(module, None, output_tensor)

    def test_modify_activations_function_exception_raises_error(self):
        """Test that function exception raises RuntimeError."""
        def failing_function(x):
            raise ValueError("Test error")
        
        controller = FunctionController(
            layer_signature="layer_0",
            function=failing_function
        )
        module = nn.Linear(10, 5)
        output_tensor = torch.randn(2, 5)
        
        with pytest.raises(RuntimeError, match="Error applying function"):
            controller.modify_activations(module, None, output_tensor)


class TestFunctionControllerIntegration:
    """Integration tests for FunctionController."""

    def test_function_controller_with_linear_layer(self):
        """Test FunctionController with actual linear layer."""
        def clamp_activations(x):
            return torch.clamp(x, min=-1.0, max=1.0)
        
        controller = FunctionController(
            layer_signature="layer_0",
            function=clamp_activations
        )
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5) * 5.0  # Large values
        
        result = controller.modify_activations(module, input_tensor, output_tensor)
        assert result is not None
        assert torch.all(result >= -1.0)
        assert torch.all(result <= 1.0)

    def test_function_controller_with_lambda(self):
        """Test FunctionController with lambda function."""
        controller = FunctionController(
            layer_signature="layer_0",
            function=lambda x: x * 0.5
        )
        module = nn.Linear(10, 5)
        output_tensor = torch.randn(2, 5)
        
        result = controller.modify_activations(module, None, output_tensor)
        assert result is not None
        assert torch.allclose(result, output_tensor * 0.5)

    def test_function_controller_modifies_output_in_place_forward_hook(self):
        """Test that FunctionController modifies output in-place for FORWARD hooks."""
        def scale_by_two(x):
            return x * 2.0
        
        controller = FunctionController(
            layer_signature="layer_0",
            function=scale_by_two,
            hook_type=HookType.FORWARD
        )
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5)
        original_value = output_tensor.clone()
        
        controller._handle_forward(module, (input_tensor,), output_tensor)
        
        # Output tensor should be modified in-place
        assert torch.allclose(output_tensor, original_value * 2.0)
        assert not torch.allclose(output_tensor, original_value)

    def test_function_controller_modifies_tuple_output_in_place(self):
        """Test that FunctionController modifies tuple output in-place."""
        def add_one(x):
            return x + 1.0
        
        controller = FunctionController(
            layer_signature="layer_0",
            function=add_one,
            hook_type=HookType.FORWARD
        )
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5)
        other_tensor = torch.randn(2, 3)
        output_tuple = (output_tensor, other_tensor)
        original_value = output_tensor.clone()
        
        controller._handle_forward(module, (input_tensor,), output_tuple)
        
        # First tensor in tuple should be modified in-place
        assert torch.allclose(output_tuple[0], original_value + 1.0)
        assert not torch.allclose(output_tuple[0], original_value)
        # Second tensor should be unchanged
        assert torch.allclose(output_tuple[1], other_tensor)

    def test_function_controller_modifies_object_with_last_hidden_state(self):
        """Test that FunctionController modifies object with last_hidden_state."""
        def multiply_by_three(x):
            return x * 3.0
        
        controller = FunctionController(
            layer_signature="layer_0",
            function=multiply_by_three,
            hook_type=HookType.FORWARD
        )
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        
        class OutputObject:
            def __init__(self):
                self.last_hidden_state = torch.randn(2, 5)
        
        output_obj = OutputObject()
        original_value = output_obj.last_hidden_state.clone()
        
        controller._handle_forward(module, (input_tensor,), output_obj)
        
        # last_hidden_state should be modified
        assert torch.allclose(output_obj.last_hidden_state, original_value * 3.0)
        assert not torch.allclose(output_obj.last_hidden_state, original_value)

