"""
Unit tests for FunctionController.
"""
import pytest
import torch
from torch import nn

from amber.hooks import FunctionController, HookType
from amber.hooks.controller import Controller


class SimpleLayer(nn.Module):
    """Simple test layer that returns a single tensor."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TupleOutputLayer(nn.Module):
    """Layer that returns a tuple of tensors."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> tuple:
        out = self.linear(x)
        return (out, out * 2)


def test_function_controller_is_controller():
    """Test that FunctionController is a subclass of Controller."""
    controller = FunctionController(
        layer_signature="test_layer",
        function=lambda x: x
    )
    assert isinstance(controller, Controller)
    assert isinstance(controller, FunctionController)


def test_function_controller_initialization():
    """Test FunctionController initialization."""
    def test_fn(x: torch.Tensor) -> torch.Tensor:
        return x * 2
    
    controller = FunctionController(
        layer_signature="layer_0",
        function=test_fn,
        hook_type=HookType.FORWARD,
        hook_id="test_id"
    )
    
    assert controller.layer_signature == "layer_0"
    assert controller.function == test_fn
    assert controller.hook_type == HookType.FORWARD
    assert controller.id == "test_id"
    assert controller.enabled is True


def test_function_controller_default_values():
    """Test FunctionController with default values."""
    controller = FunctionController(
        layer_signature=0,
        function=lambda x: x
    )
    
    assert controller.layer_signature == 0
    assert controller.hook_type == HookType.FORWARD
    assert controller.id is not None
    assert controller.enabled is True


def test_function_controller_single_tensor_forward():
    """Test FunctionController with single tensor output on forward hook."""
    layer = SimpleLayer(dim=4)
    x = torch.randn(2, 4)
    
    scale_factor = 3.0
    controller = FunctionController(
        layer_signature="test",
        function=lambda tensor: tensor * scale_factor,
        hook_type=HookType.FORWARD
    )
    
    # Get the hook function
    hook_fn = controller.get_torch_hook()
    
    # Simulate forward pass
    output = layer(x)
    modified_output = hook_fn(layer, (x,), output)
    
    # Check that output was modified
    expected = output * scale_factor
    assert torch.allclose(modified_output, expected)
    assert not torch.allclose(modified_output, output)


def test_function_controller_pre_forward_returns_tuple_unchanged():
    """Test that FunctionController with pre_forward hook returns tuple inputs unchanged."""
    layer = SimpleLayer(dim=4)
    x = torch.randn(2, 4)
    
    scale_factor = 2.5
    controller = FunctionController(
        layer_signature="test",
        function=lambda tensor: tensor * scale_factor,
        hook_type=HookType.PRE_FORWARD
    )
    
    # Get the hook function
    hook_fn = controller.get_torch_hook()
    
    # Simulate pre_forward hook - inputs is a tuple with a single tensor
    # The implementation: target = inputs (for PRE_FORWARD)
    # Since inputs is (x,), which is a tuple (not a tensor), it returns unchanged
    modified_inputs = hook_fn(layer, (x,))
    
    # Since the target is a tuple (not a tensor), it should be returned unchanged
    assert isinstance(modified_inputs, tuple)
    assert len(modified_inputs) == 1
    assert torch.allclose(modified_inputs[0], x)  # Unchanged because tuple is not a tensor


def test_function_controller_tuple_output_unchanged():
    """Test that FunctionController returns tuple outputs unchanged (only handles single tensors)."""
    layer = TupleOutputLayer(dim=4)
    x = torch.randn(2, 4)
    
    scale_factor = 1.5
    controller = FunctionController(
        layer_signature="test",
        function=lambda tensor: tensor * scale_factor,
        hook_type=HookType.FORWARD
    )
    
    hook_fn = controller.get_torch_hook()
    output = layer(x)
    modified_output = hook_fn(layer, (x,), output)
    
    # Since output is a tuple (not a single tensor), it should be returned unchanged
    assert isinstance(modified_output, tuple)
    assert len(modified_output) == 2
    assert modified_output == output  # Unchanged
    assert torch.allclose(modified_output[0], output[0])  # Unchanged
    assert torch.allclose(modified_output[1], output[1])  # Unchanged


def test_function_controller_complex_function():
    """Test FunctionController with a more complex function."""
    layer = SimpleLayer(dim=4)
    x = torch.randn(2, 4)
    
    def normalize_and_scale(tensor: torch.Tensor) -> torch.Tensor:
        norm = tensor.norm(dim=-1, keepdim=True)
        return tensor / (norm + 1e-8) * 5.0
    
    controller = FunctionController(
        layer_signature="test",
        function=normalize_and_scale,
        hook_type=HookType.FORWARD
    )
    
    hook_fn = controller.get_torch_hook()
    output = layer(x)
    modified_output = hook_fn(layer, (x,), output)
    
    # Verify normalization was applied
    norms = modified_output.norm(dim=-1)
    # After normalization and scaling, norms should be approximately 5.0
    assert torch.allclose(norms, torch.ones_like(norms) * 5.0, rtol=1e-5)


def test_function_controller_enable_disable():
    """Test that FunctionController respects enable/disable."""
    layer = SimpleLayer(dim=4)
    x = torch.randn(2, 4)
    
    controller = FunctionController(
        layer_signature="test",
        function=lambda tensor: tensor * 2.0,
        hook_type=HookType.FORWARD
    )
    
    hook_fn = controller.get_torch_hook()
    output = layer(x)
    
    # When enabled, should modify
    controller.enable()
    modified_enabled = hook_fn(layer, (x,), output)
    assert not torch.allclose(modified_enabled, output)
    
    # When disabled, should return None (which means no modification)
    controller.disable()
    modified_disabled = hook_fn(layer, (x,), output)
    assert modified_disabled is None


def test_function_controller_empty_tuple():
    """Test FunctionController with empty tuple (returns unchanged)."""
    controller = FunctionController(
        layer_signature="test",
        function=lambda tensor: tensor * 2.0,
        hook_type=HookType.PRE_FORWARD
    )
    
    hook_fn = controller.get_torch_hook()
    modified = hook_fn(None, ())
    
    # Should return empty tuple as-is (not a tensor, so unchanged)
    assert modified == ()


def test_function_controller_with_actual_module():
    """Test FunctionController integrated with an actual PyTorch module (forward hook)."""
    model = SimpleLayer(dim=8)
    x = torch.randn(3, 8)
    
    # Get baseline output without hook
    model.linear.eval()
    with torch.no_grad():
        baseline_output = model(x)
    
    # Register hook manually
    controller = FunctionController(
        layer_signature="linear",
        function=lambda tensor: tensor * 2.0,
        hook_type=HookType.FORWARD
    )
    
    hook_fn = controller.get_torch_hook()
    handle = model.linear.register_forward_hook(hook_fn)
    
    try:
        # Run with hook - output should be doubled (output is a single tensor)
        output_with_hook = model(x)
        
        # Output with hook should be 2x baseline
        assert torch.allclose(output_with_hook, baseline_output * 2.0, rtol=1e-5)
    finally:
        handle.remove()


def test_function_controller_lambda_vs_named_function():
    """Test FunctionController works with both lambda and named functions."""
    x = torch.randn(2, 4)
    
    # Test with lambda
    lambda_controller = FunctionController(
        layer_signature="test",
        function=lambda t: t * 2.0,
        hook_type=HookType.FORWARD
    )
    lambda_hook = lambda_controller.get_torch_hook()
    lambda_result = lambda_hook(None, (), x)
    assert torch.allclose(lambda_result, x * 2.0)
    
    # Test with named function
    def named_function(tensor: torch.Tensor) -> torch.Tensor:
        return tensor * 2.0
    
    named_controller = FunctionController(
        layer_signature="test",
        function=named_function,
        hook_type=HookType.FORWARD
    )
    named_hook = named_controller.get_torch_hook()
    named_result = named_hook(None, (), x)
    
    assert torch.allclose(lambda_result, named_result)
    assert torch.allclose(named_result, x * 2.0)


def test_function_controller_with_string_hook_type():
    """Test FunctionController with string hook type instead of enum."""
    controller = FunctionController(
        layer_signature="test",
        function=lambda x: x * 2.0,
        hook_type="forward"  # String instead of HookType enum
    )
    assert controller.hook_type == HookType.FORWARD
    
    controller_pre = FunctionController(
        layer_signature="test",
        function=lambda x: x * 2.0,
        hook_type="pre_forward"  # String instead of HookType enum
    )
    assert controller_pre.hook_type == HookType.PRE_FORWARD


def test_function_controller_returns_list_unchanged():
    """Test that FunctionController returns list outputs unchanged."""
    class ListOutputLayer(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
        
        def forward(self, x: torch.Tensor) -> list:
            out = self.linear(x)
            return [out, out * 2]
    
    layer = ListOutputLayer(dim=4)
    x = torch.randn(2, 4)
    
    controller = FunctionController(
        layer_signature="test",
        function=lambda tensor: tensor * 1.5,
        hook_type=HookType.FORWARD
    )
    
    hook_fn = controller.get_torch_hook()
    output = layer(x)
    modified_output = hook_fn(layer, (x,), output)
    
    # Since output is a list (not a single tensor), it should be returned unchanged
    assert isinstance(modified_output, list)
    assert len(modified_output) == 2
    assert modified_output == output  # Unchanged


def test_function_controller_pre_forward_with_single_tensor_in_tuple():
    """Test pre_forward when target is a tuple containing a single tensor."""
    # When pre_forward receives (x,) where x is a tensor, 
    # target is the tuple, not the tensor, so it's returned unchanged
    controller = FunctionController(
        layer_signature="test",
        function=lambda tensor: tensor * 3.0,
        hook_type=HookType.PRE_FORWARD
    )
    
    hook_fn = controller.get_torch_hook()
    x = torch.randn(2, 4)
    
    # Pre_forward receives inputs as tuple
    result = hook_fn(None, (x,))
    
    # Since target is a tuple (not a tensor), it's returned unchanged
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert torch.allclose(result[0], x)


def test_function_controller_direct_modify_activations_call():
    """Test modify_activations method directly."""
    controller = FunctionController(
        layer_signature="test",
        function=lambda tensor: tensor * 2.0,
        hook_type=HookType.FORWARD
    )
    
    x = torch.randn(2, 4)
    
    # Direct call to modify_activations with tensor
    result = controller.modify_activations(None, (x,), x)
    assert torch.allclose(result, x * 2.0)
    
    # Direct call with tuple (should return unchanged)
    tuple_output = (x, x * 2)
    result_tuple = controller.modify_activations(None, (x,), tuple_output)
    assert result_tuple == tuple_output


def test_function_controller_pre_forward_direct_modify_activations():
    """Test modify_activations directly with PRE_FORWARD hook type."""
    controller = FunctionController(
        layer_signature="test",
        function=lambda tensor: tensor * 2.0,
        hook_type=HookType.PRE_FORWARD
    )
    
    x = torch.randn(2, 4)
    
    # For PRE_FORWARD, target = inputs (which is a tuple)
    result = controller.modify_activations(None, (x,), None)
    assert isinstance(result, tuple)
    assert result == (x,)  # Returns unchanged since tuple is not a tensor


def test_function_controller_with_hook_id():
    """Test FunctionController with explicit hook_id."""
    controller = FunctionController(
        layer_signature="test",
        function=lambda x: x,
        hook_id="custom_hook_id"
    )
    assert controller.id == "custom_hook_id"


def test_function_controller_function_attribute():
    """Test that function is stored correctly."""
    def custom_fn(tensor: torch.Tensor) -> torch.Tensor:
        return tensor + 1.0
    
    controller = FunctionController(
        layer_signature="test",
        function=custom_fn
    )
    
    assert controller.function == custom_fn
    x = torch.randn(2, 4)
    result = controller.function(x)
    assert torch.allclose(result, x + 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
