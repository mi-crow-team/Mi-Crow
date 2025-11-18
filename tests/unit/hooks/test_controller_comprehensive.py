"""Comprehensive tests for Controller hook functionality."""
import pytest
import torch
from torch import nn

from amber.hooks.controller import Controller
from amber.hooks.hook import HookType


class TestControllerPreForward:
    """Test Controller with PRE_FORWARD hooks."""
    
    def test_pre_forward_with_tensor_input(self):
        """Test pre_forward hook with tensor input."""
        class TestController(Controller):
            def modify_activations(self, module, inputs, output):
                # Return modified input
                return inputs * 2.0
        
        controller = TestController(hook_type=HookType.PRE_FORWARD)
        
        module = nn.Linear(16, 16)
        input_tensor = torch.randn(5, 16)
        inputs = (input_tensor,)
        
        result = controller._hook_fn(module, inputs, None)
        
        # Should return modified inputs tuple
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert result[0].shape == input_tensor.shape
    
    def test_pre_forward_with_tuple_input(self):
        """Test pre_forward hook with tuple input containing tensor."""
        class TestController(Controller):
            def modify_activations(self, module, inputs, output):
                return inputs * 2.0
        
        controller = TestController(hook_type=HookType.PRE_FORWARD)
        
        module = nn.Linear(16, 16)
        input_tensor = torch.randn(5, 16)
        inputs = ((input_tensor,),)
        
        result = controller._hook_fn(module, inputs, None)
        
        assert result is not None
        assert isinstance(result, tuple)
    
    def test_pre_forward_with_empty_input(self):
        """Test pre_forward hook with empty input."""
        class TestController(Controller):
            def modify_activations(self, module, inputs, output):
                return inputs * 2.0
        
        controller = TestController(hook_type=HookType.PRE_FORWARD)
        
        module = nn.Linear(16, 16)
        inputs = ()
        
        result = controller._hook_fn(module, inputs, None)
        
        # Should return None when no tensor found
        assert result is None
    
    def test_pre_forward_with_non_tensor_input(self):
        """Test pre_forward hook with non-tensor input."""
        class TestController(Controller):
            def modify_activations(self, module, inputs, output):
                return inputs * 2.0
        
        controller = TestController(hook_type=HookType.PRE_FORWARD)
        
        module = nn.Linear(16, 16)
        inputs = ("not_a_tensor",)
        
        result = controller._hook_fn(module, inputs, None)
        
        # Should return None when no tensor found
        assert result is None
    
    def test_pre_forward_when_disabled(self):
        """Test pre_forward hook when disabled."""
        class TestController(Controller):
            def modify_activations(self, module, inputs, output):
                return inputs * 2.0
        
        controller = TestController(hook_type=HookType.PRE_FORWARD)
        controller.disable()
        
        module = nn.Linear(16, 16)
        input_tensor = torch.randn(5, 16)
        inputs = (input_tensor,)
        
        result = controller._hook_fn(module, inputs, None)
        
        # Should return None when disabled
        assert result is None


class TestControllerForward:
    """Test Controller with FORWARD hooks."""
    
    def test_forward_with_tensor_output(self):
        """Test forward hook with tensor output."""
        class TestController(Controller):
            def modify_activations(self, module, inputs, output):
                return output * 2.0
        
        controller = TestController(hook_type=HookType.FORWARD)
        
        module = nn.Linear(16, 16)
        input_tensor = torch.randn(5, 16)
        inputs = (input_tensor,)
        output = torch.randn(5, 16)
        
        result = controller._hook_fn(module, inputs, output)
        
        # Forward hooks return None (can't modify output in PyTorch)
        assert result is None
    
    def test_forward_with_tuple_output(self):
        """Test forward hook with tuple output."""
        class TestController(Controller):
            def modify_activations(self, module, inputs, output):
                return output[0] * 2.0
        
        controller = TestController(hook_type=HookType.FORWARD)
        
        module = nn.Linear(16, 16)
        input_tensor = torch.randn(5, 16)
        inputs = (input_tensor,)
        output = (torch.randn(5, 16), torch.randn(5, 8))
        
        result = controller._hook_fn(module, inputs, output)
        
        # Forward hooks return None
        assert result is None
    
    def test_forward_with_none_output(self):
        """Test forward hook with None output."""
        class TestController(Controller):
            def modify_activations(self, module, inputs, output):
                return output * 2.0
        
        controller = TestController(hook_type=HookType.FORWARD)
        
        module = nn.Linear(16, 16)
        inputs = (torch.randn(5, 16),)
        
        result = controller._hook_fn(module, inputs, None)
        
        # Should return None when output is None
        assert result is None
    
    def test_forward_when_disabled(self):
        """Test forward hook when disabled."""
        class TestController(Controller):
            def modify_activations(self, module, inputs, output):
                return output * 2.0
        
        controller = TestController(hook_type=HookType.FORWARD)
        controller.disable()
        
        module = nn.Linear(16, 16)
        inputs = (torch.randn(5, 16),)
        output = torch.randn(5, 16)
        
        result = controller._hook_fn(module, inputs, output)
        
        # Should return None when disabled
        assert result is None
    
    def test_forward_with_exception_handling(self):
        """Test forward hook handles exceptions gracefully."""
        class TestController(Controller):
            def modify_activations(self, module, inputs, output):
                raise RuntimeError("Test exception")
        
        controller = TestController(hook_type=HookType.FORWARD)
        
        module = nn.Linear(16, 16)
        inputs = (torch.randn(5, 16),)
        output = torch.randn(5, 16)
        
        result = controller._hook_fn(module, inputs, output)
        
        # Should return None on exception
        assert result is None

