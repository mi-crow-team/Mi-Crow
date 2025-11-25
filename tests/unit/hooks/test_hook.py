"""Tests for Hook base class."""

import pytest
import torch
from unittest.mock import Mock, MagicMock
from torch import nn

from amber.hooks.hook import Hook, HookType, HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT
from tests.unit.fixtures.hooks import MockDetector


class ConcreteHook(Hook):
    """Concrete implementation of Hook for testing."""

    def _hook_fn(
        self,
        module: nn.Module,
        input: HOOK_FUNCTION_INPUT,
        output: HOOK_FUNCTION_OUTPUT,
    ) -> None | HOOK_FUNCTION_INPUT:
        """Implementation of abstract method."""
        return None


class TestHookInitialization:
    """Tests for Hook initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        hook = ConcreteHook()
        assert hook.layer_signature is None
        assert hook.hook_type == HookType.FORWARD
        assert hook.id is not None
        assert hook.enabled is True

    def test_init_with_layer_signature(self):
        """Test initialization with layer signature."""
        hook = ConcreteHook(layer_signature="layer_0")
        assert hook.layer_signature == "layer_0"

    def test_init_with_hook_type_enum(self):
        """Test initialization with HookType enum."""
        hook = ConcreteHook(hook_type=HookType.PRE_FORWARD)
        assert hook.hook_type == HookType.PRE_FORWARD

    def test_init_with_hook_type_string(self):
        """Test initialization with hook type string."""
        hook = ConcreteHook(hook_type="pre_forward")
        assert hook.hook_type == HookType.PRE_FORWARD

    def test_init_with_invalid_hook_type_string_raises_error(self):
        """Test that invalid hook type string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid hook_type string"):
            ConcreteHook(hook_type="invalid")

    def test_init_with_invalid_hook_type_type_raises_error(self):
        """Test that invalid hook type type raises ValueError."""
        with pytest.raises(ValueError, match="hook_type must be HookType enum or string"):
            ConcreteHook(hook_type=123)

    def test_init_with_custom_hook_id(self):
        """Test initialization with custom hook ID."""
        hook = ConcreteHook(hook_id="custom_id")
        assert hook.id == "custom_id"

    def test_init_auto_generates_hook_id(self):
        """Test that hook ID is auto-generated if not provided."""
        hook1 = ConcreteHook()
        hook2 = ConcreteHook()
        assert hook1.id != hook2.id


class TestHookEnableDisable:
    """Tests for enable/disable functionality."""

    def test_enable(self):
        """Test enabling a hook."""
        hook = ConcreteHook()
        hook.disable()
        assert hook.enabled is False
        hook.enable()
        assert hook.enabled is True

    def test_disable(self):
        """Test disabling a hook."""
        hook = ConcreteHook()
        assert hook.enabled is True
        hook.disable()
        assert hook.enabled is False

    def test_enabled_property(self):
        """Test enabled property."""
        hook = ConcreteHook()
        assert hook.enabled is True
        hook._enabled = False
        assert hook.enabled is False


class TestHookGetTorchHook:
    """Tests for get_torch_hook method."""

    def test_get_torch_hook_forward(self):
        """Test get_torch_hook for forward hook."""
        hook = ConcreteHook(hook_type=HookType.FORWARD)
        torch_hook = hook.get_torch_hook()
        assert callable(torch_hook)

    def test_get_torch_hook_pre_forward(self):
        """Test get_torch_hook for pre-forward hook."""
        hook = ConcreteHook(hook_type=HookType.PRE_FORWARD)
        torch_hook = hook.get_torch_hook()
        assert callable(torch_hook)

    def test_torch_hook_forward_execution(self):
        """Test that forward hook wrapper executes correctly."""
        hook = ConcreteHook(hook_type=HookType.FORWARD)
        torch_hook = hook.get_torch_hook()
        
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5)
        
        # Should not raise
        result = torch_hook(module, (input_tensor,), output_tensor)
        assert result is None

    def test_torch_hook_pre_forward_execution(self):
        """Test that pre-forward hook wrapper executes correctly."""
        hook = ConcreteHook(hook_type=HookType.PRE_FORWARD)
        torch_hook = hook.get_torch_hook()
        
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        
        # Should not raise
        result = torch_hook(module, (input_tensor,))
        assert result is None

    def test_torch_hook_respects_enabled_flag(self):
        """Test that torch hook respects enabled flag."""
        hook = ConcreteHook()
        hook.disable()
        torch_hook = hook.get_torch_hook()
        
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5)
        
        # Should not raise even when disabled
        result = torch_hook(module, (input_tensor,), output_tensor)
        assert result is None

    def test_torch_hook_handles_exceptions(self):
        """Test that torch hook handles exceptions gracefully."""
        class FailingHook(ConcreteHook):
            def _hook_fn(self, module, input, output):
                raise ValueError("Test error")
        
        hook = FailingHook()
        torch_hook = hook.get_torch_hook()
        
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5)
        
        # Should not raise, but log warning
        result = torch_hook(module, (input_tensor,), output_tensor)
        assert result is None


class TestHookNormalizeHookType:
    """Tests for _normalize_hook_type method."""

    def test_normalize_hook_type_enum(self):
        """Test normalizing HookType enum."""
        hook = ConcreteHook()
        result = hook._normalize_hook_type(HookType.FORWARD)
        assert result == HookType.FORWARD

    def test_normalize_hook_type_string(self):
        """Test normalizing hook type string."""
        hook = ConcreteHook()
        result = hook._normalize_hook_type("forward")
        assert result == HookType.FORWARD

    def test_normalize_hook_type_invalid_string(self):
        """Test that invalid string raises ValueError."""
        hook = ConcreteHook()
        with pytest.raises(ValueError, match="Invalid hook_type string"):
            hook._normalize_hook_type("invalid")


class TestHookAbstractMethods:
    """Tests for abstract method requirements."""

    def test_hook_fn_must_be_implemented(self):
        """Test that _hook_fn must be implemented by subclasses."""
        # This is tested by the fact that ConcreteHook implements it
        hook = ConcreteHook()
        result = hook._hook_fn(Mock(), (), None)
        assert result is None


class TestHookIsBothControllerAndDetector:
    """Tests for _is_both_controller_and_detector method."""

    def test_is_both_controller_and_detector_false_for_hook_only(self):
        """Test that plain Hook returns False."""
        hook = ConcreteHook()
        assert hook._is_both_controller_and_detector() is False

    def test_is_both_controller_and_detector_false_for_controller_only(self):
        """Test that Controller-only hook returns False."""
        from tests.unit.fixtures.hooks import MockController
        controller = MockController()
        assert controller._is_both_controller_and_detector() is False

    def test_is_both_controller_and_detector_false_for_detector_only(self):
        """Test that Detector-only hook returns False."""
        from tests.unit.fixtures.hooks import MockDetector
        detector = MockDetector()
        assert detector._is_both_controller_and_detector() is False

    def test_is_both_controller_and_detector_true_for_dual_inheritance(self):
        """Test that hook inheriting from both Controller and Detector returns True."""
        from amber.hooks.controller import Controller
        from amber.hooks.detector import Detector
        
        class DualHook(Controller, Detector):
            def __init__(self):
                Controller.__init__(self)
                Detector.__init__(self)
            
            def modify_activations(self, module, inputs, output):
                return output
            
            def process_activations(self, module, input, output):
                pass
        
        dual_hook = DualHook()
        assert dual_hook._is_both_controller_and_detector() is True

    def test_is_both_controller_and_detector_checks_mro(self):
        """Test that method correctly checks MRO for both classes."""
        from amber.hooks.controller import Controller
        from amber.hooks.detector import Detector
        
        class DualHook(Controller, Detector):
            def __init__(self):
                Controller.__init__(self)
                Detector.__init__(self)
            
            def modify_activations(self, module, inputs, output):
                return output
            
            def process_activations(self, module, input, output):
                pass
        
        dual_hook = DualHook()
        mro_names = [cls.__name__ for cls in type(dual_hook).__mro__]
        
        # Verify both are in MRO
        assert 'Controller' in mro_names
        assert 'Detector' in mro_names
        assert dual_hook._is_both_controller_and_detector() is True

