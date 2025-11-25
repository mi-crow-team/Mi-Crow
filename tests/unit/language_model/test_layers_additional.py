"""Additional tests for LanguageModelLayers covering untested methods."""

import pytest
from unittest.mock import Mock, MagicMock

from amber.language_model.layers import LanguageModelLayers
from amber.language_model.context import LanguageModelContext
from amber.hooks import HookType
from amber.hooks.detector import Detector
from amber.hooks.controller import Controller
from tests.unit.fixtures import create_language_model, create_mock_detector, create_mock_controller


class TestLanguageModelLayersAdditional:
    """Additional test suite for LanguageModelLayers."""

    def test_resolve_layer_by_name(self, mock_language_model):
        """Test resolving layer by name."""
        layer_names = mock_language_model.layers.get_layer_names()
        if layer_names:
            layer = mock_language_model.layers._resolve_layer(layer_names[0])
            assert layer is not None
        else:
            pytest.skip("No layers available")

    def test_resolve_layer_by_index(self, mock_language_model):
        """Test resolving layer by index."""
        layer = mock_language_model.layers._resolve_layer(0)
        assert layer is not None

    def test_get_hook_type_from_hook(self, mock_language_model):
        """Test getting hook type from hook."""
        detector = create_mock_detector(layer_signature=0)
        hook_type = mock_language_model.layers._get_hook_type_from_hook(detector)
        assert hook_type == detector.hook_type

    def test_validate_hook_registration_new_layer(self, mock_language_model):
        """Test validating hook registration on new layer."""
        detector = create_mock_detector(layer_signature=0, hook_id="new_detector")
        # Should not raise
        mock_language_model.layers._validate_hook_registration(0, detector)

    def test_validate_hook_registration_duplicate_id(self, mock_language_model):
        """Test validating hook registration with duplicate ID."""
        detector1 = create_mock_detector(layer_signature=0, hook_id="duplicate_id")
        detector2 = create_mock_detector(layer_signature=1, hook_id="duplicate_id")
        
        mock_language_model.layers.register_hook(0, detector1)
        
        with pytest.raises(ValueError, match="Hook with ID 'duplicate_id' is already registered"):
            mock_language_model.layers._validate_hook_registration(1, detector2)

    def test_validate_hook_registration_mixing_types(self, mock_language_model):
        """Test validating hook registration when mixing types on same layer."""
        detector = create_mock_detector(layer_signature=0, hook_id="detector_1")
        controller = create_mock_controller(layer_signature=0, hook_id="controller_1")
        
        mock_language_model.layers.register_hook(0, detector)
        
        with pytest.raises(ValueError, match="Cannot register Controller hook"):
            mock_language_model.layers._validate_hook_registration(0, controller)

    def test_get_existing_hook_types_detector(self, mock_language_model):
        """Test getting existing hook types when detector is registered."""
        detector = create_mock_detector(layer_signature=0, hook_id="detector_1")
        mock_language_model.layers.register_hook(0, detector)
        
        types = mock_language_model.layers._get_existing_hook_types(0)
        assert "Detector" in types

    def test_get_existing_hook_types_controller(self, mock_language_model):
        """Test getting existing hook types when controller is registered."""
        controller = create_mock_controller(layer_signature=0, hook_id="controller_1")
        mock_language_model.layers.register_hook(0, controller)
        
        types = mock_language_model.layers._get_existing_hook_types(0)
        assert "Controller" in types

    def test_validate_hook_registration_dual_inheritance_with_detector(self, mock_language_model):
        """Test that dual-inheritance hook can be registered with existing Detector."""
        from amber.hooks.controller import Controller
        from amber.hooks.detector import Detector
        from amber.hooks.hook import HookType
        
        # Create a dual-inheritance hook
        class DualHook(Controller, Detector):
            def __init__(self, hook_id):
                Controller.__init__(self, hook_id=hook_id)
                Detector.__init__(self, hook_id=hook_id)
            
            def modify_activations(self, module, inputs, output):
                return output
            
            def process_activations(self, module, input, output):
                pass
        
        detector = create_mock_detector(layer_signature=0, hook_id="detector_1")
        dual_hook = DualHook(hook_id="dual_1")
        
        mock_language_model.layers.register_hook(0, detector)
        
        # Should not raise - dual hook is compatible with Detector
        mock_language_model.layers._validate_hook_registration(0, dual_hook)

    def test_validate_hook_registration_dual_inheritance_with_controller(self, mock_language_model):
        """Test that dual-inheritance hook can be registered with existing Controller."""
        from amber.hooks.controller import Controller
        from amber.hooks.detector import Detector
        
        class DualHook(Controller, Detector):
            def __init__(self, hook_id):
                Controller.__init__(self, hook_id=hook_id)
                Detector.__init__(self, hook_id=hook_id)
            
            def modify_activations(self, module, inputs, output):
                return output
            
            def process_activations(self, module, input, output):
                pass
        
        controller = create_mock_controller(layer_signature=0, hook_id="controller_1")
        dual_hook = DualHook(hook_id="dual_1")
        
        mock_language_model.layers.register_hook(0, controller)
        
        # Should not raise - dual hook is compatible with Controller
        mock_language_model.layers._validate_hook_registration(0, dual_hook)

    def test_validate_hook_registration_dual_inheritance_with_dual_hook(self, mock_language_model):
        """Test that dual-inheritance hook can be registered with existing dual hook."""
        from amber.hooks.controller import Controller
        from amber.hooks.detector import Detector
        
        class DualHook(Controller, Detector):
            def __init__(self, hook_id):
                Controller.__init__(self, hook_id=hook_id)
                Detector.__init__(self, hook_id=hook_id)
            
            def modify_activations(self, module, inputs, output):
                return output
            
            def process_activations(self, module, input, output):
                pass
        
        dual_hook1 = DualHook(hook_id="dual_1")
        dual_hook2 = DualHook(hook_id="dual_2")
        
        mock_language_model.layers.register_hook(0, dual_hook1)
        
        # Should not raise - dual hooks are compatible with each other
        mock_language_model.layers._validate_hook_registration(0, dual_hook2)

    def test_get_existing_hook_types_dual_inheritance(self, mock_language_model):
        """Test getting existing hook types when dual-inheritance hook is registered."""
        from amber.hooks.controller import Controller
        from amber.hooks.detector import Detector
        
        class DualHook(Controller, Detector):
            def __init__(self, hook_id):
                Controller.__init__(self, hook_id=hook_id)
                Detector.__init__(self, hook_id=hook_id)
            
            def modify_activations(self, module, inputs, output):
                return output
            
            def process_activations(self, module, input, output):
                pass
        
        dual_hook = DualHook(hook_id="dual_1")
        mock_language_model.layers.register_hook(0, dual_hook)
        
        types = mock_language_model.layers._get_existing_hook_types(0)
        # Dual hook should be recognized as both types
        assert "Controller" in types
        assert "Detector" in types

    def test_validate_hook_registration_single_type_with_dual_hook_allowed(self, mock_language_model):
        """Test that single-type hook can be registered with dual hook (they're compatible)."""
        from amber.hooks.controller import Controller
        from amber.hooks.detector import Detector
        
        class DualHook(Controller, Detector):
            def __init__(self, hook_id):
                Controller.__init__(self, hook_id=hook_id)
                Detector.__init__(self, hook_id=hook_id)
            
            def modify_activations(self, module, inputs, output):
                return output
            
            def process_activations(self, module, input, output):
                pass
        
        dual_hook = DualHook(hook_id="dual_1")
        # Register dual hook first
        mock_language_model.layers.register_hook(0, dual_hook)
        
        # A Controller should be allowed because dual hook has "Controller" in its types
        controller = create_mock_controller(layer_signature=0, hook_id="controller_1")
        mock_language_model.layers._validate_hook_registration(0, controller)
        
        # A Detector should also be allowed because dual hook has "Detector" in its types
        detector = create_mock_detector(layer_signature=0, hook_id="detector_1")
        mock_language_model.layers._validate_hook_registration(0, detector)

    def test_register_hook_with_hook_type_parameter(self, mock_language_model):
        """Test registering hook with explicit hook_type parameter."""
        detector = create_mock_detector(layer_signature=0, hook_id="detector_1")
        hook_id = mock_language_model.layers.register_hook(0, detector, hook_type=HookType.FORWARD)
        
        assert hook_id == "detector_1"

    def test_register_hook_with_string_hook_type(self, mock_language_model):
        """Test registering hook with string hook_type."""
        detector = create_mock_detector(layer_signature=0, hook_id="detector_1")
        hook_id = mock_language_model.layers.register_hook(0, detector, hook_type="forward")
        
        assert hook_id == "detector_1"

    def test_register_hook_pre_forward(self, mock_language_model):
        """Test registering PRE_FORWARD hook."""
        controller = create_mock_controller(layer_signature=0, hook_id="controller_1")
        controller.hook_type = HookType.PRE_FORWARD
        
        hook_id = mock_language_model.layers.register_hook(0, controller)
        
        assert hook_id == "controller_1"
        hooks = mock_language_model.layers.get_hooks(layer_signature=0, hook_type=HookType.PRE_FORWARD)
        assert len(hooks) == 1

    def test_unregister_hook_by_instance(self, mock_language_model):
        """Test unregistering hook by Hook instance."""
        detector = create_mock_detector(layer_signature=0, hook_id="detector_1")
        mock_language_model.layers.register_hook(0, detector)
        
        result = mock_language_model.layers.unregister_hook(detector)
        
        assert result is True
        hooks = mock_language_model.layers.get_detectors()
        assert len([h for h in hooks if h.id == "detector_1"]) == 0

    def test_unregister_hook_by_id(self, mock_language_model):
        """Test unregistering hook by ID string."""
        detector = create_mock_detector(layer_signature=0, hook_id="detector_1")
        mock_language_model.layers.register_hook(0, detector)
        
        result = mock_language_model.layers.unregister_hook("detector_1")
        
        assert result is True

    def test_unregister_hook_not_found(self, mock_language_model):
        """Test unregistering non-existent hook."""
        result = mock_language_model.layers.unregister_hook("nonexistent_id")
        assert result is False

    def test_get_hooks_from_registry_by_layer(self, mock_language_model):
        """Test getting hooks from registry filtered by layer."""
        detector1 = create_mock_detector(layer_signature=0, hook_id="detector_1")
        detector2 = create_mock_detector(layer_signature=1, hook_id="detector_2")
        
        mock_language_model.layers.register_hook(0, detector1)
        mock_language_model.layers.register_hook(1, detector2)
        
        hooks = mock_language_model.layers._get_hooks_from_registry(0, None)
        
        assert len(hooks) == 1
        assert hooks[0].id == "detector_1"

    def test_get_hooks_from_registry_by_type(self, mock_language_model):
        """Test getting hooks from registry filtered by type."""
        detector = create_mock_detector(layer_signature=0, hook_id="detector_1")
        controller = create_mock_controller(layer_signature=1, hook_id="controller_1")
        controller.hook_type = HookType.PRE_FORWARD
        
        mock_language_model.layers.register_hook(0, detector)
        mock_language_model.layers.register_hook(1, controller)
        
        # Get all FORWARD hooks (should only get detector)
        hooks = mock_language_model.layers._get_hooks_from_registry(None, HookType.FORWARD)
        
        assert len(hooks) == 1
        assert hooks[0].id == "detector_1"

    def test_get_hooks_from_registry_all(self, mock_language_model):
        """Test getting all hooks from registry."""
        detector = create_mock_detector(layer_signature=0, hook_id="detector_1")
        controller = create_mock_controller(layer_signature=1, hook_id="controller_1")
        
        mock_language_model.layers.register_hook(0, detector)
        mock_language_model.layers.register_hook(1, controller)
        
        hooks = mock_language_model.layers._get_hooks_from_registry(None, None)
        
        assert len(hooks) == 2

    def test_get_hooks_with_string_hook_type(self, mock_language_model):
        """Test getting hooks with string hook_type."""
        detector = create_mock_detector(layer_signature=0, hook_id="detector_1")
        mock_language_model.layers.register_hook(0, detector)
        
        hooks = mock_language_model.layers.get_hooks(layer_signature=0, hook_type="forward")
        
        assert len(hooks) == 1

    def test_enable_hook_success(self, mock_language_model):
        """Test enabling hook successfully."""
        detector = create_mock_detector(layer_signature=0, hook_id="detector_1")
        detector.disable()  # Disable first
        mock_language_model.layers.register_hook(0, detector)
        
        result = mock_language_model.layers.enable_hook("detector_1")
        
        assert result is True
        assert detector.enabled is True

    def test_enable_hook_not_found(self, mock_language_model):
        """Test enabling non-existent hook."""
        result = mock_language_model.layers.enable_hook("nonexistent_id")
        assert result is False

    def test_disable_hook_success(self, mock_language_model):
        """Test disabling hook successfully."""
        detector = create_mock_detector(layer_signature=0, hook_id="detector_1")
        detector.enable()  # Enable first
        mock_language_model.layers.register_hook(0, detector)
        
        result = mock_language_model.layers.disable_hook("detector_1")
        
        assert result is True
        assert detector.enabled is False

    def test_disable_hook_not_found(self, mock_language_model):
        """Test disabling non-existent hook."""
        result = mock_language_model.layers.disable_hook("nonexistent_id")
        assert result is False

    def test_enable_all_hooks(self, mock_language_model):
        """Test enabling all hooks."""
        detector1 = create_mock_detector(layer_signature=0, hook_id="detector_1")
        detector1.disable()
        detector2 = create_mock_detector(layer_signature=1, hook_id="detector_2")
        detector2.disable()
        
        mock_language_model.layers.register_hook(0, detector1)
        mock_language_model.layers.register_hook(1, detector2)
        
        mock_language_model.layers.enable_all_hooks()
        
        assert detector1.enabled is True
        assert detector2.enabled is True

    def test_disable_all_hooks(self, mock_language_model):
        """Test disabling all hooks."""
        detector1 = create_mock_detector(layer_signature=0, hook_id="detector_1")
        detector1.enable()
        detector2 = create_mock_detector(layer_signature=1, hook_id="detector_2")
        detector2.enable()
        
        mock_language_model.layers.register_hook(0, detector1)
        mock_language_model.layers.register_hook(1, detector2)
        
        mock_language_model.layers.disable_all_hooks()
        
        assert detector1.enabled is False
        assert detector2.enabled is False

    def test_get_controllers_filtered(self, mock_language_model):
        """Test getting only controllers."""
        detector = create_mock_detector(layer_signature=0, hook_id="detector_1")
        controller = create_mock_controller(layer_signature=1, hook_id="controller_1")
        
        mock_language_model.layers.register_hook(0, detector)
        mock_language_model.layers.register_hook(1, controller)
        
        controllers = mock_language_model.layers.get_controllers()
        
        assert len(controllers) == 1
        assert controllers[0].id == "controller_1"

    def test_get_detectors_filtered(self, mock_language_model):
        """Test getting only detectors."""
        detector = create_mock_detector(layer_signature=0, hook_id="detector_1")
        controller = create_mock_controller(layer_signature=1, hook_id="controller_1")
        
        mock_language_model.layers.register_hook(0, detector)
        mock_language_model.layers.register_hook(1, controller)
        
        detectors = mock_language_model.layers.get_detectors()
        
        assert len(detectors) == 1
        assert detectors[0].id == "detector_1"

