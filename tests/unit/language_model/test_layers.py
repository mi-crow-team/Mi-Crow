"""Tests for LanguageModelLayers."""

import pytest
from unittest.mock import Mock
from torch import nn

from amber.language_model.layers import LanguageModelLayers
from amber.language_model.context import LanguageModelContext
from amber.language_model.language_model import LanguageModel
from amber.hooks.hook import HookType
from tests.unit.fixtures.language_models import create_language_model_from_mock
from tests.unit.fixtures.stores import create_temp_store
from tests.unit.fixtures.hooks import create_mock_detector


class TestLanguageModelLayers:
    """Tests for LanguageModelLayers."""

    def test_layers_initialization(self, temp_store):
        """Test layers initialization."""
        lm = create_language_model_from_mock(temp_store)
        layers = lm.layers
        
        assert layers.context == lm.context
        assert len(layers.name_to_layer) > 0
        assert len(layers.idx_to_layer) > 0

    def test_get_layer_names(self, temp_store):
        """Test getting layer names."""
        lm = create_language_model_from_mock(temp_store)
        names = lm.layers.get_layer_names()
        
        assert isinstance(names, list)
        assert len(names) > 0

    def test_get_layer_by_name(self, temp_store):
        """Test getting layer by name."""
        lm = create_language_model_from_mock(temp_store)
        names = lm.layers.get_layer_names()
        
        if names:
            layer = lm.layers._get_layer_by_name(names[0])
            assert isinstance(layer, nn.Module)

    def test_get_layer_by_name_not_found_raises_error(self, temp_store):
        """Test that getting non-existent layer raises ValueError."""
        lm = create_language_model_from_mock(temp_store)
        
        with pytest.raises(ValueError, match="Layer name 'nonexistent' not found"):
            lm.layers._get_layer_by_name("nonexistent")

    def test_get_layer_by_index(self, temp_store):
        """Test getting layer by index."""
        lm = create_language_model_from_mock(temp_store)
        layer = lm.layers._get_layer_by_index(0)
        assert isinstance(layer, nn.Module)

    def test_get_layer_by_index_not_found_raises_error(self, temp_store):
        """Test that getting non-existent index raises ValueError."""
        lm = create_language_model_from_mock(temp_store)
        
        with pytest.raises(ValueError, match="Layer index '999' not found"):
            lm.layers._get_layer_by_index(999)

    def test_register_detector(self, temp_store):
        """Test registering a detector."""
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=0)
        
        lm.layers.register_hook(0, detector)
        detectors = lm.layers.get_detectors()
        assert len(detectors) > 0

    def test_register_controller(self, temp_store):
        """Test registering a controller."""
        lm = create_language_model_from_mock(temp_store)
        from tests.unit.fixtures.hooks import create_mock_controller
        
        controller = create_mock_controller(layer_signature=0)
        lm.layers.register_hook(0, controller)
        controllers = lm.layers.get_controllers()
        assert len(controllers) > 0

    def test_flatten_layer_names_raises_when_model_none(self, temp_store):
        """Test that _flatten_layer_names raises ValueError when model is None."""
        from amber.language_model.context import LanguageModelContext
        
        context = LanguageModelContext(language_model=Mock())
        context.model = None
        
        with pytest.raises(ValueError, match="Model must be initialized"):
            LanguageModelLayers(context)

    def test_get_layer_by_name_auto_flattens_when_empty(self, temp_store):
        """Test that _get_layer_by_name auto-flattens when name_to_layer is empty."""
        lm = create_language_model_from_mock(temp_store)
        names = lm.layers.get_layer_names()
        
        lm.layers.name_to_layer.clear()
        
        if names:
            layer = lm.layers._get_layer_by_name(names[0])
            assert isinstance(layer, nn.Module)
            assert len(lm.layers.name_to_layer) > 0

    def test_get_layer_by_index_auto_flattens_when_empty(self, temp_store):
        """Test that _get_layer_by_index auto-flattens when idx_to_layer is empty."""
        lm = create_language_model_from_mock(temp_store)
        
        lm.layers.idx_to_layer.clear()
        
        layer = lm.layers._get_layer_by_index(0)
        assert isinstance(layer, nn.Module)
        assert len(lm.layers.idx_to_layer) > 0

    def test_print_layer_names(self, temp_store, capsys):
        """Test print_layer_names method."""
        lm = create_language_model_from_mock(temp_store)
        lm.layers.print_layer_names()
        
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_register_forward_hook_for_layer(self, temp_store):
        """Test register_forward_hook_for_layer."""
        lm = create_language_model_from_mock(temp_store)
        names = lm.layers.get_layer_names()
        
        if names:
            hook_called = []
            def test_hook(module, input, output):
                hook_called.append(True)
            
            handle = lm.layers.register_forward_hook_for_layer(names[0], test_hook)
            assert handle is not None
            
            handle.remove()

    def test_register_pre_forward_hook_for_layer(self, temp_store):
        """Test register_pre_forward_hook_for_layer."""
        lm = create_language_model_from_mock(temp_store)
        names = lm.layers.get_layer_names()
        
        if names:
            hook_called = []
            def test_hook(module, input):
                hook_called.append(True)
                return None
            
            handle = lm.layers.register_pre_forward_hook_for_layer(names[0], test_hook)
            assert handle is not None
            
            handle.remove()

    def test_register_forward_hook_with_args(self, temp_store):
        """Test register_forward_hook_for_layer with hook_args."""
        lm = create_language_model_from_mock(temp_store)
        names = lm.layers.get_layer_names()
        
        if names:
            def test_hook(module, input, output):
                pass
            
            handle = lm.layers.register_forward_hook_for_layer(
                names[0], 
                test_hook, 
                hook_args={"prepend": False}
            )
            assert handle is not None
            handle.remove()

    def test_register_hook_sets_layer_signature(self, temp_store):
        """Test that register_hook sets layer_signature on hook."""
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=None)
        
        lm.layers.register_hook(0, detector)
        
        assert detector.layer_signature == 0

    def test_register_hook_sets_context(self, temp_store):
        """Test that register_hook sets context on hook."""
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=None)
        
        lm.layers.register_hook(0, detector)
        
        assert detector.context == lm.context

    def test_unregister_hook_removes_from_registry(self, temp_store):
        """Test that unregister_hook removes hook from registry."""
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=0)
        
        hook_id = lm.layers.register_hook(0, detector)
        hooks_before = lm.layers.get_hooks(layer_signature=0)
        assert len(hooks_before) == 1
        
        result = lm.layers.unregister_hook(hook_id)
        assert result is True
        
        hooks_after = lm.layers.get_hooks(layer_signature=0)
        assert len(hooks_after) == 0

    def test_unregister_hook_removes_pytorch_hook(self, temp_store):
        """Test that unregister_hook removes PyTorch hook."""
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=0)
        
        hook_id = lm.layers.register_hook(0, detector)
        
        layer = lm.layers._get_layer_by_index(0)
        hooks_before = len([h for h in layer._forward_hooks.values()])
        
        lm.layers.unregister_hook(hook_id)
        
        hooks_after = len([h for h in layer._forward_hooks.values()])
        assert hooks_after < hooks_before

    def test_get_hooks_with_layer_filter(self, temp_store):
        """Test get_hooks filtered by layer."""
        lm = create_language_model_from_mock(temp_store)
        detector1 = create_mock_detector(layer_signature=0)
        detector2 = create_mock_detector(layer_signature=1)
        
        lm.layers.register_hook(0, detector1)
        lm.layers.register_hook(1, detector2)
        
        hooks = lm.layers.get_hooks(layer_signature=0)
        assert len(hooks) == 1
        assert hooks[0].id == detector1.id

    def test_get_hooks_with_type_filter(self, temp_store):
        """Test get_hooks filtered by hook type."""
        from amber.hooks.hook import HookType
        from tests.unit.fixtures.hooks import create_mock_controller
        
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=0)
        controller = create_mock_controller(layer_signature=1)
        controller.hook_type = HookType.PRE_FORWARD
        
        lm.layers.register_hook(0, detector)
        lm.layers.register_hook(1, controller)
        
        forward_hooks = lm.layers.get_hooks(hook_type=HookType.FORWARD)
        assert len([h for h in forward_hooks if h.id == detector.id]) == 1
        
        pre_forward_hooks = lm.layers.get_hooks(hook_type=HookType.PRE_FORWARD)
        assert len([h for h in pre_forward_hooks if h.id == controller.id]) == 1

    def test_get_hooks_with_both_filters(self, temp_store):
        """Test get_hooks filtered by both layer and type."""
        from amber.hooks.hook import HookType
        
        lm = create_language_model_from_mock(temp_store)
        detector1 = create_mock_detector(layer_signature=0)
        detector2 = create_mock_detector(layer_signature=1)
        
        lm.layers.register_hook(0, detector1)
        lm.layers.register_hook(1, detector2)
        
        hooks = lm.layers.get_hooks(layer_signature=0, hook_type=HookType.FORWARD)
        assert len(hooks) == 1
        assert hooks[0].id == detector1.id

    def test_get_hooks_returns_all_when_no_filters(self, temp_store):
        """Test get_hooks returns all hooks when no filters."""
        from tests.unit.fixtures.hooks import create_mock_controller
        
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=0)
        controller = create_mock_controller(layer_signature=1)
        
        lm.layers.register_hook(0, detector)
        lm.layers.register_hook(1, controller)
        
        hooks = lm.layers.get_hooks()
        assert len(hooks) >= 2

    def test_enable_hook_enables_specific_hook(self, temp_store):
        """Test enable_hook enables specific hook."""
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=0)
        detector.disable()
        
        hook_id = lm.layers.register_hook(0, detector)
        assert detector.enabled is False
        
        result = lm.layers.enable_hook(hook_id)
        assert result is True
        assert detector.enabled is True

    def test_disable_hook_disables_specific_hook(self, temp_store):
        """Test disable_hook disables specific hook."""
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=0)
        detector.enable()
        
        hook_id = lm.layers.register_hook(0, detector)
        assert detector.enabled is True
        
        result = lm.layers.disable_hook(hook_id)
        assert result is True
        assert detector.enabled is False

    def test_enable_all_hooks_enables_all(self, temp_store):
        """Test enable_all_hooks enables all registered hooks."""
        lm = create_language_model_from_mock(temp_store)
        detector1 = create_mock_detector(layer_signature=0)
        detector2 = create_mock_detector(layer_signature=1)
        
        detector1.disable()
        detector2.disable()
        
        lm.layers.register_hook(0, detector1)
        lm.layers.register_hook(1, detector2)
        
        lm.layers.enable_all_hooks()
        
        assert detector1.enabled is True
        assert detector2.enabled is True

    def test_disable_all_hooks_disables_all(self, temp_store):
        """Test disable_all_hooks disables all registered hooks."""
        lm = create_language_model_from_mock(temp_store)
        detector1 = create_mock_detector(layer_signature=0)
        detector2 = create_mock_detector(layer_signature=1)
        
        detector1.enable()
        detector2.enable()
        
        lm.layers.register_hook(0, detector1)
        lm.layers.register_hook(1, detector2)
        
        lm.layers.disable_all_hooks()
        
        assert detector1.enabled is False
        assert detector2.enabled is False

    def test_get_controllers_returns_only_controllers(self, temp_store):
        """Test get_controllers returns only Controller instances."""
        from tests.unit.fixtures.hooks import create_mock_controller
        
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=0)
        controller = create_mock_controller(layer_signature=1)
        
        lm.layers.register_hook(0, detector)
        lm.layers.register_hook(1, controller)
        
        controllers = lm.layers.get_controllers()
        assert len(controllers) == 1
        assert controllers[0].id == controller.id

    def test_get_detectors_returns_only_detectors(self, temp_store):
        """Test get_detectors returns only Detector instances."""
        from tests.unit.fixtures.hooks import create_mock_controller
        
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=0)
        controller = create_mock_controller(layer_signature=1)
        
        lm.layers.register_hook(0, detector)
        lm.layers.register_hook(1, controller)
        
        detectors = lm.layers.get_detectors()
        assert len(detectors) == 1
        assert detectors[0].id == detector.id

    def test_register_hook_with_string_layer_signature(self, temp_store):
        """Test register_hook with string layer signature."""
        lm = create_language_model_from_mock(temp_store)
        names = lm.layers.get_layer_names()
        
        if names:
            detector = create_mock_detector(layer_signature=None)
            hook_id = lm.layers.register_hook(names[0], detector)
            
            assert hook_id == detector.id
            assert detector.layer_signature == names[0]

    def test_register_hook_with_int_layer_signature(self, temp_store):
        """Test register_hook with int layer signature."""
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=None)
        hook_id = lm.layers.register_hook(0, detector)
        
        assert hook_id == detector.id
        assert detector.layer_signature == 0

    def test_register_hook_creates_registry_entry(self, temp_store):
        """Test that register_hook creates registry entry."""
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=0)
        
        assert 0 not in lm.context._hook_registry
        
        lm.layers.register_hook(0, detector)
        
        assert 0 in lm.context._hook_registry
        assert HookType.FORWARD in lm.context._hook_registry[0]

    def test_register_hook_adds_to_id_map(self, temp_store):
        """Test that register_hook adds hook to ID map."""
        from amber.hooks.hook import HookType
        
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=0)
        
        hook_id = lm.layers.register_hook(0, detector)
        
        assert hook_id in lm.context._hook_id_map
        layer_sig, hook_type, hook = lm.context._hook_id_map[hook_id]
        assert layer_sig == 0
        assert hook_type == HookType.FORWARD
        assert hook.id == hook_id
