"""Tests for hook metadata collection."""

import pytest
from unittest.mock import Mock, MagicMock

from amber.language_model.hook_metadata import collect_hooks_metadata
from amber.language_model.context import LanguageModelContext
from amber.hooks.detector import Detector
from amber.hooks.controller import Controller
from amber.hooks import HookType


class TestCollectHooksMetadata:
    """Test suite for collect_hooks_metadata."""

    def test_collect_empty_registry(self):
        """Test collecting metadata from empty registry."""
        context = Mock(spec=LanguageModelContext)
        context._hook_registry = {}
        
        result = collect_hooks_metadata(context)
        assert result == {}

    def test_collect_single_detector(self):
        """Test collecting metadata from single detector."""
        context = Mock(spec=LanguageModelContext)
        detector = Mock(spec=Detector)
        detector.id = "detector_1"
        detector.hook_type = HookType.FORWARD
        detector.layer_signature = "layer_0"
        detector.__class__.__name__ = "LayerActivationDetector"
        detector.enabled = True
        
        context._hook_registry = {
            "layer_0": {
                HookType.FORWARD: [(detector, None)]
            }
        }
        
        result = collect_hooks_metadata(context)
        
        assert "layer_0" in result
        assert len(result["layer_0"]) == 1
        assert result["layer_0"][0]["hook_id"] == "detector_1"
        assert result["layer_0"][0]["hook_type"] == HookType.FORWARD.value
        assert result["layer_0"][0]["layer_signature"] == "layer_0"
        assert result["layer_0"][0]["hook_class"] == "LayerActivationDetector"
        assert result["layer_0"][0]["enabled"] is True

    def test_collect_single_controller(self):
        """Test collecting metadata from single controller."""
        context = Mock(spec=LanguageModelContext)
        controller = Mock(spec=Controller)
        controller.id = "controller_1"
        controller.hook_type = HookType.PRE_FORWARD
        controller.layer_signature = "layer_1"
        controller.__class__.__name__ = "FunctionController"
        controller.enabled = False
        
        context._hook_registry = {
            "layer_1": {
                HookType.PRE_FORWARD: [(controller, None)]
            }
        }
        
        result = collect_hooks_metadata(context)
        
        assert "layer_1" in result
        assert len(result["layer_1"]) == 1
        assert result["layer_1"][0]["hook_id"] == "controller_1"
        assert result["layer_1"][0]["hook_type"] == HookType.PRE_FORWARD.value
        assert result["layer_1"][0]["enabled"] is False

    def test_collect_multiple_hooks_same_layer(self):
        """Test collecting metadata from multiple hooks on same layer."""
        context = Mock(spec=LanguageModelContext)
        detector1 = Mock(spec=Detector)
        detector1.id = "detector_1"
        detector1.hook_type = HookType.FORWARD
        detector1.layer_signature = "layer_0"
        detector1.__class__.__name__ = "Detector1"
        detector1.enabled = True
        
        detector2 = Mock(spec=Detector)
        detector2.id = "detector_2"
        detector2.hook_type = HookType.FORWARD
        detector2.layer_signature = "layer_0"
        detector2.__class__.__name__ = "Detector2"
        detector2.enabled = False
        
        context._hook_registry = {
            "layer_0": {
                HookType.FORWARD: [(detector1, None), (detector2, None)]
            }
        }
        
        result = collect_hooks_metadata(context)
        
        assert "layer_0" in result
        assert len(result["layer_0"]) == 2
        assert result["layer_0"][0]["hook_id"] == "detector_1"
        assert result["layer_0"][1]["hook_id"] == "detector_2"

    def test_collect_multiple_layers(self):
        """Test collecting metadata from multiple layers."""
        context = Mock(spec=LanguageModelContext)
        detector1 = Mock(spec=Detector)
        detector1.id = "detector_1"
        detector1.hook_type = HookType.FORWARD
        detector1.layer_signature = "layer_0"
        detector1.__class__.__name__ = "Detector"
        detector1.enabled = True
        
        controller1 = Mock(spec=Controller)
        controller1.id = "controller_1"
        controller1.hook_type = HookType.PRE_FORWARD
        controller1.layer_signature = "layer_1"
        controller1.__class__.__name__ = "Controller"
        controller1.enabled = True
        
        context._hook_registry = {
            "layer_0": {
                HookType.FORWARD: [(detector1, None)]
            },
            "layer_1": {
                HookType.PRE_FORWARD: [(controller1, None)]
            }
        }
        
        result = collect_hooks_metadata(context)
        
        assert "layer_0" in result
        assert "layer_1" in result
        assert len(result["layer_0"]) == 1
        assert len(result["layer_1"]) == 1

    def test_collect_with_int_layer_signature(self):
        """Test collecting metadata with integer layer signature."""
        context = Mock(spec=LanguageModelContext)
        detector = Mock(spec=Detector)
        detector.id = "detector_1"
        detector.hook_type = HookType.FORWARD
        detector.layer_signature = 0
        detector.__class__.__name__ = "Detector"
        detector.enabled = True
        
        context._hook_registry = {
            0: {
                HookType.FORWARD: [(detector, None)]
            }
        }
        
        result = collect_hooks_metadata(context)
        
        assert "0" in result
        assert result["0"][0]["layer_signature"] == "0"

    def test_collect_with_none_layer_signature(self):
        """Test collecting metadata with None layer signature."""
        context = Mock(spec=LanguageModelContext)
        detector = Mock(spec=Detector)
        detector.id = "detector_1"
        detector.hook_type = HookType.FORWARD
        detector.layer_signature = None
        detector.__class__.__name__ = "Detector"
        detector.enabled = True
        
        context._hook_registry = {
            "layer_0": {
                HookType.FORWARD: [(detector, None)]
            }
        }
        
        result = collect_hooks_metadata(context)
        
        assert "layer_0" in result
        assert result["layer_0"][0]["layer_signature"] is None

    def test_collect_with_hook_type_string(self):
        """Test collecting metadata when hook_type is a string."""
        context = Mock(spec=LanguageModelContext)
        detector = Mock(spec=Detector)
        detector.id = "detector_1"
        detector.hook_type = "forward"  # String instead of enum
        detector.layer_signature = "layer_0"
        detector.__class__.__name__ = "Detector"
        detector.enabled = True
        
        context._hook_registry = {
            "layer_0": {
                HookType.FORWARD: [(detector, None)]
            }
        }
        
        result = collect_hooks_metadata(context)
        
        assert "layer_0" in result
        # Should handle string hook_type
        assert result["layer_0"][0]["hook_type"] == "forward"

