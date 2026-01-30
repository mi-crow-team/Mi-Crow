
"""Tests for hook metadata collection."""



import pytest


from unittest.mock import Mock, MagicMock



from mi_crow.language_model.hook_metadata import collect_hooks_metadata


from mi_crow.language_model.context import LanguageModelContext


from mi_crow.hooks.detector import Detector


from mi_crow.hooks.controller import Controller


from mi_crow.hooks import HookType


def _mock_hook(base_class: type, class_name: str) -> Mock:
    """Create a mock hook with a specific class name without mutating base classes."""
    hook_class = type(class_name, (base_class,), {})
    return Mock(spec=hook_class)




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


        detector = _mock_hook(Detector, "LayerActivationDetector")


        detector.id = "detector_1"


        detector.hook_type = HookType.FORWARD


        detector.layer_signature = "layer_0"


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


        controller = _mock_hook(Controller, "FunctionController")


        controller.id = "controller_1"


        controller.hook_type = HookType.PRE_FORWARD


        controller.layer_signature = "layer_1"


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


        detector1 = _mock_hook(Detector, "Detector1")


        detector1.id = "detector_1"


        detector1.hook_type = HookType.FORWARD


        detector1.layer_signature = "layer_0"


        detector1.enabled = True



        detector2 = _mock_hook(Detector, "Detector2")


        detector2.id = "detector_2"


        detector2.hook_type = HookType.FORWARD


        detector2.layer_signature = "layer_0"


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


        detector1 = _mock_hook(Detector, "Detector")


        detector1.id = "detector_1"


        detector1.hook_type = HookType.FORWARD


        detector1.layer_signature = "layer_0"


        detector1.enabled = True



        controller1 = _mock_hook(Controller, "Controller")


        controller1.id = "controller_1"


        controller1.hook_type = HookType.PRE_FORWARD


        controller1.layer_signature = "layer_1"


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


        detector = _mock_hook(Detector, "Detector")


        detector.id = "detector_1"


        detector.hook_type = HookType.FORWARD


        detector.layer_signature = 0


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


        detector = _mock_hook(Detector, "Detector")


        detector.id = "detector_1"


        detector.hook_type = HookType.FORWARD


        detector.layer_signature = None


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


        detector = _mock_hook(Detector, "Detector")


        detector.id = "detector_1"


        detector.hook_type = "forward"


        detector.layer_signature = "layer_0"


        detector.enabled = True



        context._hook_registry = {
            "layer_0": {
                HookType.FORWARD: [(detector, None)]
            }
        }



        result = collect_hooks_metadata(context)



        assert "layer_0" in result



        assert result["layer_0"][0]["hook_type"] == "forward"



