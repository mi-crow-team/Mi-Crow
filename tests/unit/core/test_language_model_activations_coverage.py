"""Additional tests to improve coverage for language_model_activations.py."""
import pytest
import torch
from unittest.mock import Mock, MagicMock

from amber.core.language_model_activations import LanguageModelActivations


class _FakeLayers:
    def __init__(self, layer_names=None):
        self._next_id = 0
        self.id_to_detector = {}
        self.layer_names = layer_names or ["L0", "L1"]
        self.register_calls = []
        self.unregister_calls = []

    def register_hook(self, layer_signature, detector):
        self._next_id += 1
        hook_id = f"h{self._next_id}"
        self.id_to_detector[hook_id] = detector
        self.register_calls.append((layer_signature, detector))
        return hook_id

    def unregister_hook(self, hook_id):
        self.unregister_calls.append(hook_id)
        self.id_to_detector.pop(hook_id, None)

    def get_layer_names(self):
        return list(self.layer_names)


class _FakeLanguageModel:
    def __init__(self, layers):
        self.layers = layers

    def _inference(self, texts, **kwargs):
        # Simulate inference
        pass


class _FakeContext:
    def __init__(self, language_model):
        self.language_model = language_model
        self.model = None
        self.store = None


def test_capture_activations_raises_when_detector_returns_none():
    """Test that capture_activations raises RuntimeError when detector returns None (line 84)."""
    layers = _FakeLayers()
    lm = _FakeLanguageModel(layers)
    ctx = _FakeContext(lm)
    acts = LanguageModelActivations(ctx)
    
    # Create a detector that returns None
    class NoneDetector:
        def get_captured(self):
            return None
    
    # Mock _setup_detector to return our None detector
    original_setup = acts._setup_detector
    def mock_setup(layer_sig, hook_id):
        detector = NoneDetector()
        hook_id = layers.register_hook(layer_sig, detector)
        return detector, hook_id
    
    acts._setup_detector = mock_setup
    
    # Should raise RuntimeError
    with pytest.raises(RuntimeError, match="Failed to capture activations"):
        acts.capture_activations(["test"], layer_signature="L0")


def test_capture_activations_all_layers_with_none_layer_signatures():
    """Test capture_activations_all_layers when layer_signatures is None (line 112->113)."""
    layers = _FakeLayers(layer_names=["layer1", "layer2", "layer3"])
    lm = _FakeLanguageModel(layers)
    ctx = _FakeContext(lm)
    acts = LanguageModelActivations(ctx)
    
    # Mock detectors to return activations
    class MockDetector:
        def __init__(self):
            self.captured = torch.randn(2, 5)
        
        def get_captured(self):
            return self.captured
    
    original_setup = acts._setup_detector
    def mock_setup(layer_sig, hook_id):
        detector = MockDetector()
        hook_id = layers.register_hook(layer_sig, detector)
        return detector, hook_id
    
    acts._setup_detector = mock_setup
    
    # Call with None layer_signatures - should get all layers
    result = acts.capture_activations_all_layers(["test"], layer_signatures=None)
    
    # Should have entries for all layers
    assert set(result.keys()) == {"layer1", "layer2", "layer3"}
    for v in result.values():
        assert isinstance(v, torch.Tensor)


def test_capture_activations_all_layers_with_none_activations():
    """Test capture_activations_all_layers when some detectors return None (line 138->136)."""
    layers = _FakeLayers(layer_names=["layer1", "layer2"])
    lm = _FakeLanguageModel(layers)
    ctx = _FakeContext(lm)
    acts = LanguageModelActivations(ctx)
    
    # Mock detectors - one returns None, one returns tensor
    class MixedDetector:
        def __init__(self, return_none=False):
            self.return_none = return_none
            self.captured = torch.randn(2, 5) if not return_none else None
        
        def get_captured(self):
            return self.captured
    
    detector_count = [0]
    original_setup = acts._setup_detector
    def mock_setup(layer_sig, hook_id):
        detector = MixedDetector(return_none=(detector_count[0] == 0))
        detector_count[0] += 1
        hook_id = layers.register_hook(layer_sig, detector)
        return detector, hook_id
    
    acts._setup_detector = mock_setup
    
    # Call - should only include layers with non-None activations
    result = acts.capture_activations_all_layers(["test"], layer_signatures=["layer1", "layer2"])
    
    # Should only have entry for layer2 (layer1 returned None)
    assert "layer2" in result
    assert "layer1" not in result  # None activations are skipped

