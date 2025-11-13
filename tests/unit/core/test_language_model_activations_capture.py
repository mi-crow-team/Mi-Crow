from typing import Dict, Tuple, Any

import torch

from amber.core.language_model_activations import LanguageModelActivations


class _FakeLayers:
    def __init__(self, layer_names: list[str] | None = None, raise_on_unregister: bool = False):
        self._next_id = 0
        self.id_to_detector: Dict[str, Any] = {}
        self.register_calls: list[Tuple[str | int, Any]] = []
        self.unregister_calls: list[str] = []
        self.layer_names = layer_names or ["L0", "L1"]
        self.raise_on_unregister = raise_on_unregister

    def register_hook(self, layer_signature: str | int, detector: Any, hook_type: Any = None) -> str:
        self._next_id += 1
        hook_id = f"h{self._next_id}"
        self.id_to_detector[hook_id] = detector
        self.register_calls.append((layer_signature, detector))
        return hook_id

    def unregister_hook(self, hook_id: str) -> None:
        self.unregister_calls.append(hook_id)
        if self.raise_on_unregister:
            raise RuntimeError("cannot unregister")
        self.id_to_detector.pop(hook_id, None)

    def get_layer_names(self) -> list[str | int]:
        return list(self.layer_names)


class _FakeLanguageModel:
    def __init__(self, layers: _FakeLayers):
        self.layers = layers

    def _inference(self, texts, *, tok_kwargs=None, autocast=True, autocast_dtype=None, with_controllers=True):
        # Simulate that each registered detector captured some tensor
        for detector in list(self.layers.id_to_detector.values()):
            # LayerActivationDetector stores activations in _tensor_metadata['activations'] (one tensor per batch)
            if not hasattr(detector, '_tensor_metadata'):
                detector._tensor_metadata = {}
            if not hasattr(detector, '_tensor_batches'):
                detector._tensor_batches = {}
            tensor = torch.ones(2, 3)
            detector._tensor_metadata['activations'] = tensor
            if 'activations' not in detector._tensor_batches:
                detector._tensor_batches['activations'] = []
            detector._tensor_batches['activations'].append(tensor)
        # Return (output, enc) tuple
        return torch.ones(2, 3), {"input_ids": torch.ones(2, 3), "attention_mask": torch.ones(2, 3)}


class _FakeContext:
    def __init__(self, language_model: _FakeLanguageModel):
        self.language_model = language_model
        self.model = None
        self.store = None


def test_capture_activations_returns_tensor_and_cleans_up():
    layers = _FakeLayers()
    lm = _FakeLanguageModel(layers)
    ctx = _FakeContext(lm)
    acts = LanguageModelActivations(ctx)

    out = acts.capture_activations(["hello"], layer_signature="L0", autocast=False)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 3)
    # Hook must be unregistered
    assert len(layers.unregister_calls) == 1


def test_capture_activations_all_layers_collects_for_each_layer_and_cleans_up():
    layers = _FakeLayers(layer_names=["A", "B", 2])
    lm = _FakeLanguageModel(layers)
    ctx = _FakeContext(lm)
    acts = LanguageModelActivations(ctx)

    out = acts.capture_activations_all_layers(["x"], layer_signatures=None, autocast=False)
    # Should have entries for each layer name
    assert set(out.keys()) == {"A", "B", 2}
    for v in out.values():
        assert isinstance(v, torch.Tensor)
        assert v.shape == (2, 3)
    # All hooks must be unregistered
    assert len(layers.unregister_calls) == 3


def test_cleanup_detector_swallows_errors():
    layers = _FakeLayers(raise_on_unregister=True)
    lm = _FakeLanguageModel(layers)
    ctx = _FakeContext(lm)
    acts = LanguageModelActivations(ctx)
    # Register one and force unregister to raise; should not propagate
    _, hook_id = acts._setup_detector("L0", "sfx")
    acts._cleanup_detector(hook_id)
    # Even though error raised internally, test should reach here
    assert layers.unregister_calls == [hook_id]


