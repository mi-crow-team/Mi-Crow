"""Additional tests to improve coverage for activations.py."""
import pytest
import torch
from unittest.mock import Mock, MagicMock

<<<<<<< Updated upstream
from amber.core.language_model_activations import LanguageModelActivations
=======
from amber.language_model.activations import LanguageModelActivations
>>>>>>> Stashed changes


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


