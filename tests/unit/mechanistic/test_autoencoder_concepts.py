import pytest
import torch

from amber.mechanistic.autoencoder.concepts.autoencoder_concepts import AutoencoderConcepts
from amber.mechanistic.autoencoder.concepts.concept_models import NeuronText
from amber.mechanistic.autoencoder.autoencoder_context import AutoencoderContext
from amber.mechanistic.autoencoder.autoencoder import Autoencoder


class _DummyHandle:
    def remove(self):
        # no-op
        pass


class _FakeLayers:
    def register_forward_hook_for_layer(self, layer_signature, hook, hook_args=None):
        # store for potential introspection if needed
        self._hook = hook
        return _DummyHandle()
    
    def register_hook(self, layer_signature, hook, hook_type=None):
        """Fake register_hook that returns a hook_id."""
        import uuid
        return hook.id or str(uuid.uuid4())
    
    def unregister_hook(self, hook_id):
        """Fake unregister_hook."""
        pass


class _FakeLM:
    def __init__(self):
        self._layers = _FakeLayers()
        self.enabled_text_tracking = False
        self._trackers: list = []

    @property
    def layers(self):
        return self._layers

    def enable_input_text_tracking(self):
        self.enabled_text_tracking = True

    def register_activation_text_tracker(self, tracker):
        self._trackers.append(tracker)

    def unregister_activation_text_tracker(self, tracker):
        if tracker in self._trackers:
            self._trackers.remove(tracker)


def test_get_top_texts_without_tracker_returns_empty():
    # Create autoencoder and context
    autoencoder = Autoencoder(n_latents=4, n_inputs=10)
    concepts = AutoencoderConcepts(autoencoder.context)
    assert concepts.get_top_texts_for_neuron(0) == []
    assert concepts.get_all_top_texts() == []


def test_enable_and_disable_text_tracking_positive_max_over_tokens():
    lm = _FakeLM()
    # Create autoencoder and context
    autoencoder = Autoencoder(n_latents=3, n_inputs=10)
    autoencoder.context.lm = lm
    autoencoder.context.lm_layer_signature = "sig"
    concepts = AutoencoderConcepts(autoencoder.context)

    # Enable tracking
    autoencoder.context.text_tracking_enabled = True
    autoencoder.context.text_tracking_k = 3
    autoencoder.context.text_tracking_negative = False
    concepts.enable_text_tracking()
    assert concepts.top_texts_tracker is not None

    tracker = concepts.top_texts_tracker
    assert tracker is not None

    # Provide texts for a batch of 2
    tracker.set_current_texts(["hello", "world"])

    # Create activations [B, T, D] = [2, 2, 3]
    # For positive tracking, we take max over tokens per neuron
    tens = torch.tensor([
        [[1.0, 0.2, -0.5], [0.5, 0.1, -0.2]],  # sample 0 -> max per D: [1.0, 0.2, -0.2]
        [[0.7, 0.4, 0.3], [0.9, 0.05, 0.6]],   # sample 1 -> max per D: [0.9, 0.4, 0.6]
    ])

    # Call the internal hook directly to avoid needing a real model
    tracker.process_activations(None, None, tens)

    # Top texts for neuron 0 should be sorted by descending score
    tops0 = concepts.get_top_texts_for_neuron(0)
    # The new behavior collects individual token activations, not just max per text
    # So we get: hello (1.0), world (0.9), world (0.7)
    assert len(tops0) == 3
    assert tops0[0].text == "hello"
    assert tops0[0].score == pytest.approx(1.0)
    assert tops0[1].text == "world" 
    assert tops0[1].score == pytest.approx(0.9)
    assert tops0[2].text == "world"
    assert tops0[2].score == pytest.approx(0.7)

    # Get all
    all_tops = concepts.get_all_top_texts()
    assert len(all_tops) == 3

    # Reset and ensure cleared
    concepts.reset_top_texts()
    # After reset with no new updates, get_all returns an empty list
    assert concepts.get_all_top_texts() == []

    # Disable tracking detaches and clears tracker
    concepts.disable_text_tracking()
    assert concepts.top_texts_tracker is None


def test_negative_tracking_uses_min_over_tokens_and_asc_sort():
    lm = _FakeLM()
    # Create autoencoder and context
    autoencoder = Autoencoder(n_latents=2, n_inputs=10)
    autoencoder.context.lm = lm
    autoencoder.context.lm_layer_signature = "sig"
    concepts = AutoencoderConcepts(autoencoder.context)

    autoencoder.context.text_tracking_enabled = True
    autoencoder.context.text_tracking_k = 2
    autoencoder.context.text_tracking_negative = True
    concepts.enable_text_tracking()
    tracker = concepts.top_texts_tracker
    assert tracker is not None

    tracker.set_current_texts(["a", "b"])
    tens = torch.tensor([
        [[1.0, -1.0], [2.0, -0.5]],  # sample 0 -> min per D: [1.0, -1.0]
        [[0.3, -0.7], [0.2, -0.9]],  # sample 1 -> min per D: [0.2, -0.9]
    ])
    tracker.process_activations(None, None, tens)

    # For negative mode, items should be sorted ascending by actual score
    tops0 = concepts.get_top_texts_for_neuron(1)  # neuron 1 collects -1.0 (a) and -0.9 (b)
    assert [t.text for t in tops0] == ["a", "b"]
    assert [pytest.approx(t.score) for t in tops0] == [pytest.approx(-1.0), pytest.approx(-0.9)]
