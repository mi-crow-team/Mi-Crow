import pytest
import torch

from amber.mechanistic.autoencoder.concepts.autoencoder_concepts import AutoencoderConcepts
from amber.mechanistic.autoencoder.concepts.concept_models import NeuronText


class _DummyHandle:
    def remove(self):
        # no-op
        pass


class _FakeLayers:
    def register_forward_hook_for_layer(self, layer_signature, hook, hook_args=None):
        # store for potential introspection if needed
        self._hook = hook
        return _DummyHandle()


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
    concepts = AutoencoderConcepts(n_size=4)
    assert concepts.get_top_texts_for_neuron(0) == []
    assert concepts.get_all_top_texts() == []


def test_enable_and_disable_text_tracking_positive_max_over_tokens():
    lm = _FakeLM()
    concepts = AutoencoderConcepts(n_size=3, lm=lm, lm_layer_signature="sig")

    # Enable tracking
    concepts.enable_text_tracking(k=3, negative=False)
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
    tracker._activations_hook(None, None, tens)

    # Top texts for neuron 0 should be sorted by descending score
    tops0 = concepts.get_top_texts_for_neuron(0)
    # Scores for neuron 0 are 1.0 (hello) and 0.9 (world); positive mode sorts descending
    assert [t.text for t in tops0] == ["hello", "world"]
    assert isinstance(tops0[0], NeuronText)
    assert tops0[0].score == pytest.approx(1.0)

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
    concepts = AutoencoderConcepts(n_size=2, lm=lm, lm_layer_signature="sig")

    concepts.enable_text_tracking(k=2, negative=True)
    tracker = concepts.top_texts_tracker
    assert tracker is not None

    tracker.set_current_texts(["a", "b"])
    tens = torch.tensor([
        [[1.0, -1.0], [2.0, -0.5]],  # sample 0 -> min per D: [1.0, -1.0]
        [[0.3, -0.7], [0.2, -0.9]],  # sample 1 -> min per D: [0.2, -0.9]
    ])
    tracker._activations_hook(None, None, tens)

    # For negative mode, items should be sorted ascending by actual score
    tops0 = concepts.get_top_texts_for_neuron(1)  # neuron 1 collects -1.0 (a) and -0.9 (b)
    assert [t.text for t in tops0] == ["a", "b"]
    assert [pytest.approx(t.score) for t in tops0] == [pytest.approx(-1.0), pytest.approx(-0.9)]
