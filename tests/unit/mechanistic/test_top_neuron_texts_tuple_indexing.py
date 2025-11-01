import pytest
import torch
from torch import nn

from amber.mechanistic.autoencoder.concepts.top_neuron_texts import TopNeuronTexts
from amber.mechanistic.autoencoder.concepts.concept_models import NeuronText
from amber.mechanistic.autoencoder.autoencoder import Autoencoder


def create_top_neuron_texts(lm, layer_signature, k=5, nth_tensor=1):
    """Helper to create TopNeuronTexts with proper context."""
    sae = Autoencoder(n_latents=8, n_inputs=8)
    context = sae.context
    context.lm = lm
    context.lm_layer_signature = layer_signature
    context.text_tracking_k = k
    return TopNeuronTexts(context, k=k, nth_tensor=nth_tensor)


class TinyLM(nn.Module):
    def __init__(self, d: int = 4):
        super().__init__()
        self.emb = nn.Embedding(64, d)
        self.lin = nn.Linear(d, d)

    def forward(self, input_ids, attention_mask=None):
        x = self.emb(input_ids)
        y = self.lin(x)
        # Return a tuple to exercise nth_tensor indexing
        return (y, y.mean(dim=-1))


class FakeLayers:
    def __init__(self, model):
        self.model = model
        self._hook = None

    def register_forward_hook_for_layer(self, layer_signature, hook, hook_args=None):
        # store and return dummy handle
        self._hook = hook
        class H:
            def remove(self_inner):
                pass
        return H()
    
    def register_hook(self, layer_signature, hook, hook_type=None):
        """Fake register_hook that returns a hook_id."""
        import uuid
        return hook.id or str(uuid.uuid4())
    
    def unregister_hook(self, hook_id):
        """Fake unregister_hook."""
        pass


class FakeLMWrapper:
    def __init__(self, model: nn.Module):
        self.model = model
        self.layers = FakeLayers(model)
        self._trackers = []

    def register_activation_text_tracker(self, tracker):
        self._trackers.append(tracker)

    def unregister_activation_text_tracker(self, tracker):
        if tracker in self._trackers:
            self._trackers.remove(tracker)


def test_nth_tensor_out_of_range_raises():
    lm = FakeLMWrapper(TinyLM())
    # ask for 2nd index (third item) but model returns 2 items -> should raise in hook
    tnt = create_top_neuron_texts(lm=lm, layer_signature="sig", k=2, nth_tensor=2)
    tnt.set_current_texts(["a"])  # set texts to avoid assert

    with pytest.raises(ValueError):
        # Call hook directly with a 2-item tuple
        tnt.process_activations(None, None, (torch.randn(1, 2, 4), torch.randn(1, 2)))


def test_reduce_with_bd_tensor_is_noop_and_getters_bounds():
    lm = FakeLMWrapper(TinyLM())
    tnt = create_top_neuron_texts(lm=lm, layer_signature="sig", k=2)
    tnt.set_current_texts(["txt1"])  # B=1

    # Provide [B, D] tensor (no token dimension) and ensure it is accepted
    bd = torch.randn(1, 3)
    tnt.process_activations(None, None, bd)

    # For [B, D], reduction is a no-op, so scores are taken directly and heaps get one entry per neuron
    # get_top_texts out of bounds returns empty
    assert tnt.get_top_texts(100) == []
    all_lists = tnt.get_all()
    assert isinstance(all_lists, list) and len(all_lists) == bd.shape[1]
    # Each neuron should have one NeuronText with the provided text
    assert all(len(lst) == 1 and lst[0].text == "txt1" for lst in all_lists)
