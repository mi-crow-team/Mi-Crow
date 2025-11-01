import torch
from torch import nn

from amber.mechanistic.autoencoder.concepts.top_neuron_texts import TopNeuronTexts
from amber.mechanistic.autoencoder.autoencoder import Autoencoder


class FakeLayers:
    def register_forward_hook_for_layer(self, layer_signature, hook, hook_args=None):
        # Keep reference to simulate handle
        self.hook = hook
        class H:
            def remove(self_inner):
                # no-op
                pass
        return H()
    
    def register_hook(self, layer_signature, hook, hook_type=None):
        """Fake register_hook that returns a hook_id."""
        import uuid
        return hook.id or str(uuid.uuid4())
    
    def unregister_hook(self, hook_id):
        """Fake unregister_hook."""
        pass


class FakeLM:
    def __init__(self):
        self.layers = FakeLayers()
        self._trackers = []

    def register_activation_text_tracker(self, tracker):
        self._trackers.append(tracker)

    def unregister_activation_text_tracker(self, tracker):
        if tracker in self._trackers:
            self._trackers.remove(tracker)


class Obj:
    def __init__(self, t: torch.Tensor):
        self.last_hidden_state = t


def test_hook_accepts_object_with_last_hidden_state_and_detach_is_safe():
    lm = FakeLM()
    # Create context for TopNeuronTexts
    sae = Autoencoder(n_latents=8, n_inputs=8)
    context = sae.context
    context.lm = lm
    context.lm_layer_signature = "sig"
    context.text_tracking_k = 1
    tnt = TopNeuronTexts(context, k=1)
    tnt.set_current_texts(["hello"])  # set texts for update

    tens = torch.randn(1, 2, 3)
    # Call hook with object exposing last_hidden_state to cover that branch
    tnt.process_activations(None, None, Obj(tens))

    # Heaps should have entries now
    all_lists = tnt.get_all()
    assert len(all_lists) == tens.shape[-1]

    # Detach should be safe and remove tracker from LM
    tnt.detach()
    assert tnt not in lm._trackers
