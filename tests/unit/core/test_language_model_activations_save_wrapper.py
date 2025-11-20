from types import SimpleNamespace

import torch

<<<<<<< Updated upstream
from amber.core.language_model_activations import LanguageModelActivations
=======
from amber.language_model.activations import LanguageModelActivations
>>>>>>> Stashed changes


class _DummyLayers:
    def register_forward_hook_for_layer(self, layer_signature, fn):
        # Keep the hook and return a handle with remove()
        self._hook = fn

        class _H:
            def remove(self_inner):
                pass

        return _H()


class _DummyLM:
    def __init__(self):
        # Minimal attributes used by LanguageModelActivations
        self.model = SimpleNamespace()
        # Layers object that can register hook
        self.layers = _DummyLayers()
        # Tokenizer interface used; we won't actually call infer
        self.lm_tokenizer = SimpleNamespace(tokenize=lambda texts, **kw: {"input_ids": torch.zeros((len(texts), 1), dtype=torch.long)})
        # Store is unused in this wrapper test
        self.store = None

    def __call__(self, **enc):  # called during infer; noop
        # Trigger hook as if the layer produced a tensor tuple
        if hasattr(self.layers, "_hook"):
            out = (torch.randn(1, 1, 4),)
            self.layers._hook(None, None, out)
        return SimpleNamespace()


def test_language_model_activations_save_calls_infer(monkeypatch):
    dummy = _DummyLM()
    act = LanguageModelActivations(dummy)  # type: ignore[arg-type]

    called = {}

    def fake_save_activations_dataset(dataset, layer_signature, **kwargs):
        called["ok"] = (dataset, layer_signature, kwargs)
        return "test_run"

    monkeypatch.setattr(LanguageModelActivations, "save_activations_dataset", staticmethod(fake_save_activations_dataset))

    # Call save wrapper and ensure delegation works
    ret = act.save_activations_dataset(dataset=SimpleNamespace(iter_batches=lambda bs: [["hi"]]), layer_signature="x")
    assert ret == "test_run"
    assert "ok" in called
