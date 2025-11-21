from unittest.mock import MagicMock

import pytest
import torch

from amber.hooks.hook import HookType
from amber.mechanistic.sae.modules.topk_sae import TopKSae


class StubEngine:
    def __init__(self):
        self.loaded = None

    def encode(self, x):
        return x + 1, x * 0.5

    def decode(self, z):
        return z + 2

    def forward(self, x):
        return x - 1, x * 0.5, x + 3

    def state_dict(self):
        return {"weight": torch.ones(1)}

    def load_state_dict(self, state):
        self.loaded = state

    def parameters(self):
        return [torch.nn.Parameter(torch.ones(1, requires_grad=True))]

    def to(self, *args, **kwargs):
        return self


@pytest.fixture(autouse=True)
def stub_engine(monkeypatch):
    monkeypatch.setattr(
        TopKSae,
        "_initialize_sae_engine",
        lambda self: StubEngine(),
    )


def make_topk():
    return TopKSae(n_latents=2, n_inputs=2, k=1)


def test_encode_decode_forward():
    sae = make_topk()
    x = torch.ones(2, 2)
    assert torch.equal(sae.encode(x), x * 0.5)
    assert torch.equal(sae.decode(torch.zeros(2, 2)), torch.full((2, 2), 2.0))
    assert torch.equal(sae.forward(x), x + 3)


def test_modify_activations_forward_updates_texts():
    sae = make_topk()
    sae.hook_type = HookType.FORWARD
    sae._text_tracking_enabled = True
    tracker = MagicMock()
    tracker.get_current_texts.return_value = ["text"]
    lm = MagicMock()
    lm.get_input_tracker.return_value = tracker
    sae.context.lm = lm
    spy = MagicMock()
    sae.concepts.update_top_texts_from_latents = spy

    output = torch.ones(1, 2, 2)
    result = sae.modify_activations(module=None, inputs=(output,), output=output)

    assert result.shape == output.shape
    spy.assert_called_once()


def test_modify_activations_pre_forward_tuple():
    sae = make_topk()
    sae.hook_type = HookType.PRE_FORWARD

    input_tensor = torch.ones(1, 2)
    inputs = (input_tensor, torch.zeros(1, 2))
    result = sae.modify_activations(module=None, inputs=inputs, output=None)

    assert isinstance(result, tuple)
    assert torch.equal(result[0], torch.full((1, 2), 2.5))


def test_save_and_load(monkeypatch, tmp_path):
    captured = {}

    def fake_save(payload, path):
        captured["payload"] = payload

    monkeypatch.setattr("amber.mechanistic.sae.modules.topk_sae.torch.save", fake_save)
    sae = make_topk()
    sae.concepts.multiplication.data = torch.full((2,), 2.0)
    sae.save("model", path=tmp_path)
    assert "payload" in captured

    load_payload = {
        "sae_state_dict": {"weight": torch.ones(1)},
        "amber_metadata": {
            "n_latents": 2,
            "n_inputs": 2,
            "k": 1,
            "device": "cpu",
            "concepts_state": {
                "multiplication": torch.full((2,), 3.0),
                "bias": torch.full((2,), 4.0),
            },
        },
    }
    monkeypatch.setattr("amber.mechanistic.sae.modules.topk_sae.torch.load", lambda *args, **kwargs: load_payload)
    loaded = TopKSae.load(tmp_path / "model.pt")
    assert torch.equal(loaded.concepts.multiplication, torch.full((2,), 3.0))


def test_load_missing_metadata_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("amber.mechanistic.sae.modules.topk_sae.torch.load", lambda *args, **kwargs: {})
    with pytest.raises(ValueError):
        TopKSae.load(tmp_path / "missing.pt")


def test_modify_activations_text_tracking(monkeypatch):
    sae = make_topk()
    sae._text_tracking_enabled = True

    tracker = MagicMock()
    tracker.get_current_texts.return_value = ["alpha"]

    class DummyLM:
        def __init__(self):
            self.tokenizer = None

        def get_input_tracker(self):
            return tracker

    sae.context.lm = DummyLM()
    spy = MagicMock()
    sae.concepts.update_top_texts_from_latents = spy

    output = torch.ones(1, 2, 2)
    sae.modify_activations(module=None, inputs=(output,), output=output)
    spy.assert_called_once()


def test_modify_activations_object_output():
    class Wrapper:
        def __init__(self, tensor):
            self.last_hidden_state = tensor

    sae = make_topk()
    wrapper = Wrapper(torch.ones(1, 2))
    result = sae.modify_activations(module=None, inputs=(wrapper.last_hidden_state,), output=wrapper)
    assert isinstance(result, Wrapper)
    assert not torch.equal(result.last_hidden_state, torch.ones(1, 2))


def test_modify_activations_list_without_tensor_returns_original():
    sae = make_topk()
    output = ["a", "b"]
    result = sae.modify_activations(module=None, inputs=(torch.ones(1, 2),), output=output)
    assert result == output


def test_save_and_load_roundtrip(tmp_path):
    sae = make_topk()
    sae.concepts.multiplication.data = torch.full((2,), 2.5)
    sae.save("demo", tmp_path)

    loaded = TopKSae.load(tmp_path / "demo.pt")
    assert torch.allclose(loaded.concepts.multiplication, torch.full((2,), 2.5))

