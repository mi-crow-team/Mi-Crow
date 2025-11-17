import pytest
import torch
from torch import nn
from pathlib import Path
import tempfile

from amber.core.language_model import LanguageModel
from amber.store.local_store import LocalStore


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Sequential(
                nn.Linear(4, 2),
                nn.Sigmoid(),
            ),
        )

    def forward(self, x):
        return self.feat(x)


@pytest.fixture()
def tiny_lm():
    model = TinyNet()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / "store")
    # tokenizer is unused by LanguageModel core behaviors we test here
    return LanguageModel(model=model, tokenizer=None, store=store)


def test_flatten_layer_names_and_indices_consistency(tiny_lm: LanguageModel):
    names = tiny_lm.layers.get_layer_names()
    # Ensure names captured and indices map same count
    assert isinstance(names, list)
    assert len(names) > 0
    assert len(names) == len(tiny_lm.layers.idx_to_layer)
    # Names should be unique and all keys present in name_to_layer
    assert len(set(names)) == len(names)
    assert set(names) == set(tiny_lm.layers.name_to_layer.keys())


def test_get_layer_by_name_and_index_and_errors(tiny_lm: LanguageModel):
    # Pick a valid name and its corresponding module
    names = tiny_lm.layers.get_layer_names()
    first_name = names[0]
    layer_by_name = tiny_lm.layers._get_layer_by_name(first_name)
    # Find its index by searching idx_to_layer
    idx = None
    for k, v in tiny_lm.layers.idx_to_layer.items():
        if v is layer_by_name:
            idx = k
            break
    assert idx is not None
    layer_by_idx = tiny_lm.layers._get_layer_by_index(idx)
    assert layer_by_idx is layer_by_name

    # Error cases
    with pytest.raises(ValueError):
        tiny_lm.layers._get_layer_by_name("nonexistent_layer_name")
    with pytest.raises(ValueError):
        tiny_lm.layers._get_layer_by_index(10_000)


def test_register_forward_and_pre_forward_hooks_are_called(tiny_lm: LanguageModel):
    calls = {"pre": 0, "fwd": 0}

    def pre_hook(module, inputs):
        calls["pre"] += 1

    def fwd_hook(module, inputs, output):
        calls["fwd"] += 1

    # Attach hooks to a middle layer (e.g., the first Linear)
    # Find any nn.Linear layer name
    linear_name = None
    for name, layer in tiny_lm.layers.name_to_layer.items():
        if isinstance(layer, nn.Linear):
            linear_name = name
            break
    assert linear_name is not None

    tiny_lm.layers.register_pre_forward_hook_for_layer(linear_name, pre_hook)
    tiny_lm.layers.register_forward_hook_for_layer(linear_name, fwd_hook)

    x = torch.randn(3, 4)
    _ = tiny_lm.model(x)

    # At least one call should have happened for the chosen layer
    assert calls["pre"] >= 1
    assert calls["fwd"] >= 1


def test_register_hooks_by_index_also_works(tiny_lm: LanguageModel):
    calls = {"fwd": 0}

    def fwd_hook(module, inputs, output):
        calls["fwd"] += 1

    # Choose some valid index (e.g., 0)
    any_index = next(iter(tiny_lm.layers.idx_to_layer.keys()))
    tiny_lm.layers.register_forward_hook_for_layer(any_index, fwd_hook)

    x = torch.randn(2, 4)
    _ = tiny_lm.model(x)

    assert calls["fwd"] >= 1


def test_lazy_rebuild_of_maps_on_access(tiny_lm: LanguageModel):
    # Clear maps to simulate state before flatten
    tiny_lm.layers.name_to_layer.clear()
    tiny_lm.layers.idx_to_layer.clear()
    # Access by name should trigger a rebuild
    # Find some valid name by first rebuilding and capturing it, then clear again.
    tiny_lm.layers._flatten_layer_names()
    some_name = next(iter(tiny_lm.layers.name_to_layer.keys()))
    # Clear and then call _get_layer_by_name
    tiny_lm.layers.name_to_layer.clear()
    tiny_lm.layers.idx_to_layer.clear()
    layer = tiny_lm.layers._get_layer_by_name(some_name)
    assert isinstance(layer, nn.Module)

    # Clear again and access by index to trigger second branch
    tiny_lm.layers.name_to_layer.clear()
    tiny_lm.layers.idx_to_layer.clear()
    # Capture a valid index
    tiny_lm.layers._flatten_layer_names()
    some_idx = next(iter(tiny_lm.layers.idx_to_layer.keys()))
    tiny_lm.layers.name_to_layer.clear()
    tiny_lm.layers.idx_to_layer.clear()
    layer2 = tiny_lm.layers._get_layer_by_index(some_idx)
    assert isinstance(layer2, nn.Module)


def test_register_pre_forward_hook_by_index(tiny_lm: LanguageModel):
    calls = {"pre": 0}

    def pre_hook(module, inputs):
        calls["pre"] += 1

    # pick any index
    any_idx = next(iter(tiny_lm.layers.idx_to_layer.keys()))
    tiny_lm.layers.register_pre_forward_hook_for_layer(any_idx, pre_hook)

    x = torch.randn(2, 4)
    _ = tiny_lm.model(x)

    assert calls["pre"] >= 1




def test_pre_forward_hook_with_real_model():
    """Test pre-forward hooks with a more realistic model setup."""
    from amber.core.language_model import LanguageModel
    
    class SmallLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(32, 4)
            self.lin = nn.Linear(4, 4)

        def forward(self, input_ids, attention_mask=None):
            x = self.emb(input_ids)
            return self.lin(x)

    class Tok:
        def __call__(self, texts, **kwargs):
            padding = kwargs.get("padding", False)
            rt = kwargs.get("return_tensors", "pt")
            ids = [[1 + i for i, _ in enumerate(t)] for t in texts]
            T = max(len(x) for x in ids)
            if padding:
                ids = [row + [0] * (T - len(row)) for row in ids]
            attn = [[1] * len(t) + [0] * (T - len(t)) for t in texts]
            if rt == "pt":
                return {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attn, dtype=torch.long),
                }
            raise ValueError

    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / "store")
    lm = LanguageModel(model=SmallLM(), tokenizer=Tok(), store=store)

    # choose the linear layer by index and then by name to exercise both paths
    names = list(lm.layers.name_to_layer.keys())
    lin_name = [n for n in names if n.endswith("lin") or ".lin" in n][0]
    lin_index = list(lm.layers.name_to_layer.keys()).index(lin_name)

    seen = {"calls": 0, "last_shape": None}

    def pre_hook(_module, inputs):
        seen["calls"] += 1
        if inputs and isinstance(inputs[0], torch.Tensor):
            seen["last_shape"] = tuple(inputs[0].shape)

    # by index
    h1 = lm.layers.register_pre_forward_hook_for_layer(lin_index, pre_hook)
    # by name
    h2 = lm.layers.register_pre_forward_hook_for_layer(lin_name, pre_hook)

    out, enc = lm.forwards(["ab", "c"], autocast=False)
    assert isinstance(out, torch.Tensor)
    # both hooks should have fired once
    assert seen["calls"] == 2
    assert seen["last_shape"][0] == enc["input_ids"].shape[0]

    # cleanup handles should not raise
    h1.remove()
    h2.remove()
