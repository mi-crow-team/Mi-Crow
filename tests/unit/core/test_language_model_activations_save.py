from typing import Any, Dict, Iterable, List, Tuple

import torch

from amber.core.language_model_activations import LanguageModelActivations


class _Param:
    def __init__(self):
        class _Dev:
            type = "cpu"

        self.device = _Dev()


class _FakeModel:
    def parameters(self) -> Iterable[_Param]:
        # Return something that yields at least one object with .device
        return iter([_Param()])


class _FakeLayers:
    def __init__(self, layer_names: List[str | int]):
        self._next = 0
        self.id_to_detector: Dict[str, Any] = {}
        self.layer_names = layer_names
        self.unregistered: List[str] = []

    def register_hook(self, layer_signature: str | int, detector: Any) -> str:
        self._next += 1
        hid = f"hid{self._next}"
        self.id_to_detector[hid] = detector
        return hid

    def unregister_hook(self, hook_id: str) -> None:
        self.unregistered.append(hook_id)
        self.id_to_detector.pop(hook_id, None)

    def get_layer_names(self) -> List[str | int]:
        return list(self.layer_names)


class _FakeLM:
    def __init__(self, layers: _FakeLayers):
        self.layers = layers

    def _inference(self, texts, *, tok_kwargs=None, autocast=True, autocast_dtype=None, with_controllers=True):
        B = len(texts)
        T = 3
        D = 4
        # simulate captured flattened activations with N = B*T
        for det in list(self.layers.id_to_detector.values()):
            setattr(det, "captured_activations", torch.ones(B * T, D))
        # Return (output, enc) tuple
        inp_ids = torch.arange(B * T).view(B, T)
        attn = torch.ones(B, T, dtype=torch.long)
        enc = {"input_ids": inp_ids, "attention_mask": attn}
        output = torch.ones(B, T, D)
        return output, enc


class _FakeStore:
    def __init__(self):
        self.meta: Dict[str, Dict[str, Any]] = {}
        self.batches: List[Tuple[str, int, Dict[str, Any]]] = []

    def put_run_meta(self, key: str, value: Dict[str, Any]):
        self.meta[key] = value

    def put_run_batch(self, key: str, idx: int, payload: Dict[str, Any]):
        self.batches.append((key, idx, payload))


class _FakeDataset:
    def __init__(self, batches: List[List[str]], cache_dir: str = "cache"):
        self._batches = batches
        self.cache_dir = cache_dir

    def iter_batches(self, batch_size: int):
        for b in self._batches:
            yield b

    def __len__(self):
        return sum(len(b) for b in self._batches)


class _FakeContext:
    def __init__(self, lm: _FakeLM, model: _FakeModel, store: _FakeStore):
        self.language_model = lm
        self.model = model
        self.store = store


def test_infer_and_save_happy_path_cpu_with_inputs_and_reshape():
    layers = _FakeLayers(["L0"])
    lm = _FakeLM(layers)
    model = _FakeModel()
    store = _FakeStore()
    ctx = _FakeContext(lm, model, store)
    acts = LanguageModelActivations(ctx)

    ds = _FakeDataset([["a", "b"], ["c", "d"]])

    acts.infer_and_save(
        ds,
        layer_signature="L0",
        run_name="run1",
        store=store,
        batch_size=2,
        dtype=torch.float32,
        autocast=False,
        save_inputs=True,
        verbose=False,
    )

    # Meta stored once
    assert "run1" in store.meta
    # Two batches saved
    assert len(store.batches) == 2
    for _, _, payload in store.batches:
        # Activations reshaped to [B, T, D] and on CPU
        act = payload["activations"]
        assert isinstance(act, torch.Tensor)
        assert tuple(act.shape) == (2, 3, 4)
        assert not act.is_cuda
        # Inputs saved
        assert "input_ids" in payload and "attention_mask" in payload

    # Hook unregistered
    assert len(layers.unregistered) == 1


def test_infer_and_save_all_layers_without_inputs_and_dtype_copy_path():
    layers = _FakeLayers(["A", "B"]) 
    lm = _FakeLM(layers)
    model = _FakeModel()
    store = _FakeStore()
    ctx = _FakeContext(lm, model, store)
    acts = LanguageModelActivations(ctx)

    ds = _FakeDataset([["x", "y", "z"]])

    acts.infer_and_save_all_layers(
        ds,
        layer_signatures=None,
        run_name="run_all",
        store=store,
        batch_size=3,
        dtype=torch.float16,  # exercise dtype branch
        autocast=False,
        save_inputs=False,
        verbose=False,
    )

    # Meta stored once
    assert "run_all" in store.meta
    # One batch saved
    assert len(store.batches) == 1
    (_, idx, payload) = store.batches[0]
    assert idx == 0
    # Should contain keys for each layer
    assert "activations_A" in payload and "activations_B" in payload
    for k, v in payload.items():
        if k.startswith("activations_"):
            assert isinstance(v, torch.Tensor)
            # With save_inputs=False, reshape is not attempted; expect flattened [B*T, D]
            assert tuple(v.shape) == (3 * 3, 4)
            assert not v.is_cuda


def test_infer_and_save_all_layers_with_inputs_and_verbose_logs_and_reshape():
    layers = _FakeLayers(["Lx", "Ly", "Lz"]) 
    lm = _FakeLM(layers)
    model = _FakeModel()
    store = _FakeStore()
    ctx = _FakeContext(lm, model, store)
    acts = LanguageModelActivations(ctx)

    ds = _FakeDataset([["p", "q"], ["r", "s"]])

    acts.infer_and_save_all_layers(
        ds,
        layer_signatures=None,
        run_name="run_all_verbose",
        store=store,
        batch_size=2,
        dtype=None,  # exercise dtype None path
        autocast=False,
        save_inputs=True,  # exercise reshape path
        verbose=True,  # exercise logging branches
    )

    # Meta stored once and batches saved twice
    assert "run_all_verbose" in store.meta
    assert len(store.batches) == 2
    for _, _, payload in store.batches:
        # Should contain keys for each layer and input tensors when save_inputs=True
        assert "input_ids" in payload and "attention_mask" in payload
        for name in ["Lx", "Ly", "Lz"]:
            key = f"activations_{name}"
            assert key in payload
            t = payload[key]
            assert isinstance(t, torch.Tensor)
            assert tuple(t.shape) == (2, 3, 4)

import torch
from torch import nn
from datasets import Dataset

from amber.core.language_model import LanguageModel
from amber.core.language_model_activations import LanguageModelActivations
from amber.adapters.text_snippet_dataset import TextSnippetDataset
from amber.store import LocalStore


class TinyLM(nn.Module):
    def __init__(self, d_model: int = 8, vocab_size: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, d_model)

        class Cfg:
            pad_token_id = None
            name_or_path = "TinyLM"
        self.config = Cfg()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        x = self.embed(input_ids)
        B, T, D = x.shape
        y = self.proj(x.view(B * T, D)).view(B, T, D)
        return y


class FakeTokenizer:
    def __init__(self):
        self.eos_token = "<eos>"
        self.eos_token_id = 0
        self.pad_token = None
        self.pad_token_id = None

    def __len__(self):
        return 256

    def __call__(self, texts, **kwargs):
        # simple fixed-length tokenization for determinism
        max_len = 4
        ids = []
        attn = []
        for t in texts:
            n = min(max_len, max(1, len(t) % (max_len + 1)))
            row = list(range(1, 1 + n))
            pad_id = self.pad_token_id or self.eos_token_id
            ids.append(row + [pad_id] * (max_len - n))
            attn.append([1] * n + [0] * (max_len - n))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn)}


def _layer_sig(lm: nn.Module) -> str:
    return f"{lm.__class__.__name__.lower()}_proj"


def test_infer_and_save_writes_batches_and_meta(tmp_path):
    # Build tiny dataset
    base = Dataset.from_dict({"text": ["a", "bb", "ccc", "dddd", "ee"]})
    ds = TextSnippetDataset(base, cache_dir=tmp_path)

    model = TinyLM()
    tok = FakeTokenizer()
    lm = LanguageModel(model=model, tokenizer=tok, store=LocalStore(tmp_path / "store"))

    lma = LanguageModelActivations(lm.context)
    run = "unittest_run"
    layer_sig = _layer_sig(model)

    lma.infer_and_save(
        ds,
        layer_signature=layer_sig,
        run_name=run,
        store=lm.store,
        batch_size=2,
        dtype=torch.float32,
        autocast=False,
        save_inputs=True,
        verbose=False,
    )

    # Expect 3 batches for 5 examples with batch_size=2
    batches = lm.store.list_run_batches(run)
    assert batches == [0, 1, 2]

    # Load a batch and check keys
    batch0 = lm.store.get_run_batch(run, 0)
    assert set(batch0.keys()) >= {"activations", "input_ids", "attention_mask"}
    acts = batch0["activations"]
    assert isinstance(acts, torch.Tensor)
    # Expect [B, T, D] with B=2, T=4, D=d_model=8
    assert acts.dim() == 3 and acts.shape[1] == 4

    # Metadata exists and contains run_name
    meta = lm.store.get_run_meta(run)
    assert meta.get("run_name") == run
