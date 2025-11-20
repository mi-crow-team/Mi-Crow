from typing import Any, Dict, Iterable, List, Tuple

import torch


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

    def register_hook(self, layer_signature: str | int, detector: Any, hook_type: Any = None) -> str:
        self._next += 1
        hid = f"hid{self._next}"
        self.id_to_detector[hid] = detector
        return hid

    def unregister_hook(self, hook_id: str) -> None:
        self.unregistered.append(hook_id)
        self.id_to_detector.pop(hook_id, None)

    def get_layer_names(self) -> List[str | int]:
        return list(self.layer_names)
    
    def get_detectors(self) -> List[Any]:
        return list(self.id_to_detector.values())


class _FakeLM:
    def __init__(self, layers: _FakeLayers):
        self.layers = layers
        self.store = None

    def _get_device(self):
        """Get device for fake LM."""
        return torch.device("cpu")

    def _inference(self, texts, *, tok_kwargs=None, autocast=True, autocast_dtype=None, with_controllers=True):
        B = len(texts)
        T = 3
        D = 4
        # simulate captured flattened activations with N = B*T
        for det in list(self.layers.id_to_detector.values()):
            # LayerActivationDetector stores activations in _tensor_metadata['activations'] (one tensor per batch)
            if not hasattr(det, '_tensor_metadata'):
                det.tensor_metadata = {}
            if not hasattr(det, '_tensor_batches'):
                det.tensor_batches = {}
            tensor = torch.ones(B * T, D)
            det.tensor_metadata['activations'] = tensor
            if 'activations' not in det.tensor_batches:
                det.tensor_batches['activations'] = []
            det.tensor_batches['activations'].append(tensor)
        # Return (output, enc) tuple
        inp_ids = torch.arange(B * T).view(B, T)
        attn = torch.ones(B, T, dtype=torch.long)
        enc = {"input_ids": inp_ids, "attention_mask": attn}
        output = torch.ones(B, T, D)
        return output, enc
    
    def save_detector_metadata(self, run_name: str, batch_index: int) -> str:
        if self.store is None:
            raise ValueError("Store must be provided or set on the language model")
        detectors_metadata, detectors_tensor_metadata = self.get_all_detector_metadata()
        return self.store.put_detector_metadata(run_name, batch_index, detectors_metadata, detectors_tensor_metadata)
    
    def get_all_detector_metadata(self):
        detectors = self.layers.get_detectors()
        detectors_metadata = {}
        detectors_tensor_metadata = {}
        for detector in detectors:
            layer_sig = getattr(detector, 'layer_signature', 'unknown')
            detectors_metadata[layer_sig] = getattr(detector, 'metadata', {})
            detectors_tensor_metadata[layer_sig] = getattr(detector, 'tensor_metadata', {})
        return detectors_metadata, detectors_tensor_metadata


class _FakeStore:
    def __init__(self):
        self.meta: Dict[str, Dict[str, Any]] = {}
        self.batches: List[Tuple[str, int, Dict[str, Any]]] = []

    def put_run_meta(self, key: str, value: Dict[str, Any]):
        self.meta[key] = value

    def put_run_batch(self, key: str, idx: int, payload: Dict[str, Any]):
        self.batches.append((key, idx, payload))
    
    def put_detector_metadata(self, run_id: str, batch_index: int, metadata: Dict[str, Any], tensor_metadata: Dict[str, Dict[str, torch.Tensor]]):
        # Extract activations from tensor_metadata and save as batch
        for layer_sig, tensors in tensor_metadata.items():
            if "activations" in tensors:
                payload = {"activations": tensors["activations"]}
                self.batches.append((run_id, batch_index, payload))
        return f"runs/{run_id}/batch_{batch_index}"


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
        self.device = torch.device("cpu")


def test_save_activations_dataset_happy_path_cpu_with_inputs_and_reshape():
    layers = _FakeLayers(["L0"])
    model = _FakeModel()
    store = _FakeStore()
    lm = _FakeLM(layers)
    lm.store = store
    ctx = _FakeContext(lm, model, store)
    acts = LanguageModelActivations(ctx)

    ds = _FakeDataset([["a", "b"], ["c", "d"]])

    acts.save_activations_dataset(
        ds,
        layer_signature="L0",
        run_name="run1",
        batch_size=2,
        dtype=torch.float32,
        autocast=False,
        verbose=False,
    )

    # Meta stored once
    assert "run1" in store.meta
    # Two batches saved
    assert len(store.batches) == 2
    for _, _, payload in store.batches:
        # Activations saved as captured (flattened [B*T, D])
        act = payload["activations"]
        assert isinstance(act, torch.Tensor)
        # Shape is [B*T, D] = [6, 4] for B=2, T=3, D=4
        assert tuple(act.shape) == (6, 4)
        assert not act.is_cuda

    # Hook unregistered
    assert len(layers.unregistered) == 1


import torch
from torch import nn
from datasets import Dataset

<<<<<<< Updated upstream
from amber.core.language_model_activations import LanguageModelActivations
=======
from amber.language_model.language_model import LanguageModel
from amber.language_model.activations import LanguageModelActivations
>>>>>>> Stashed changes
from amber.adapters.text_snippet_dataset import TextSnippetDataset
from amber.store.local_store import LocalStore


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


def test_save_activations_dataset_writes_batches_and_meta(tmp_path):
    # Build tiny dataset
    base = Dataset.from_dict({"text": ["a", "bb", "ccc", "dddd", "ee"]})
    ds = TextSnippetDataset(base, cache_dir=tmp_path)

    model = TinyLM()
    tok = FakeTokenizer()
    lm = LanguageModel(model=model, tokenizer=tok, store=LocalStore(tmp_path / "store"))

    lma = LanguageModelActivations(lm.context)
    run = "unittest_run"
    layer_sig = _layer_sig(model)

    lma.save_activations_dataset(
        ds,
        layer_signature=layer_sig,
        run_name=run,
        batch_size=2,
        dtype=torch.float32,
        autocast=False,
        verbose=False,
    )

    # Expect 3 batches for 5 examples with batch_size=2
    batches = lm.store.list_run_batches(run)
    assert batches == [0, 1, 2]

    # Load a batch and check keys
    batch0 = lm.store.get_run_batch(run, 0)
    assert "activations" in batch0
    acts = batch0["activations"]
    assert isinstance(acts, torch.Tensor)
    # Activations should be saved (shape depends on layer output)
    assert acts.dim() >= 2

    # Metadata exists and contains run_name
    meta = lm.store.get_run_meta(run)
    assert meta.get("run_name") == run
