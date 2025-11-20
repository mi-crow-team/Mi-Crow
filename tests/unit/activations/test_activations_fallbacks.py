from typing import Sequence, Any
import tempfile
from pathlib import Path

import torch
from torch import nn
from datasets import Dataset
import tempfile
from pathlib import Path

from amber.language_model.language_model import LanguageModel
from amber.adapters.text_snippet_dataset import TextSnippetDataset
from amber.store.local_store import LocalStore
import tempfile
from pathlib import Path


class Tok:
    def __call__(self, texts: Sequence[str], **kwargs: Any):
        padding = kwargs.get("padding", False)
        rt = kwargs.get("return_tensors", "pt")
        lens = [len(t) or 1 for t in texts]
        T = max(lens) if padding else max(lens)
        ids = []
        attn = []
        for l in lens:
            ids.append(list(range(1, l + 1)) + [0] * (T - l))
            attn.append([1] * l + [0] * (T - l))
        if rt == "pt":
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }
        raise ValueError


class LastHiddenObj:
    def __init__(self, t: torch.Tensor):
        self.last_hidden_state = t


class ObjOutLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return LastHiddenObj(self.lin(x))


class ObjOutLM(nn.Module):
    def __init__(self, vocab_size: int = 32, d_model: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.obj = ObjOutLayer(d_model)

    def forward(self, input_ids, attention_mask=None):
        x = self.emb(input_ids)
        return self.obj(x)


def make_ds(texts, tmp_path):
    base = Dataset.from_dict({"text": texts})
    return TextSnippetDataset(base, dataset_dir=tmp_path)


def test_hook_fallback_reads_last_hidden_state_attr(tmp_path):
    """Test that hook fallback mechanism works for layers returning objects with last_hidden_state attribute."""
    tok = Tok()
    net = ObjOutLM()
    store = LocalStore(tmp_path / "store2")
    lm = LanguageModel(model=net, tokenizer=tok, store=store)

    # target the obj layer whose output is a custom object
    layer_name = None
    for name, layer in lm.layers.name_to_layer.items():
        if isinstance(layer, ObjOutLayer):
            layer_name = name
            break
    assert layer_name is not None

    ds = make_ds(["x", "yy"], tmp_path / "cache2")
    # Store is already set on lm from initialization

    lm.activations.save_activations_dataset(
        ds,
        layer_signature=layer_name,
        run_name="obj",
        batch_size=2,
        autocast=False,
    )

    b0 = store.get_run_batch("obj", 0)
    # Ensure activations were captured despite layer returning object
    assert "activations" in b0 and b0["activations"].ndim == 3
