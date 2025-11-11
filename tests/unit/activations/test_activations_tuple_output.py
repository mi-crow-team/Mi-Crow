from typing import Sequence, Any

import torch
from torch import nn
from datasets import Dataset

from amber.core.language_model import LanguageModel
from amber.adapters.text_snippet_dataset import TextSnippetDataset
from amber.store.local_store import LocalStore


class FakeTokenizer:
    def __call__(self, texts: Sequence[str], **kwargs: Any):
        # simple tokenization: each char -> id, pad to max
        padding = kwargs.get("padding", False)
        return_tensors = kwargs.get("return_tensors", "pt")
        max_len = max(len(t) for t in texts) if padding else max(len(t) for t in texts)
        ids = []
        attn = []
        for t in texts:
            row = [ord(c) % 97 + 1 for c in t]
            pad = max_len - len(row)
            ids.append(row + [0] * pad)
            attn.append([1] * len(row) + [0] * pad)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn)}
        raise ValueError


class TupleOutLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(200, 4)
        self.lin = nn.Linear(4, 4)

    def forward(self, input_ids, attention_mask=None):
        x = self.emb(input_ids)
        x = self.lin(x)
        # Return a tuple where first item is tensor to trigger tuple path
        return (x, {"aux": 1})


def make_ds(texts, tmp_path):
    base = Dataset.from_dict({"text": texts})
    return TextSnippetDataset(base, cache_dir=tmp_path)


def test_tuple_output_handling(tmp_path):
    """Test that tuple outputs from model layers are handled correctly."""
    tok = FakeTokenizer()
    net = TupleOutLM()
    store = LocalStore(tmp_path/"store")
    lm = LanguageModel(model=net, tokenizer=tok, store=store)

    # pick the linear layer by name
    target_name = None
    for name, layer in lm.layers.name_to_layer.items():
        if isinstance(layer, nn.Linear):
            target_name = name
            break
    assert target_name is not None

    ds = make_ds(["ab", "cde", "f"], tmp_path/"cache")
    store = LocalStore(tmp_path/"store")

    lm.activations.infer_and_save(
        ds,
        layer_signature=target_name,
        run_name="tuple_run",
        store=store,
        batch_size=2,
        autocast=False,
    )

    # Validate payload exists
    batches = store.list_run_batches("tuple_run")
    assert batches == [0, 1]
    b0 = store.get_run_batch("tuple_run", 0)
    assert set(b0.keys()) >= {"activations", "input_ids", "attention_mask"}
