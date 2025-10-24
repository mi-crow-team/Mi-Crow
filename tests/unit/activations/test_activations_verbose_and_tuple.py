import logging
from typing import Sequence, Any

import torch
from torch import nn
from datasets import Dataset

from amber.core.language_model import LanguageModel
from amber.adapters.text_snippet_dataset import TextSnippetDataset
from amber.store import LocalStore


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


def test_save_model_activations_verbose_and_tuple_output(tmp_path, caplog):
    tok = FakeTokenizer()
    net = TupleOutLM()
    lm = LanguageModel(model=net, tokenizer=tok)

    # pick the linear layer by name
    target_name = None
    for name, layer in lm.layers.name_to_layer.items():
        if isinstance(layer, nn.Linear):
            target_name = name
            break
    assert target_name is not None

    ds = make_ds(["ab", "cde", "f"], tmp_path/"cache")
    store = LocalStore(tmp_path/"store")

    with caplog.at_level(logging.INFO):
        lm.activations.infer_and_save(
            ds,
            layer_signature=target_name,
            run_name="vrun",
            store=store,
            batch_size=2,
            autocast=False,
            verbose=True,
            free_cuda_cache_every=1,  # on CPU branch it won't empty, but still hit condition checks
        )
    # Ensure some verbose logs present
    assert any("Starting save_model_activations" in rec.message for rec in caplog.records)
    assert any("Saved batch" in rec.message for rec in caplog.records)
    assert any("Completed save_model_activations" in rec.message for rec in caplog.records)

    # Validate payload exists
    batches = store.list_run_batches("vrun")
    assert batches == [0, 1]
    b0 = store.get_run_batch("vrun", 0)
    assert set(b0.keys()) >= {"activations", "input_ids", "attention_mask"}
