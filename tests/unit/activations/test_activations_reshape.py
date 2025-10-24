from typing import Sequence, Any

import torch
from torch import nn
from datasets import Dataset

from amber.core.language_model import LanguageModel
from amber.adapters.text_snippet_dataset import TextSnippetDataset
from amber.store import LocalStore


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


class FlattenThenReshapeLM(nn.Module):
    """
    Linear layer produces [B*T, D]. The model reshapes back to [B, T, D],
    but our activation hook will be attached to the Linear so it captures 2D.
    """

    def __init__(self, vocab_size: int = 64, d_model: int = 6):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.flat_proj = nn.Linear(d_model, d_model)

    def forward(self, input_ids, attention_mask=None):
        x = self.emb(input_ids)  # [B, T, D]
        B, T, D = x.shape
        y = self.flat_proj(x.view(B * T, D))  # [B*T, D]
        return y.view(B, T, D)


def make_ds(texts, tmp_path):
    base = Dataset.from_dict({"text": texts})
    return TextSnippetDataset(base, cache_dir=tmp_path)


def test_captured_2d_activations_are_reshaped_to_3d(tmp_path):
    """Test that 2D activations captured from flattened layers are properly reshaped to 3D."""
    tok = Tok()
    net = FlattenThenReshapeLM()
    lm = LanguageModel(model=net, tokenizer=tok)

    # Find the flat_proj layer name
    layer_name = None
    for name, layer in lm.layers.name_to_layer.items():
        if isinstance(layer, nn.Linear) and "flat_proj" in name:
            layer_name = name
            break
    assert layer_name is not None

    ds = make_ds(["aa", "bbb", "c"], tmp_path / "cache")
    store = LocalStore(tmp_path / "store")

    lm.activations.infer_and_save(
        ds,
        layer_signature=layer_name,
        run_name="reshape",
        store=store,
        batch_size=2,
        autocast=False,
    )

    b0 = store.get_run_batch("reshape", 0)
    acts = b0["activations"]
    inp = b0["input_ids"]
    assert acts.ndim == 3
    assert acts.shape[0] == inp.shape[0] and acts.shape[1] == inp.shape[1]
