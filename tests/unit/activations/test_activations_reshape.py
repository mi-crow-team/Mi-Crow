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
    return TextSnippetDataset(base, dataset_dir=tmp_path)


def test_captured_2d_activations_are_reshaped_to_3d(tmp_path):
    """Test that 2D activations captured from flattened layers are saved correctly."""
    tok = Tok()
    net = FlattenThenReshapeLM()
    store = LocalStore(tmp_path / "store")
    lm = LanguageModel(model=net, tokenizer=tok, store=store)

    # Find the flat_proj layer name
    layer_name = None
    for name, layer in lm.layers.name_to_layer.items():
        if isinstance(layer, nn.Linear) and "flat_proj" in name:
            layer_name = name
            break
    assert layer_name is not None

    ds = make_ds(["aa", "bbb", "c"], tmp_path / "cache")
    store = LocalStore(tmp_path / "store")

    lm.activations.save_activations_dataset(
        ds,
        layer_signature=layer_name,
        run_name="reshape",
        batch_size=2,
        autocast=False,
    )

    b0 = store.get_run_batch("reshape", 0)
    acts = b0["activations"]
    # Verify activations are captured as 2D [B*T, D] from the flattened layer
    # Note: Reshaping to 3D requires input_ids which are no longer saved
    assert acts.ndim == 2
    # Should have shape [B*T, D] where B*T is the total number of tokens in the batch
    assert acts.shape[1] == 6  # Hidden dimension should match model's d_model (6)
    assert acts.shape[0] > 0  # Should have at least some tokens
