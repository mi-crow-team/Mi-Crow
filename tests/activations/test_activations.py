import math
from typing import Sequence, Any

import torch
from torch import nn

from amber.adapters.language_model import LanguageModel
from amber.adapters.text_snippet_dataset import TextSnippetDataset
from amber.store import LocalStore
from datasets import Dataset


class FakeTokenizer:
    def __init__(self, vocab: dict[str, int] | None = None, pad_id: int = 0):
        self.vocab = vocab or {}
        self.pad_id = pad_id

    def _encode_one(self, text: str) -> list[int]:
        # Very small whitespace tokenizer mapping tokens to incremental ids
        ids = []
        for tok in text.split():
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab) + 1  # 0 is padding
            ids.append(self.vocab[tok])
        if not ids:
            ids = [self.pad_id]
        return ids

    def __call__(self, texts: Sequence[str], **kwargs: Any):
        padding = kwargs.get("padding", False)
        truncation = kwargs.get("truncation", False)
        max_length = kwargs.get("max_length")
        return_tensors = kwargs.get("return_tensors", "pt")

        encoded = [self._encode_one(t) for t in texts]
        if truncation and max_length is not None:
            encoded = [e[: max_length] for e in encoded]
        lengths = [len(e) for e in encoded]
        max_len = max(lengths) if padding else max(lengths)
        if padding:
            encoded = [e + [self.pad_id] * (max_len - len(e)) for e in encoded]
        input_ids = torch.tensor(encoded, dtype=torch.long)
        attention_mask = torch.tensor([[1] * l + [0] * (max_len - l) for l in lengths], dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        raise ValueError("Only return_tensors='pt' supported in FakeTokenizer")


class ToyLM(nn.Module):
    def __init__(self, vocab_size: int = 100, d_model: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, input_ids, attention_mask=None):  # noqa: D401
        x = self.embed(input_ids)  # [B, T, D]
        x = self.block(x)          # [B, T, D]
        x = self.proj(x)           # [B, T, D]
        return x  # last_hidden_state-like tensor


def make_snippet_ds(texts: list[str], tmp_path) -> TextSnippetDataset:
    base = Dataset.from_dict({"text": texts})
    return TextSnippetDataset(base, cache_dir=tmp_path)


def test_save_model_activations_persists_batches_and_shapes(tmp_path):
    # Build tiny LM wrapper
    tok = FakeTokenizer()
    net = ToyLM(vocab_size=50, d_model=8)
    lm = LanguageModel(model=net, tokenizer=tok)

    # Dataset of 10 items, batch size 4 -> 3 batches (4,4,2)
    texts = [f"hello {i}" for i in range(10)]
    ds = make_snippet_ds(texts, tmp_path / "ds_cache")

    # Pick a layer name that exists (e.g., the proj Linear layer)
    target_layer_name = None
    for name, layer in lm.name_to_layer.items():
        if isinstance(layer, nn.Linear) and "proj" in name:
            target_layer_name = name
            break
    assert target_layer_name is not None, "Expected to find proj linear layer in flattened names"

    store = LocalStore(tmp_path / "store")

    # Import and run activation saver
    from amber.activations import save_model_activations

    save_model_activations(
        lm,
        ds,
        store,
        run_name="runA",
        layer_signature=target_layer_name,
        batch_size=4,
        autocast=False,  # keep CPU simple
    )

    # Verify batches written
    batches = store.list_run_batches("runA")
    assert batches == [0, 1, 2]

    # Load first batch and validate keys and shapes
    batch0 = store.get_run_batch("runA", 0)
    assert set(batch0.keys()) >= {"activations", "input_ids", "attention_mask"}
    acts = batch0["activations"]
    inp = batch0["input_ids"]
    attn = batch0["attention_mask"]

    # Shapes: [B, T, D] for activations; [B, T] for inputs
    assert acts.ndim == 3
    assert inp.ndim == 2 and attn.ndim == 2
    B, T, D = acts.shape
    assert B == 4 and D == 8 and T == inp.shape[1] == attn.shape[1]


def test_save_model_activations_options_maxlen_dtype_noinputs(tmp_path):
    tok = FakeTokenizer()
    net = ToyLM(vocab_size=30, d_model=6)
    lm = LanguageModel(model=net, tokenizer=tok)

    texts = ["a b c d", "e f", "g", "h i j k l m", "n o p"]
    ds = make_snippet_ds(texts, tmp_path / "ds2")

    # Target: first Linear in block
    target_layer_name = None
    for name, layer in lm.name_to_layer.items():
        if isinstance(layer, nn.Linear) and "block" in name:
            target_layer_name = name
            break
    assert target_layer_name is not None

    store = LocalStore(tmp_path / "store2")

    from amber.activations import save_model_activations

    # Use max_length=3 and downcast to float16; don't save inputs
    save_model_activations(
        lm,
        ds,
        store,
        run_name="runB",
        layer_signature=target_layer_name,
        batch_size=2,
        max_length=3,
        dtype=torch.float16,
        save_inputs=False,
        autocast=False,
    )

    batches = store.list_run_batches("runB")
    # 5 texts, batch size 2 -> 3 batches (2,2,1)
    assert batches == [0, 1, 2]

    b1 = store.get_run_batch("runB", 1)
    # Inputs should not be present
    assert "input_ids" not in b1 and "attention_mask" not in b1
    acts = b1["activations"]
    assert acts.dtype == torch.float16
    # Sequence length should be truncated to 3
    assert acts.shape[1] == 3
