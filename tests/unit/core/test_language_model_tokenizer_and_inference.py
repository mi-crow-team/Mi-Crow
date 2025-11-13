import types
import pytest
import torch
from torch import nn
import tempfile
from pathlib import Path
from amber.store.local_store import LocalStore

from amber.core.language_model import LanguageModel
import tempfile
from pathlib import Path
from amber.store.local_store import LocalStore


class TinyConfig:
    def __init__(self):
        self.pad_token_id = None


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 128, d_model: int = 8):
        super().__init__()
        self.config = TinyConfig()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # input_ids: [B, T]
        x = self.embed(input_ids)
        B, T, D = x.shape
        y = self.proj(x.view(B * T, D)).view(B, T, D)
        return y  # a plain Tensor so hooks can capture it directly


class FakeTokenizer:
    def __init__(self, pad_token=None, eos_token="<eos>", eos_token_id=0):
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.eos_token_id = eos_token_id
        self.pad_token_id = None

    def __len__(self):
        return 130

    def __call__(self, texts, **kwargs):  # callable path
        padding = kwargs.get("padding", False)
        rt = kwargs.get("return_tensors", None)
        # Make variable-length token id sequences based on text length
        ids = [list(range(1, 1 + min(5, len(t)))) for t in texts]
        max_len = max(len(x) for x in ids) if padding else None
        if padding:
            ids = [x + [self.pad_token_id or self.eos_token_id] * (max_len - len(x)) for x in ids]
            attn = [[1] * len(x[:- (max_len - len(x))] if (max_len - len(x)) > 0 else x) for x in ids]
            # The above is awkward; simpler construct attention based on non-pad
            attn = []
            for row in ids:
                row_attn = [1 if tok != (self.pad_token_id or self.eos_token_id) else 0 for tok in row]
                # ensure at least one token attended if all pads (shouldn't happen here)
                attn.append(row_attn)
        if rt == "pt":
            out = {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }
            return out
        return {"input_ids": ids, "attention_mask": attn}

    def add_special_tokens(self, spec):
        if "pad_token" in spec:
            self.pad_token = spec["pad_token"]
            self.pad_token_id = 1  # assign some id

    def pad(self, encoded, return_tensors="pt"):
        # Not used in callable path in these tests, but implement for completeness
        max_len = max(len(e["input_ids"]) for e in encoded)
        ids = []
        attn = []
        for e in encoded:
            x = e["input_ids"]
            pad_id = self.pad_token_id or self.eos_token_id
            padded = x + [pad_id] * (max_len - len(x))
            ids.append(padded)
            attn.append([1] * len(x) + [0] * (max_len - len(x)))
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": attn}


class FakeNonCallableTokenizer(FakeTokenizer):
    # Remove callability; provide batch_encode_plus fallback
    def __call__(self, *args, **kwargs):  # type: ignore[override]
        raise TypeError("not callable")

    def batch_encode_plus(self, texts, **kwargs):
        # mimic pad behavior via pad()
        encoded = [
            {"input_ids": list(range(1, 1 + min(4, len(t))))}
            for t in texts
        ]
        if kwargs.get("padding"):
            return self.pad(encoded, return_tensors=kwargs.get("return_tensors", "pt"))
        return encoded


def test_tokenizer_sets_pad_from_eos_and_updates_model_config():
    model = TinyLM()
    tok = FakeTokenizer(pad_token=None, eos_token="<eos>", eos_token_id=0)
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)

    # Request padding; LM wrapper should set pad_token from eos and update config
    out = lm.tokenize(["a", "bb"], padding=True, return_tensors="pt")
    assert isinstance(out["input_ids"], torch.Tensor)
    assert lm.tokenizer.pad_token == "<eos>"
    assert model.config.pad_token_id == tok.eos_token_id

    # Run a tiny forward via _inference and ensure tensors are on CPU
    inputs = ["hello", "world!"]
    output, enc = lm._inference(inputs)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    assert isinstance(input_ids, torch.Tensor) and isinstance(attn, torch.Tensor)
    assert input_ids.device.type == "cpu" and attn.device.type == "cpu"


def test_tokenizer_non_callable_uses_batch_encode_plus(tmp_path):
    model = TinyLM()
    tok = FakeNonCallableTokenizer(pad_token=None, eos_token="<eos>")
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)

    # Ensure padding logic also sets pad token from eos
    out = lm.tokenize(["x", "yyy"], padding=True, return_tensors="pt")
    assert out["input_ids"].shape[0] == 2
    assert lm.tokenizer.pad_token == "<eos>"


def test_inference_invokes_text_trackers_and_forwards_returns_output_and_enc():
    model = TinyLM()
    tok = FakeTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)

    # Register a tracker that records received texts
    class Tracker:
        def __init__(self):
            self.seen = None

        def set_current_texts(self, texts):
            self.seen = list(texts)

    tr = Tracker()
    # Directly add to activation text trackers list
    lm._activation_text_trackers.append(tr)

    texts = ["foo", "barbaz"]
    out, enc = lm.forwards(texts)
    assert isinstance(out, torch.Tensor)
    assert set(enc.keys()) >= {"input_ids", "attention_mask"}
    assert tr.seen == texts

    # Remove tracker
    if tr in lm._activation_text_trackers:
        lm._activation_text_trackers.remove(tr)
