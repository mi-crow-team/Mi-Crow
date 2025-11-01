import types
import torch
from torch import nn
import pytest

from amber.core.language_model import LanguageModel


class TinyConfig:
    def __init__(self):
        self.pad_token_id = None


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 16, d_model: int = 4):
        super().__init__()
        self.config = TinyConfig()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        return self.embed(input_ids)


class EncodePlusOnlyTokenizer:
    """
    A tokenizer that is not usable as a callable and has no batch_encode_plus,
    but supports encode_plus and pad. This forces LanguageModelTokenizer to hit
    the encode_plus + pad fallback path.
    """

    def __init__(self, has_eos: bool = False):
        # simulate missing eos to trigger add_special_tokens path when padding
        self.eos_token = "<eos>" if has_eos else None
        self.eos_token_id = 0 if has_eos else None
        self.pad_token = None
        self.pad_token_id = None

    def __len__(self):
        # used when resize_token_embeddings is called
        return 17

    def __call__(self, *args, **kwargs):  # type: ignore[override]
        # Pretend to be callable but raise TypeError to trigger fallback
        raise TypeError("not callable")

    def encode_plus(self, text, **kwargs):
        # return variable-length ids to exercise padding
        ids = list(range(1, 1 + min(3, len(str(text)))))
        return {"input_ids": ids}

    def pad(self, encoded, return_tensors="pt"):
        max_len = max(len(e["input_ids"]) for e in encoded)
        ids = []
        attn = []
        pad_id = self.pad_token_id if self.pad_token_id is not None else 0
        for e in encoded:
            row = e["input_ids"]
            pad = max_len - len(row)
            ids.append(row + [pad_id] * pad)
            attn.append([1] * len(row) + [0] * pad)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": attn}

    def add_special_tokens(self, spec):
        if "pad_token" in spec:
            self.pad_token = spec["pad_token"]
            # assign a made-up id for pad token
            self.pad_token_id = 16


def test_tokenizer_encode_plus_and_pad_fallback_respects_return_tensors():
    model = TinyLM()
    tok = EncodePlusOnlyTokenizer(has_eos=True)
    lm = LanguageModel(model=model, tokenizer=tok)  # type: ignore[arg-type]

    out = lm.tokenize(["a", "bbbb"], padding=True, return_tensors="pt")
    assert isinstance(out["input_ids"], torch.Tensor)
    assert out["input_ids"].shape[0] == 2
    # pad_token should be derived from eos and model config updated
    assert lm.tokenizer.pad_token == "<eos>"
    assert lm.model.config.pad_token_id == tok.eos_token_id


def test_tokenizer_adds_pad_token_and_resizes_embeddings_when_no_eos(monkeypatch):
    model = TinyLM()
    tok = EncodePlusOnlyTokenizer(has_eos=False)

    # Spy on resize_token_embeddings to ensure it gets called
    called = {"n": 0, "size": None}

    def fake_resize(n):
        called["n"] = called["n"] + 1
        called["size"] = n

    # Attach method to model instance
    model.resize_token_embeddings = fake_resize  # type: ignore[attr-defined]

    lm = LanguageModel(model=model, tokenizer=tok)  # type: ignore[arg-type]

    out = lm.tokenize(["x", "yy"], padding=True, return_tensors="pt")
    assert isinstance(out["input_ids"], torch.Tensor)

    # Since no eos_token, tokenizer should add a new pad token and call resize
    assert tok.pad_token == "[PAD]"
    assert tok.pad_token_id is not None
    assert called["n"] == 1 and called["size"] == len(tok)
    # model.config.pad_token_id should be set as well
    assert lm.model.config.pad_token_id == tok.pad_token_id
