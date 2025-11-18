import pytest
from pathlib import Path
import tempfile

from torch import nn

from amber.core.language_model import LanguageModel
from amber.store.local_store import LocalStore


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {})()

    def forward(self, *args, **kwargs):  # pragma: no cover - not used here
        return None

    def resize_token_embeddings(self, n):
        # pretend to resize; no-op
        self._resized_to = n


def test_tokenize_adds_pad_token_via_eos_when_missing_pad(monkeypatch):
    class Tok:
        def __init__(self):
            self.pad_token = None
            self.eos_token = "<eos>"
            self.eos_token_id = 99

        def __call__(self, texts, **kwargs):
            # Return minimal structure
            return {"input_ids": [], "attention_mask": []}

    tok = Tok()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / "store")
    lm = LanguageModel(model=DummyModel(), tokenizer=tok, store=store)

    out = lm.tokenize(["a", "b"], padding=True, return_tensors="pt")
    assert hasattr(tok, "pad_token") and tok.pad_token == tok.eos_token
    assert getattr(lm.model.config, "pad_token_id") == tok.eos_token_id
    assert isinstance(out, dict)


def test_tokenize_adds_new_pad_and_resizes_embeddings_when_no_eos(monkeypatch):
    class Tok:
        def __init__(self):
            self.pad_token = None
            self.pad_token_id = 123

        def add_special_tokens(self, d):
            self.pad_token = d.get("pad_token")
            self.pad_token_id = 0 if self.pad_token == "[PAD]" else -1

        def __len__(self):
            return 10

        def __call__(self, texts, **kwargs):  # Most callable path
            return {"input_ids": [], "attention_mask": []}

    tok = Tok()
    model = DummyModel()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / "store")
    lm = LanguageModel(model=model, tokenizer=tok, store=store)

    # Force padding True to trigger pad-token logic
    _ = lm.tokenize(["x", "y"], padding=True, return_tensors="pt")
    # Ensure pad added and model resized config set
    assert tok.pad_token == "[PAD]"
    assert getattr(model, "_resized_to", None) == len(tok)
    assert getattr(model.config, "pad_token_id") == tok.pad_token_id


def test_tokenize_batch_encode_plus_and_encode_plus_fallbacks():
    class NonCallableTok:
        def __init__(self):
            self.pad_token = "<pad>"

        def batch_encode_plus(self, texts, **kwargs):
            return {"input_ids": [[1], [2]], "attention_mask": [[1], [1]]}

    temp_dir = tempfile.mkdtemp()
    store1 = LocalStore(Path(temp_dir) / "store1")
    lm1 = LanguageModel(model=DummyModel(), tokenizer=NonCallableTok(), store=store1)
    out1 = lm1.tokenize(["a", "b"], padding=True)
    assert "input_ids" in out1

    class EncodePlusTok:
        def __init__(self):
            self.pad_token = "<pad>"

        def encode_plus(self, t, **kwargs):
            return {"input_ids": [1], "attention_mask": [1]}

        def pad(self, items, return_tensors="pt"):
            assert return_tensors == "pt"
            return {"input_ids": [[1], [1]], "attention_mask": [[1], [1]]}

    temp_dir2 = tempfile.mkdtemp()
    store2 = LocalStore(Path(temp_dir2) / "store2")
    lm2 = LanguageModel(model=DummyModel(), tokenizer=EncodePlusTok(), store=store2)
    out2 = lm2.tokenize(["a", "b"], padding=True, return_tensors="pt")
    assert "input_ids" in out2


def test_tokenize_raises_when_not_usable():
    class BadTok:
        pad_token = "<pad>"
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / "store")
    lm = LanguageModel(model=DummyModel(), tokenizer=BadTok(), store=store)
    with pytest.raises(TypeError):
        lm.tokenize(["a"]) 
