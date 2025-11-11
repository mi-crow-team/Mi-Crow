from types import SimpleNamespace
import tempfile

import torch
from torch import nn
from datasets import Dataset
import tempfile

from amber.core.language_model import LanguageModel
from amber.store.local_store import LocalStore
from amber.adapters.text_snippet_dataset import TextSnippetDataset
import tempfile


class FakeTokenizer:
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, texts, **kwargs):
        padding = kwargs.get("padding", False)
        truncation = kwargs.get("truncation", False)
        max_length = kwargs.get("max_length")
        return_tensors = kwargs.get("return_tensors", "pt")

        # Very small whitespace tokenizer mapping tokens to incremental ids per call
        vocab: dict[str, int] = {}
        def _enc_one(t: str) -> list[int]:
            ids: list[int] = []
            for tok in t.split():
                if tok not in vocab:
                    vocab[tok] = len(vocab) + 1  # 0 is padding
                ids.append(vocab[tok])
            if not ids:
                ids = [self.pad_id]
            return ids

        encoded = [_enc_one(t) for t in texts]
        if truncation and max_length is not None:
            encoded = [e[: max_length] for e in encoded]
        lengths = [len(e) for e in encoded]
        max_len = max(lengths) if padding else max(lengths)
        if padding:
            encoded = [e + [self.pad_id] * (max_len - len(e)) for e in encoded]
        input_ids = torch.tensor(encoded, dtype=torch.long)
        # Return only input_ids to exercise branch where attention_mask may be missing
        if return_tensors == "pt":
            return {"input_ids": input_ids}
        raise ValueError("Only return_tensors='pt' supported in FakeTokenizer")


class ReturnNamespace(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        y = self.inner(x)
        return SimpleNamespace(last_hidden_state=y)


class ToyLMBranchy(nn.Module):
    def __init__(self, vocab_size: int = 50, d_model: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.lin = nn.Linear(d_model, d_model)
        self.out = ReturnNamespace(nn.Linear(d_model, d_model))

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        x = self.lin(x)
        x = self.out(x)
        return x


def _make_ds(texts: list[str], cache_dir) -> TextSnippetDataset:
    base = Dataset.from_dict({"text": texts})
    return TextSnippetDataset(base, cache_dir=cache_dir)


def test_infer_and_save_captures_last_hidden_state_and_defaults(tmp_path, monkeypatch):
    """Test that namespace outputs with last_hidden_state are captured correctly."""
    # Monkeypatch datetime used in LanguageModelActivations to force deterministic run_name
    class _FixedDT:
        class datetime:
            @staticmethod
            def now():
                class _N:
                    def strftime(self, fmt):
                        return "20250101_000000"
                return _N()
    import amber.core.language_model_activations as _act_mod
    monkeypatch.setattr(_act_mod, "datetime", _FixedDT, raising=False)

    tok = FakeTokenizer()
    net = ToyLMBranchy(vocab_size=30, d_model=6)
    store = LocalStore(tmp_path / "store")
    lm = LanguageModel(model=net, tokenizer=tok, store=store)

    texts = ["a b", "c d e", "f", "g h", "i j k"]  # 5 items -> batches (2,2,1)
    ds = _make_ds(texts, tmp_path / "cacheA")

    # Use the wrapper module that returns a namespace; capture by name
    target_name = None
    for name, layer in lm.layers.name_to_layer.items():
        if isinstance(layer, ReturnNamespace):
            target_name = name
            break
    assert target_name is not None

    # Pre-scan existing run directories
    from pathlib import Path
    runs_dir = Path(lm.store.base_path) / "activations"
    runs_dir.mkdir(parents=True, exist_ok=True)
    before_runs = {p.name for p in runs_dir.iterdir() if p.is_dir()}

    # store=None and run_name=None exercise defaults; verbose=True exercises logging path
    lm.activations.infer_and_save(
        ds,
        layer_signature=target_name,
        run_name=None,
        store=None,
        batch_size=2,
        autocast=True,  # on CPU -> Noop autocast branch
        verbose=True,
    )

    # Discover newly created run_id under the default store path
    after_runs = {p.name for p in runs_dir.iterdir() if p.is_dir()}
    new_runs = sorted(after_runs - before_runs)
    assert len(new_runs) == 1
    run_id = new_runs[0]
    batches = lm.store.list_run_batches(run_id)
    assert batches == [0, 1, 2]
    b0 = lm.store.get_run_batch(run_id, 0)
    # Activations present and inputs include input_ids; attention_mask may be missing
    assert "activations" in b0
    assert "input_ids" in b0
