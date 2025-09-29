from types import SimpleNamespace

import torch
from torch import nn
from datasets import Dataset

from amber.core.language_model import LanguageModel
from amber.adapters.text_snippet_dataset import TextSnippetDataset


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


class ReturnTuple(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        y = self.inner(x)
        return (y,)


class ReturnNumber(nn.Module):
    def forward(self, x):
        # Return a non-tensor to exercise the 'no activations captured' path
        return 123


class ToyLMBranchy(nn.Module):
    def __init__(self, vocab_size: int = 50, d_model: int = 8, mode: str = "namespace"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.lin = nn.Linear(d_model, d_model)
        if mode == "namespace":
            self.out = ReturnNamespace(nn.Linear(d_model, d_model))
        elif mode == "tuple":
            self.out = ReturnTuple(nn.Linear(d_model, d_model))
        elif mode == "nontensor":
            self.out = ReturnNumber()
        else:
            raise ValueError("unknown mode")

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        x = self.lin(x)
        x = self.out(x)
        return x


def _make_ds(texts: list[str], cache_dir) -> TextSnippetDataset:
    base = Dataset.from_dict({"text": texts})
    return TextSnippetDataset(base, cache_dir=cache_dir)


def test_infer_and_save_captures_last_hidden_state_and_defaults(tmp_path, monkeypatch):
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
    net = ToyLMBranchy(vocab_size=30, d_model=6, mode="namespace")
    lm = LanguageModel(model=net, tokenizer=tok)

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
    runs_dir = Path(lm.store.base_path) / "runs"
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


def test_infer_and_save_without_activations_saves_inputs_only_and_index_signature(tmp_path):
    tok = FakeTokenizer()
    net = ToyLMBranchy(vocab_size=20, d_model=4, mode="nontensor")
    lm = LanguageModel(model=net, tokenizer=tok)

    texts = ["x y", "z", "u v w"]
    ds = _make_ds(texts, tmp_path / "cacheB")

    # Find numeric index of the ReturnNumber layer
    idx_target = None
    for idx, layer in lm.layers.idx_to_layer.items():
        if isinstance(layer, ReturnNumber):
            idx_target = idx
            break
    assert idx_target is not None

    run_id = "inputs_only"
    lm.activations.infer_and_save(
        ds,
        layer_signature=idx_target,  # use integer signature path
        run_name=run_id,
        store=lm.store,
        batch_size=2,
        autocast=True,
        verbose=True,
    )

    batches = lm.store.list_run_batches(run_id)
    assert batches == [0, 1]
    for bi in batches:
        b = lm.store.get_run_batch(run_id, bi)
        # No activations key because hook returned a non-tensor object
        assert "activations" not in b
        # Inputs are still saved
        assert "input_ids" in b


def test_infer_and_save_tuple_output_branch(tmp_path):
    tok = FakeTokenizer()
    net = ToyLMBranchy(vocab_size=25, d_model=5, mode="tuple")
    lm = LanguageModel(model=net, tokenizer=tok)

    texts = ["aa bb", "cc", "dd ee ff"]
    ds = _make_ds(texts, tmp_path / "cacheC")

    # Locate the tuple-returning module by name
    target_name = None
    for name, layer in lm.layers.name_to_layer.items():
        if isinstance(layer, ReturnTuple):
            target_name = name
            break
    assert target_name is not None

    run_id = "tuple_run"
    lm.activations.infer_and_save(
        ds,
        layer_signature=target_name,
        run_name=run_id,
        store=lm.store,
        batch_size=2,
        autocast=False,
        verbose=False,
    )

    batches = lm.store.list_run_batches(run_id)
    assert batches == [0, 1]
    b0 = lm.store.get_run_batch(run_id, 0)
    assert "activations" in b0 and b0["activations"].ndim == 3
