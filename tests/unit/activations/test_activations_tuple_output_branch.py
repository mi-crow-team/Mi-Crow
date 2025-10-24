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


class ReturnTuple(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        y = self.inner(x)
        return (y,)


class ToyLMBranchy(nn.Module):
    def __init__(self, vocab_size: int = 50, d_model: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.lin = nn.Linear(d_model, d_model)
        self.out = ReturnTuple(nn.Linear(d_model, d_model))

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        x = self.lin(x)
        x = self.out(x)
        return x


def _make_ds(texts: list[str], cache_dir) -> TextSnippetDataset:
    base = Dataset.from_dict({"text": texts})
    return TextSnippetDataset(base, cache_dir=cache_dir)


def test_infer_and_save_tuple_output_branch(tmp_path):
    """Test that tuple outputs are handled correctly."""
    tok = FakeTokenizer()
    net = ToyLMBranchy(vocab_size=25, d_model=5)
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
