import torch
from torch import nn
from datasets import Dataset
import tempfile
from pathlib import Path

from amber.core.language_model import LanguageModel
from amber.store.local_store import LocalStore
from amber.adapters.text_snippet_dataset import TextSnippetDataset
import tempfile
from pathlib import Path


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


class ReturnNumber(nn.Module):
    def forward(self, x):
        # Return a non-tensor to exercise the 'no activations captured' path
        return 123


class ToyLMBranchy(nn.Module):
    def __init__(self, vocab_size: int = 50, d_model: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.lin = nn.Linear(d_model, d_model)
        self.out = ReturnNumber()

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        x = self.lin(x)
        x = self.out(x)
        return x


def _make_ds(texts: list[str], cache_dir) -> TextSnippetDataset:
    base = Dataset.from_dict({"text": texts})
    return TextSnippetDataset(base, cache_dir=cache_dir)


def test_infer_and_save_without_activations_saves_inputs_only_and_index_signature(tmp_path):
    """Test that when no activations are captured, inputs are still saved and index signatures work."""
    tok = FakeTokenizer()
    net = ToyLMBranchy(vocab_size=20, d_model=4)
    store = LocalStore(tmp_path / "store")
    lm = LanguageModel(model=net, tokenizer=tok, store=store)

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
