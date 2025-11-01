import torch
from torch import nn
from datasets import Dataset

from amber.core.language_model import LanguageModel
from amber.core.language_model_activations import LanguageModelActivations
from amber.adapters.text_snippet_dataset import TextSnippetDataset
from amber.store import LocalStore


class TinyLM(nn.Module):
    def __init__(self, d_model: int = 8, vocab_size: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, d_model)

        class Cfg:
            pad_token_id = None
            name_or_path = "TinyLM"
        self.config = Cfg()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        x = self.embed(input_ids)
        B, T, D = x.shape
        y = self.proj(x.view(B * T, D)).view(B, T, D)
        return y


class FakeTokenizer:
    def __init__(self):
        self.eos_token = "<eos>"
        self.eos_token_id = 0
        self.pad_token = None
        self.pad_token_id = None

    def __len__(self):
        return 256

    def __call__(self, texts, **kwargs):
        # simple fixed-length tokenization for determinism
        max_len = 4
        ids = []
        attn = []
        for t in texts:
            n = min(max_len, max(1, len(t) % (max_len + 1)))
            row = list(range(1, 1 + n))
            pad_id = self.pad_token_id or self.eos_token_id
            ids.append(row + [pad_id] * (max_len - n))
            attn.append([1] * n + [0] * (max_len - n))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn)}


def _layer_sig(lm: nn.Module) -> str:
    return f"{lm.__class__.__name__.lower()}_proj"


def test_infer_and_save_writes_batches_and_meta(tmp_path):
    # Build tiny dataset
    base = Dataset.from_dict({"text": ["a", "bb", "ccc", "dddd", "ee"]})
    ds = TextSnippetDataset(base, cache_dir=tmp_path)

    model = TinyLM()
    tok = FakeTokenizer()
    lm = LanguageModel(model=model, tokenizer=tok, store=LocalStore(tmp_path / "store"))

    lma = LanguageModelActivations(lm.context)
    run = "unittest_run"
    layer_sig = _layer_sig(model)

    lma.infer_and_save(
        ds,
        layer_signature=layer_sig,
        run_name=run,
        store=lm.store,
        batch_size=2,
        dtype=torch.float32,
        autocast=False,
        save_inputs=True,
        verbose=False,
    )

    # Expect 3 batches for 5 examples with batch_size=2
    batches = lm.store.list_run_batches(run)
    assert batches == [0, 1, 2]

    # Load a batch and check keys
    batch0 = lm.store.get_run_batch(run, 0)
    assert set(batch0.keys()) >= {"activations", "input_ids", "attention_mask"}
    acts = batch0["activations"]
    assert isinstance(acts, torch.Tensor)
    # Expect [B, T, D] with B=2, T=4, D=d_model=8
    assert acts.dim() == 3 and acts.shape[1] == 4

    # Metadata exists and contains run_name
    meta = lm.store.get_run_meta(run)
    assert meta.get("run_name") == run
