import math
import torch
from torch import nn

from datasets import Dataset

from amber.core.language_model import LanguageModel
from amber.adapters.text_snippet_dataset import TextSnippetDataset
from amber.store import LocalStore
from amber.mechanistic.autoencoder.autoencoder import Autoencoder
from amber.mechanistic.autoencoder.train import SAETrainer, SAETrainingConfig, train_sae


class FakeTokenizer:
    def __init__(self, vocab: dict[str, int] | None = None, pad_id: int = 0):
        self.vocab = vocab or {}
        self.pad_id = pad_id

    def _encode_one(self, text: str) -> list[int]:
        ids = []
        for tok in text.split():
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab) + 1  # 0 is padding
            ids.append(self.vocab[tok])
        if not ids:
            ids = [self.pad_id]
        return ids

    def __call__(self, texts, **kwargs):
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
    def __init__(self, vocab_size: int = 100, d_model: int = 12):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)  # [B, T, D]
        x = self.block(x)          # [B, T, D]
        x = self.proj(x)           # [B, T, D]
        return x


def make_snippet_ds(texts: list[str], cache_dir) -> TextSnippetDataset:
    base = Dataset.from_dict({"text": texts})
    return TextSnippetDataset(base, cache_dir=cache_dir)


def test_sae_trains_from_saved_activations_e2e(tmp_path):
    # 1) Tiny LM + tokenizer wrapped by LanguageModel
    tok = FakeTokenizer()
    d_model = 12
    net = ToyLM(vocab_size=50, d_model=d_model)
    lm = LanguageModel(model=net, tokenizer=tok)

    # 2) Small dataset -> ensure multiple batches
    texts = [f"hello {i}" for i in range(9)]  # 9 items
    ds = make_snippet_ds(texts, tmp_path / "ds_cache")

    # 3) Choose a concrete layer name (e.g., proj)
    target_layer_name = None
    for name, layer in lm.layers.name_to_layer.items():
        if isinstance(layer, nn.Linear) and "proj" in name:
            target_layer_name = name
            break
    assert target_layer_name is not None, "Expected to find proj linear layer"

    store = LocalStore(tmp_path / "store")

    # 4) Save activations to the store (CPU only, no autocast)
    lm.activations.infer_and_save(
        ds,
        layer_signature=target_layer_name,
        run_name="sae_run",
        store=store,
        batch_size=4,  # -> 3 batches: 4,4,1
        autocast=False,
    )

    # Verify batches exist
    assert store.list_run_batches("sae_run") == [0, 1, 2]

    # 5) Build a small SAE and train for a couple of epochs from saved activations
    sae = Autoencoder(n_latents=6, n_inputs=d_model, activation="TopK_4")

    cfg = SAETrainingConfig(
        epochs=2,
        batch_size=8,  # minibatch slicing inside a stored batch
        lr=5e-3,
        l1_lambda=0.0,
        device="cpu",
        max_batches_per_epoch=10,
        project_decoder_grads=True,
        renorm_decoder_every=5,
    )

    trainer = SAETrainer(sae, store, run_id="sae_run", config=cfg)
    history = trainer.train()

    # Basic assertions on training history
    assert set(history.keys()) >= {"loss", "recon_mse", "l1"}
    assert len(history["loss"]) == cfg.epochs
    for k in ("loss", "recon_mse", "l1"):
        for v in history[k]:
            assert isinstance(v, float)
            assert math.isfinite(v)
            assert v >= 0.0

    # 6) Backward-compat wrapper also should work
    loss_hist = train_sae(sae, store, run_id="sae_run", epochs=1, batch_size=8, learning_rate=1e-2)
    assert isinstance(loss_hist, list) and len(loss_hist) == 1
