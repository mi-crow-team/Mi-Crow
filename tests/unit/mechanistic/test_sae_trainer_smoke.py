import torch

from amber.mechanistic.autoencoder.autoencoder import Autoencoder
from amber.mechanistic.autoencoder.train import SAETrainer, SAETrainingConfig, train_sae
from amber.store import LocalStore


def make_fake_run(store: LocalStore, run_id: str, hidden_dim: int = 16) -> None:
    # Two batches with different token lengths
    batch0 = {
        "activations": torch.randn(2, 3, hidden_dim),  # [batch, seq, dim]
        "input_ids": torch.randint(0, 100, (2, 3)),
    }
    batch1 = {
        "activations": torch.randn(1, 5, hidden_dim),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    store.put_run_batch(run_id, 0, batch0)
    store.put_run_batch(run_id, 1, batch1)


def test_sae_trainer_smoke(tmp_path):
    store = LocalStore(tmp_path)
    run_id = "demo_run"
    hidden_dim = 16
    make_fake_run(store, run_id, hidden_dim)

    # Autoencoder with small latent size
    ae = Autoencoder(n_latents=8, n_inputs=hidden_dim, activation="TopK_4")

    cfg = SAETrainingConfig(
        epochs=1,
        batch_size=4,
        lr=1e-2,
        l1_lambda=0.0,
        device="cpu",
        max_batches_per_epoch=10,
        checkpoint_dir=tmp_path / "ckpts",
        project_decoder_grads=True,
        renorm_decoder_every=5,
    )

    trainer = SAETrainer(ae, store, run_id, cfg)
    history = trainer.train()

    assert "loss" in history and len(history["loss"]) == cfg.epochs
    assert "recon_mse" in history and len(history["recon_mse"]) == cfg.epochs

    # Wrapper should also work
    loss_hist = train_sae(ae, store, run_id, epochs=1, batch_size=4, learning_rate=1e-2)
    assert isinstance(loss_hist, list) and len(loss_hist) == 1
