from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch import nn

from amber.mechanistic.autoencoder.autoencoder import Autoencoder
from amber.store import Store
from amber.utils import get_logger


@dataclass
class SAETrainingConfig:
    epochs: int = 1
    batch_size: int = 1024  # mini-batch size within each stored batch
    lr: float = 1e-3
    l1_lambda: float = 0.0  # sparsity penalty on latents
    device: str | torch.device = "cpu"
    dtype: Optional[torch.dtype] = None
    max_batches_per_epoch: Optional[int] = None  # limit for quicker iterations
    validate_every: Optional[int] = None  # if set, run a quick eval every N epochs
    checkpoint_dir: Optional[Path | str] = None
    project_decoder_grads: bool = False  # whether to project decoder grads each step
    renorm_decoder_every: Optional[int] = None  # if set, enforce unit-norm every N steps


class SAETrainer:
    """Trainer for Autoencoder using activations stored via Store/LanguageModelActivations.

    - Streams batches from a Store run_id created by LanguageModelActivations.infer_and_save.
    - Extracts the 'activations' tensor, flattens leading dims to [N, hidden_dim],
      and trains the Autoencoder to reconstruct inputs.
    - Supports optional L1 sparsity on latents and periodic checkpointing.
    """

    def __init__(
            self,
            model: Autoencoder,
            store: Store,
            run_id: str,
            config: SAETrainingConfig | None = None,
    ) -> None:
        self.model = model
        self.store = store
        self.run_id = run_id
        self.cfg = config or SAETrainingConfig()
        self.logger = get_logger(__name__)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.criterion = nn.MSELoss()

        # Device/dtype
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)

    def _iterate_activation_minibatches(self) -> Iterator[torch.Tensor]:
        """Yield input minibatches shaped [B, hidden_dim] for training.

        Loads each stored batch dict, reads 'activations' tensor of shape
        [batch, seq, hidden_dim] or [batch, hidden_dim], flattens leading dims
        to 2D, casts/places on device, and yields in mini-batches of cfg.batch_size.
        """
        for batch in self.store.iter_run_batches(self.run_id):
            if not isinstance(batch, dict) or "activations" not in batch:
                # Skip malformed batches
                continue
            acts = batch["activations"]
            if not isinstance(acts, torch.Tensor):
                continue
            # Ensure 2D [N, D]
            if acts.dim() > 2:
                d = acts.shape[-1]
                acts = acts.view(-1, d)
            elif acts.dim() == 1:
                acts = acts.view(1, -1)
            # dtype handling (keep storage on CPU for slicing; move per-mini-batch)
            if self.cfg.dtype is not None:
                acts = acts.to(self.cfg.dtype)

            # Yield mini-batches
            bs = max(1, int(self.cfg.batch_size))
            n = acts.shape[0]
            for start in range(0, n, bs):
                yield acts[start:start + bs]

    def train(self) -> dict[str, list[float]]:
        self.model.train()
        history: dict[str, list[float]] = {"loss": [], "recon_mse": [], "l1": []}
        step = 0
        for epoch in range(self.cfg.epochs):
            epoch_losses = []
            epoch_mse = []
            epoch_l1 = []
            batches_seen = 0
            for inputs in self._iterate_activation_minibatches():
                if self.cfg.max_batches_per_epoch is not None and batches_seen >= self.cfg.max_batches_per_epoch:
                    break
                batches_seen += 1

                # Move to device as late as possible to keep host memory low
                x = inputs.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                recon, latents, recon_full, _lat_full = self.model(x)

                mse = self.criterion(recon, x)
                l1 = latents.abs().mean() if self.cfg.l1_lambda > 0 else torch.tensor(0.0, device=self.device)
                loss = mse + self.cfg.l1_lambda * l1

                loss.backward()

                if self.cfg.project_decoder_grads:
                    with torch.no_grad():
                        self.model.project_grads_decode()

                self.optimizer.step()

                if self.cfg.renorm_decoder_every and (step + 1) % int(self.cfg.renorm_decoder_every) == 0:
                    with torch.no_grad():
                        self.model.scale_to_unit_norm()

                epoch_losses.append(float(loss.detach().cpu()))
                epoch_mse.append(float(mse.detach().cpu()))
                epoch_l1.append(float(l1.detach().cpu()))
                step += 1

            # Aggregate and log
            if epoch_losses:
                mean_loss = sum(epoch_losses) / len(epoch_losses)
                mean_mse = sum(epoch_mse) / len(epoch_mse)
                mean_l1 = sum(epoch_l1) / len(epoch_l1)
            else:
                mean_loss = mean_mse = mean_l1 = float("nan")

            history["loss"].append(mean_loss)
            history["recon_mse"].append(mean_mse)
            history["l1"].append(mean_l1)

            self.logger.info(
                f"[SAETrainer] epoch={epoch+1}/{self.cfg.epochs} batches={batches_seen} "
                f"loss={mean_loss:.6f} mse={mean_mse:.6f} l1={mean_l1:.6f}")

            # Optional checkpoint
            if self.cfg.checkpoint_dir is not None:
                ckpt_dir = Path(self.cfg.checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                # Use Autoencoder.save for consistency
                try:
                    self.model.save(f"epoch_{epoch+1:03d}", path=str(ckpt_dir))
                except Exception:
                    # Fallback to torch.save for robustness
                    torch.save(self.model.state_dict(), str(ckpt_dir / f"epoch_{epoch+1:03d}.pt"))

        return history


# Backward-compat wrapper for old API name (kept minimal)

def train_sae(
        sae: Autoencoder,
        store: Store,
        run_id: str,
        epochs: int = 1,
        batch_size: int | None = None,
        learning_rate: float = 1e-3,
        l1_lambda: float = 0.0,
) -> list[float]:
    """Train an Autoencoder from Store activations. Returns loss history per epoch.

    This is a thin wrapper around SAETrainer to preserve a simple function API.
    """
    cfg = SAETrainingConfig(
        epochs=epochs,
        batch_size=batch_size or 1024,
        lr=learning_rate,
        l1_lambda=l1_lambda,
    )
    trainer = SAETrainer(sae, store, run_id, cfg)
    hist = trainer.train()
    return hist["loss"]
