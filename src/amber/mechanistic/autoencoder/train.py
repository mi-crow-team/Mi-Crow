from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import os

import torch
from torch import nn

from amber.mechanistic.autoencoder.autoencoder import Autoencoder
from amber.store import Store
from amber.utils import get_logger

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        return iterable if iterable is not None else range(0)


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
    verbose: bool = False  # if True, enable tqdm progress bars and extra logging
    # Memory-/stability-friendly options
    use_amp: bool = True  # enable autocast mixed precision when possible
    amp_dtype: Optional[torch.dtype] = None  # override autocast dtype; defaults to cfg.dtype if set
    grad_accum_steps: int = 1  # gradient accumulation to emulate larger batches with lower memory
    free_cuda_cache_every: Optional[int] = None  # call torch.cuda.empty_cache() every N optimizer steps
    # Weights & Biases (auto-enable only if user is logged in)
    wandb_enable: Optional[bool] = None  # None = auto (enable if logged in), True = force, False = disable
    wandb_project: Optional[str] = None  # default: derived from package name
    wandb_run_name: Optional[str] = None  # default: uses run_id
    wandb_entity: Optional[str] = None  # optional: org/user name


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
        if self.cfg.dtype is not None:
            try:
                self.model.to(self.device, dtype=self.cfg.dtype)
            except TypeError:
                # older PyTorch may not support dtype kwarg on .to for modules
                self.model.to(self.device)
                for p in self.model.parameters():
                    p.data = p.data.to(self.cfg.dtype)
        else:
            self.model.to(self.device)

        # AMP scaler (only meaningful on CUDA). Disable on MPS to avoid instability.
        self._use_amp = bool(self.cfg.use_amp) and (self.device.type in ("cuda", "cpu"))
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp and self.device.type == "cuda")

        # Weights & Biases (auto-enable if logged in)
        self._wandb_run = None
        self._wandb = None
        try:
            if self.cfg.wandb_enable is not False:
                import wandb  # type: ignore
                self._wandb = wandb
                # Determine if user is logged in; prefer API when available
                is_logged_in = False
                try:
                    is_logged_in = bool(getattr(wandb, "is_logged_in", lambda: False)())
                except Exception:
                    is_logged_in = False
                # Fallback: env key present implies configured
                if not is_logged_in:
                    is_logged_in = bool(os.environ.get("WANDB_API_KEY"))
                if self.cfg.wandb_enable or is_logged_in:
                    project = self.cfg.wandb_project or "amber"
                    name = self.cfg.wandb_run_name or f"sae_{self.run_id}"
                    self._wandb_run = wandb.init(project=project, name=name, entity=self.cfg.wandb_entity)
                    # Log static config
                    wandb.config.update({
                        "run_id": self.run_id,
                        "epochs": self.cfg.epochs,
                        "batch_size": self.cfg.batch_size,
                        "lr": self.cfg.lr,
                        "l1_lambda": self.cfg.l1_lambda,
                        "device": str(self.device),
                        "dtype": str(self.cfg.dtype),
                        "use_amp": self._use_amp,
                        "grad_accum_steps": self.cfg.grad_accum_steps,
                    }, allow_val_change=True)
                    if self.cfg.verbose:
                        self.logger.info("[SAETrainer] Weights & Biases enabled: project=%s name=%s", project, name)
        except Exception:
            # Any wandb issues should not break training
            self._wandb_run = None
            self._wandb = None

        # Verbose init logging
        if self.cfg.verbose:
            self.logger.info(
                "[SAETrainer] device=%s dtype=%s use_amp=%s grad_accum_steps=%s batch_size=%s lr=%s",
                str(self.device), str(self.cfg.dtype), str(self._use_amp), str(self.cfg.grad_accum_steps),
                str(self.cfg.batch_size), str(self.cfg.lr),
            )

    def _iterate_activation_minibatches(self) -> Iterator[torch.Tensor]:
        """Yield input minibatches shaped [B, hidden_dim] for training.

        Loads each stored batch dict, reads 'activations' tensor of shape
        [batch, seq, hidden_dim] or [batch, hidden_dim], flattens leading dims
        to 2D, casts/places on device, and yields in mini-batches of cfg.batch_size.
        """
        idx = 0
        for batch in self.store.iter_run_batches(self.run_id):
            idx += 1
            if not isinstance(batch, dict) or "activations" not in batch:
                if self.cfg.verbose:
                    self.logger.info("[SAETrainer] Skipping non-dict or missing 'activations' in batch #%d", idx)
                continue
            acts = batch["activations"]
            if not isinstance(acts, torch.Tensor):
                if self.cfg.verbose:
                    self.logger.info("[SAETrainer] Skipping non-tensor 'activations' in batch #%d (type=%s)", idx,
                                     type(acts))
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

        if self.cfg.verbose:
            self.logger.info(
                f"[SAETrainer] Starting training run_id={self.run_id} epochs={self.cfg.epochs} batch_size={self.cfg.batch_size}")

        epoch_iter = tqdm(
            range(self.cfg.epochs),
            desc="Epochs",
            disable=not self.cfg.verbose,
            leave=True,
        )
        for epoch in epoch_iter:
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

                # Determine autocast dtype
                amp_dtype = self.cfg.amp_dtype or self.cfg.dtype
                # Begin forward/backward
                if (step % max(1, int(self.cfg.grad_accum_steps))) == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=self._use_amp):
                    recon, latents, recon_full, _lat_full = self.model(x)
                    mse = self.criterion(recon, x)
                    l1 = latents.abs().mean() if self.cfg.l1_lambda > 0 else torch.tensor(0.0, device=self.device)
                    loss = mse + self.cfg.l1_lambda * l1

                if self._scaler.is_enabled():
                    self._scaler.scale(loss / max(1, int(self.cfg.grad_accum_steps))).backward()
                else:
                    (loss / max(1, int(self.cfg.grad_accum_steps))).backward()

                do_step = ((step + 1) % max(1, int(self.cfg.grad_accum_steps))) == 0
                if do_step:
                    if self.cfg.project_decoder_grads:
                        with torch.no_grad():
                            self.model.project_grads_decode()

                    if self._scaler.is_enabled():
                        self._scaler.step(self.optimizer)
                        self._scaler.update()
                    else:
                        self.optimizer.step()

                    if self.cfg.renorm_decoder_every and (step + 1) % int(self.cfg.renorm_decoder_every) == 0:
                        with torch.no_grad():
                            self.model.scale_to_unit_norm()

                    # Optional: free CUDA cache to reduce fragmentation
                    if self.cfg.free_cuda_cache_every and (step + 1) % int(
                            self.cfg.free_cuda_cache_every) == 0 and torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

                cur_loss = float(loss.detach().cpu())
                cur_mse = float(mse.detach().cpu())
                cur_l1 = float(l1.detach().cpu())
                epoch_losses.append(cur_loss)
                epoch_mse.append(cur_mse)
                epoch_l1.append(cur_l1)
                step += 1

                if self.cfg.verbose:
                    try:
                        epoch_iter.set_postfix(
                            {"loss": f"{cur_loss:.5f}", "mse": f"{cur_mse:.5f}", "l1": f"{cur_l1:.5f}"})
                    except Exception:
                        pass

            # Aggregate and log
            if epoch_losses:
                mean_loss = sum(epoch_losses) / len(epoch_losses)
                mean_mse = sum(epoch_mse) / len(epoch_mse)
                mean_l1 = sum(epoch_l1) / len(epoch_l1)
            else:
                mean_loss = mean_mse = mean_l1 = float("nan")
                self.logger.warning(
                    "[SAETrainer] No mini-batches processed in epoch %d; check store/run_id or batch_size settings.",
                    epoch + 1)

            history["loss"].append(mean_loss)
            history["recon_mse"].append(mean_mse)
            history["l1"].append(mean_l1)

            # Update epoch-level progress bar with mean metrics
            if self.cfg.verbose:
                try:
                    epoch_iter.set_postfix({
                        "loss": f"{mean_loss:.5f}",
                        "mse": f"{mean_mse:.5f}",
                        "l1": f"{mean_l1:.5f}",
                    })
                except Exception:
                    pass

            # Optional checkpoint
            if self.cfg.checkpoint_dir is not None:
                ckpt_dir = Path(self.cfg.checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                # Use Autoencoder.save for consistency
                try:
                    self.model.save(f"epoch_{epoch + 1:03d}", path=str(ckpt_dir))
                except Exception:
                    # Fallback to torch.save for robustness
                    torch.save(self.model.state_dict(), str(ckpt_dir / f"epoch_{epoch + 1:03d}.pt"))

            # Log epoch metrics to Weights & Biases
            if self._wandb_run is not None:
                try:
                    self._wandb.log({
                        "epoch": epoch + 1,
                        "loss": mean_loss,
                        "recon_mse": mean_mse,
                        "l1": mean_l1,
                    })
                except Exception:
                    pass

        if self.cfg.verbose:
            self.logger.info("[SAETrainer] Completed training")

        # Close wandb run if opened
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass

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
        verbose: bool = False,
) -> list[float]:
    """Train an Autoencoder from Store activations. Returns loss history per epoch.

    This is a thin wrapper around SAETrainer to preserve a simple function API.
    """
    cfg = SAETrainingConfig(
        epochs=epochs,
        batch_size=batch_size or 1024,
        lr=learning_rate,
        l1_lambda=l1_lambda,
        verbose=verbose,
    )
    trainer = SAETrainer(sae, store, run_id, cfg)
    hist = trainer.train()
    return hist["loss"]
