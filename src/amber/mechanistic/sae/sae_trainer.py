"""Training utilities for SAE models using overcomplete's training functions."""

from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import torch

from amber.store.store_dataloader import StoreDataloader
from amber.utils import get_logger

if TYPE_CHECKING:
    from amber.mechanistic.sae.sae import Sae
    from amber.store.store import Store

logger = get_logger(__name__)


@dataclass
class SaeTrainingConfig:
    """Configuration for SAE training (compatible with overcomplete.train_sae)."""
    epochs: int = 1
    batch_size: int = 1024
    lr: float = 1e-3
    l1_lambda: float = 0.0
    device: str | torch.device = "cpu"
    dtype: Optional[torch.dtype] = None
    max_batches_per_epoch: Optional[int] = None
    verbose: bool = False
    use_amp: bool = True
    amp_dtype: Optional[torch.dtype] = None
    grad_accum_steps: int = 1
    clip_grad: float = 1.0  # Gradient clipping (overcomplete parameter)
    monitoring: int = 1  # 0=silent, 1=basic, 2=detailed (overcomplete parameter)
    scheduler: Optional[Any] = None  # Learning rate scheduler (overcomplete parameter)
    max_nan_fallbacks: int = 5  # For train_sae_amp (overcomplete parameter)


class SaeTrainer:
    """
    Composite trainer class for SAE models using overcomplete's training functions.
    
    This trainer handles training of any SAE that has a sae_engine attribute
    compatible with overcomplete's train_sae functions.
    """

    def __init__(self, sae: "Sae") -> None:
        """
        Initialize SaeTrainer.
        
        Args:
            sae: The SAE instance to train
        """
        self.sae = sae
        self.logger = get_logger(__name__)

    def train(
            self,
            store: "Store",
            run_id: str,
            config: SaeTrainingConfig | None = None
    ) -> dict[str, list[float]]:
        """
        Train SAE using activations from a Store.
        
        This method uses overcomplete's train_sae or train_sae_amp functions
        to train the underlying overcomplete SAE engine.
        
        Args:
            store: Store instance containing activations
            run_id: Run ID to train on
            config: Training configuration
            
        Returns:
            Dictionary with training history (converted from overcomplete logs format)
        """
        try:
            from overcomplete.sae.train import train_sae, train_sae_amp
        except ImportError:
            raise ImportError("overcomplete.sae.train module not available. Cannot use overcomplete training.")

        cfg = config or SaeTrainingConfig()

        # Set up device
        device_str = str(cfg.device)
        device = torch.device(device_str)
        self.sae.sae_engine.to(device)
        if cfg.dtype is not None:
            try:
                self.sae.sae_engine.to(device, dtype=cfg.dtype)
            except (TypeError, AttributeError):
                self.sae.sae_engine.to(device)

        # Set up optimizer - train sae_engine parameters
        optimizer = torch.optim.AdamW(self.sae.sae_engine.parameters(), lr=cfg.lr)

        # Create criterion function matching overcomplete's signature: criterion(x, x_hat, z_pre, z, dictionary)
        def criterion(x: torch.Tensor, x_hat: torch.Tensor, z_pre: torch.Tensor, z: torch.Tensor,
                      dictionary: torch.Tensor) -> torch.Tensor:
            """Loss function compatible with overcomplete's train_sae."""
            # Reconstruction loss (MSE)
            recon_loss = ((x_hat - x) ** 2).mean()

            # L1 sparsity penalty if configured
            l1_penalty = z.abs().mean() * cfg.l1_lambda if cfg.l1_lambda > 0 else torch.tensor(0.0, device=x.device)

            return recon_loss + l1_penalty

        dataloader = StoreDataloader(
            store=store,
            run_id=run_id,
            batch_size=cfg.batch_size,
            dtype=cfg.dtype,
            max_batches=cfg.max_batches_per_epoch,
            logger_instance=self.logger
        )

        # Set monitoring level - use config value, but override with 2 if verbose is True
        monitoring = cfg.monitoring
        if cfg.verbose and monitoring < 2:
            monitoring = 2

        if cfg.verbose:
            self.logger.info(
                f"[SaeTrainer] Starting training run_id={run_id} epochs={cfg.epochs} batch_size={cfg.batch_size} "
                f"device={device_str} use_amp={cfg.use_amp}"
            )

        if cfg.use_amp and device.type in ("cuda", "cpu"):
            logs = train_sae_amp(
                model=self.sae.sae_engine,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=cfg.scheduler,
                nb_epochs=cfg.epochs,
                clip_grad=cfg.clip_grad,
                monitoring=monitoring,
                device=device_str,
                max_nan_fallbacks=cfg.max_nan_fallbacks
            )
        else:
            # Use train_sae for standard training
            logs = train_sae(
                model=self.sae.sae_engine,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=cfg.scheduler,
                nb_epochs=cfg.epochs,
                clip_grad=cfg.clip_grad,
                monitoring=monitoring,
                device=device_str
            )

        history: dict[str, list[float]] = {
            "loss": logs.get("avg_loss", []),
            "recon_mse": [],
            "l1": []
        }

        if "r2" in logs:
            history["recon_mse"] = [(1.0 - r2) for r2 in logs["r2"]]

        if "z" in logs and logs["z"]:
            for z_batch_list in logs["z"]:
                if isinstance(z_batch_list, list) and len(z_batch_list) > 0:
                    l1_vals = [z.abs().mean().item() for z in z_batch_list if isinstance(z, torch.Tensor)]
                    history["l1"].append(sum(l1_vals) / len(l1_vals) if l1_vals else 0.0)
                else:
                    history["l1"].append(0.0)
        elif "z_sparsity" in logs:
            history["l1"] = logs["z_sparsity"]
        else:
            history["l1"] = [0.0] * len(history["loss"])

        if cfg.verbose:
            self.logger.info("[SaeTrainer] Completed training")

        return history
