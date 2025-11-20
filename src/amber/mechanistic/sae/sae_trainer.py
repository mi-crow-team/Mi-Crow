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
    # Wandb configuration
    use_wandb: bool = False  # Enable wandb logging
    wandb_project: Optional[str] = None  # Wandb project name (defaults to "sae-training" if not set)
    wandb_entity: Optional[str] = None  # Wandb entity/team name
    wandb_name: Optional[str] = None  # Wandb run name (defaults to run_id if not set)
    wandb_tags: Optional[list[str]] = None  # Additional tags for wandb run
    wandb_config: Optional[dict[str, Any]] = None  # Additional config to log to wandb
    wandb_mode: str = "online"  # Wandb mode: "online", "offline", or "disabled"
    wandb_slow_metrics_frequency: int = 50  # Log slow metrics (L0, dead features) every N epochs (default: 50)


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
            layer_signature: str | int,
            config: SaeTrainingConfig | None = None
    ) -> dict[str, list[float]]:
        try:
            from overcomplete.sae.train import train_sae, train_sae_amp
        except ImportError:
            raise ImportError("overcomplete.sae.train module not available. Cannot use overcomplete training.")

        cfg = config or SaeTrainingConfig()

        # Initialize wandb if enabled
        wandb_run = None
        if cfg.use_wandb:
            try:
                import wandb
                wandb_project = cfg.wandb_project or "sae-training"
                wandb_name = cfg.wandb_name or run_id
                wandb_mode = cfg.wandb_mode.lower() if cfg.wandb_mode else "online"

                # Try to initialize wandb
                wandb_run = wandb.init(
                    project=wandb_project,
                    entity=cfg.wandb_entity,
                    name=wandb_name,
                    mode=wandb_mode,
                    config={
                        "run_id": run_id,
                        "epochs": cfg.epochs,
                        "batch_size": cfg.batch_size,
                        "lr": cfg.lr,
                        "l1_lambda": cfg.l1_lambda,
                        "device": str(cfg.device),
                        "dtype": str(cfg.dtype) if cfg.dtype else None,
                        "use_amp": cfg.use_amp,
                        "clip_grad": cfg.clip_grad,
                        "max_batches_per_epoch": cfg.max_batches_per_epoch,
                        **(cfg.wandb_config or {}),
                    },
                    tags=cfg.wandb_tags or [],
                )
            except ImportError:
                self.logger.warning("[SaeTrainer] wandb not installed, skipping wandb logging")
                self.logger.warning("[SaeTrainer] Install with: pip install wandb")
            except Exception as e:
                self.logger.warning(f"[SaeTrainer] Unexpected error initializing wandb: {e}")
                self.logger.warning("[SaeTrainer] Continuing training without wandb logging")

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
            layer=layer_signature,
            key="activations",
            batch_size=cfg.batch_size,
            dtype=cfg.dtype,
            max_batches=cfg.max_batches_per_epoch,
            logger_instance=self.logger
        )

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
            "l1": [],
            "r2": [],
            "l0": [],  # Number of active features (sparsity)
            "dead_features_pct": [],  # Percentage of dead features
        }

        # Extract R2 scores and compute reconstruction MSE
        if "r2" in logs:
            history["r2"] = logs["r2"]
            history["recon_mse"] = [(1.0 - r2) for r2 in logs["r2"]]
        else:
            history["r2"] = [0.0] * len(history["loss"])

        # Extract L1 sparsity (fast metric - computed every epoch)
        # L0 and dead features (slow metrics - computed less frequently)
        if "z" in logs and logs["z"]:
            n_latents = self.sae.context.n_latents if hasattr(self.sae, 'context') and hasattr(self.sae.context,
                                                                                               'n_latents') else None
            slow_metrics_freq = cfg.wandb_slow_metrics_frequency if cfg.use_wandb else 1

            for epoch_idx, z_batch_list in enumerate(logs["z"]):
                if isinstance(z_batch_list, list) and len(z_batch_list) > 0:
                    # Fast metric: Compute L1 (mean absolute value) - average across all batches
                    l1_vals = [z.abs().mean().item() for z in z_batch_list if isinstance(z, torch.Tensor)]
                    history["l1"].append(sum(l1_vals) / len(l1_vals) if l1_vals else 0.0)

                    # Slow metrics: Only compute when needed (every N epochs or last epoch)
                    should_compute_slow = (epoch_idx % slow_metrics_freq == 0) or (epoch_idx == len(logs["z"]) - 1)

                    if should_compute_slow:
                        # Compute L0 (number of active features) - average across all batches
                        l0_vals = []
                        # Concatenate all batches to compute dead features across entire epoch
                        all_z_epoch = []
                        for z in z_batch_list:
                            if isinstance(z, torch.Tensor):
                                # L0: average number of non-zero features per sample
                                active = (z.abs() > 1e-6).float()  # Threshold for "active"
                                l0_vals.append(active.sum(dim=-1).mean().item())
                                all_z_epoch.append(z)

                        history["l0"].append(sum(l0_vals) / len(l0_vals) if l0_vals else 0.0)

                        # Compute dead features percentage across all batches in epoch
                        if all_z_epoch and n_latents is not None:
                            # Concatenate all batches: [batch_size_1, n_latents], [batch_size_2, n_latents], ... -> [total_samples, n_latents]
                            z_concatenated = torch.cat(all_z_epoch, dim=0)  # [total_samples, n_latents]
                            # Check which features are active at least once across all samples
                            feature_activity = (z_concatenated.abs() > 1e-6).any(dim=0).float()  # [n_latents]
                            dead_count = (feature_activity == 0).sum().item()
                            dead_features_pct = dead_count / n_latents * 100.0 if n_latents > 0 else 0.0
                            history["dead_features_pct"].append(dead_features_pct)
                        else:
                            history["dead_features_pct"].append(0.0)
                    else:
                        # Don't compute slow metrics this epoch - use previous value or 0
                        # We'll interpolate or use last known value when logging
                        history["l0"].append(None)  # Mark as not computed
                        history["dead_features_pct"].append(None)  # Mark as not computed
                else:
                    history["l1"].append(0.0)
                    history["l0"].append(0.0)
                    history["dead_features_pct"].append(0.0)
        elif "z_sparsity" in logs:
            history["l1"] = logs["z_sparsity"]
            # Fill L0 and dead_features_pct with zeros if not available
            history["l0"] = [0.0] * len(history["loss"])
            history["dead_features_pct"] = [0.0] * len(history["loss"])
        else:
            history["l1"] = [0.0] * len(history["loss"])
            history["l0"] = [0.0] * len(history["loss"])
            history["dead_features_pct"] = [0.0] * len(history["loss"])

        # Log metrics to wandb if enabled
        if wandb_run is not None:
            try:
                # Log metrics for each epoch
                num_epochs = len(history["loss"])
                slow_metrics_freq = cfg.wandb_slow_metrics_frequency

                # Helper to get last known value for slow metrics
                def get_last_known_value(values, idx):
                    """Get the last non-None value up to idx, or 0.0 if none found."""
                    for i in range(idx, -1, -1):
                        if i < len(values) and values[i] is not None:
                            return values[i]
                    return 0.0

                for epoch in range(1, num_epochs + 1):
                    epoch_idx = epoch - 1
                    should_log_slow = (epoch % slow_metrics_freq == 0) or (epoch == num_epochs)

                    # Fast metrics (logged every epoch)
                    metrics = {
                        "epoch": epoch,
                        "train/loss": history["loss"][epoch_idx] if epoch_idx < len(history["loss"]) else 0.0,
                        "train/reconstruction_mse": history["recon_mse"][epoch_idx] if epoch_idx < len(
                            history["recon_mse"]) else 0.0,
                        "train/r2_score": history["r2"][epoch_idx] if epoch_idx < len(history["r2"]) else 0.0,
                        "train/l1_penalty": history["l1"][epoch_idx] if epoch_idx < len(history["l1"]) else 0.0,
                        "train/learning_rate": cfg.lr,
                    }

                    # Slow metrics (logged only when computed)
                    if should_log_slow:
                        l0_val = history["l0"][epoch_idx] if epoch_idx < len(history["l0"]) and history["l0"][
                            epoch_idx] is not None else get_last_known_value(history["l0"], epoch_idx)
                        dead_pct = history["dead_features_pct"][epoch_idx] if epoch_idx < len(
                            history["dead_features_pct"]) and history["dead_features_pct"][
                                                                                  epoch_idx] is not None else get_last_known_value(
                            history["dead_features_pct"], epoch_idx)
                        metrics["train/l0_sparsity"] = l0_val
                        metrics["train/dead_features_pct"] = dead_pct

                    wandb_run.log(metrics)

                # Log final metrics to summary
                # Get last computed values for slow metrics (handle None values)
                final_l0 = get_last_known_value(history["l0"], len(history["l0"]) - 1) if history["l0"] else 0.0
                final_dead_pct = get_last_known_value(history["dead_features_pct"],
                                                      len(history["dead_features_pct"]) - 1) if history[
                    "dead_features_pct"] else 0.0

                final_metrics = {
                    "final/loss": history["loss"][-1] if history["loss"] else 0.0,
                    "final/reconstruction_mse": history["recon_mse"][-1] if history["recon_mse"] else 0.0,
                    "final/r2_score": history["r2"][-1] if history["r2"] else 0.0,
                    "final/l1_penalty": history["l1"][-1] if history["l1"] else 0.0,
                    "final/l0_sparsity": final_l0,
                    "final/dead_features_pct": final_dead_pct,
                    "training/num_epochs": num_epochs,
                }

                # Add best metrics
                if history["loss"]:
                    best_loss_idx = min(range(len(history["loss"])), key=lambda i: history["loss"][i])
                    final_metrics["best/loss"] = history["loss"][best_loss_idx]
                    final_metrics["best/loss_epoch"] = best_loss_idx + 1

                if history["r2"]:
                    best_r2_idx = max(range(len(history["r2"])), key=lambda i: history["r2"][i])
                    final_metrics["best/r2_score"] = history["r2"][best_r2_idx]
                    final_metrics["best/r2_epoch"] = best_r2_idx + 1

                wandb_run.summary.update(final_metrics)

                if cfg.verbose:
                    try:
                        url = wandb_run.url
                        self.logger.info(f"[SaeTrainer] Metrics logged to wandb: {url}")
                    except (AttributeError, RuntimeError):
                        # Offline mode or URL not available
                        self.logger.info("[SaeTrainer] Metrics logged to wandb (offline mode)")
            except Exception as e:
                self.logger.warning(f"[SaeTrainer] Failed to log metrics to wandb: {e}")

        if cfg.verbose:
            self.logger.info("[SaeTrainer] Completed training")

        return history
