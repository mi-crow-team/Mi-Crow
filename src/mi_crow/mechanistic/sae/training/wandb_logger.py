"""Wandb logging utilities for SAE training."""

from typing import Any, Optional

from mi_crow.mechanistic.sae.sae_trainer import SaeTrainingConfig
from mi_crow.utils import get_logger

logger = get_logger(__name__)


class WandbLogger:
    """
    Handles wandb logging for SAE training.
    
    Encapsulates all wandb-related operations including initialization,
    metric logging, and summary updates.
    """

    def __init__(self, config: SaeTrainingConfig, run_id: str):
        """
        Initialize WandbLogger.
        
        Args:
            config: Training configuration
            run_id: Training run identifier
        """
        self.config = config
        self.run_id = run_id
        self.wandb_run: Optional[Any] = None
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize wandb run if enabled in config.
        
        Returns:
            True if wandb was successfully initialized, False otherwise
        """
        if not self.config.use_wandb:
            return False

        try:
            import wandb
        except ImportError:
            logger.warning("[WandbLogger] wandb not installed, skipping wandb logging")
            logger.warning("[WandbLogger] Install with: pip install wandb")
            return False

        try:
            wandb_project = self.config.wandb_project or "sae-training"
            wandb_name = self.config.wandb_name or self.run_id
            wandb_mode = self.config.wandb_mode.lower() if self.config.wandb_mode else "online"

            self.wandb_run = wandb.init(
                project=wandb_project,
                entity=self.config.wandb_entity,
                name=wandb_name,
                mode=wandb_mode,
                config=self._build_wandb_config(),
                tags=self.config.wandb_tags or [],
            )
            self._initialized = True
            return True
        except Exception as e:
            logger.warning(f"[WandbLogger] Unexpected error initializing wandb: {e}")
            logger.warning("[WandbLogger] Continuing training without wandb logging")
            return False

    def _build_wandb_config(self) -> dict[str, Any]:
        """Build wandb configuration dictionary."""
        return {
            "run_id": self.run_id,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "lr": self.config.lr,
            "l1_lambda": self.config.l1_lambda,
            "device": str(self.config.device),
            "dtype": str(self.config.dtype) if self.config.dtype else None,
            "use_amp": self.config.use_amp,
            "clip_grad": self.config.clip_grad,
            "max_batches_per_epoch": self.config.max_batches_per_epoch,
            **(self.config.wandb_config or {}),
        }

    def log_metrics(
            self,
            history: dict[str, list[float | None]],
            verbose: bool = False
    ) -> None:
        """
        Log training metrics to wandb.
        
        Args:
            history: Dictionary with training history (loss, r2, l1, l0, etc.)
            verbose: Whether to log verbose information
        """
        if not self._initialized or self.wandb_run is None:
            return

        try:
            num_epochs = len(history.get("loss", []))
            slow_metrics_freq = self.config.wandb_slow_metrics_frequency

            # Helper to get last known value for slow metrics
            def get_last_known_value(values: list[float | None], idx: int) -> float:
                """Get the last non-None value up to idx, or 0.0 if none found."""
                for i in range(idx, -1, -1):
                    if i < len(values) and values[i] is not None:
                        return values[i]
                return 0.0

            # Log metrics for each epoch
            for epoch in range(1, num_epochs + 1):
                epoch_idx = epoch - 1
                should_log_slow = (epoch % slow_metrics_freq == 0) or (epoch == num_epochs)

                metrics = self._build_epoch_metrics(history, epoch_idx, should_log_slow, get_last_known_value)
                self.wandb_run.log(metrics)

            # Log final summary metrics
            self._log_summary_metrics(history, get_last_known_value)

            if verbose:
                self._log_wandb_url()

        except Exception as e:
            logger.warning(f"[WandbLogger] Failed to log metrics to wandb: {e}")

    def _build_epoch_metrics(
            self,
            history: dict[str, list[float | None]],
            epoch_idx: int,
            should_log_slow: bool,
            get_last_known_value: Any
    ) -> dict[str, Any]:
        """Build metrics dictionary for a single epoch."""
        metrics = {
            "epoch": epoch_idx + 1,
            "train/loss": self._safe_get(history["loss"], epoch_idx, 0.0),
            "train/reconstruction_mse": self._safe_get(history["recon_mse"], epoch_idx, 0.0),
            "train/r2_score": self._safe_get(history["r2"], epoch_idx, 0.0),
            "train/l1_penalty": self._safe_get(history["l1"], epoch_idx, 0.0),
            "train/learning_rate": self.config.lr,
        }

        # Add slow metrics if computed this epoch
        if should_log_slow:
            l0_val = self._get_slow_metric(history["l0"], epoch_idx, get_last_known_value)
            dead_pct = self._get_slow_metric(history["dead_features_pct"], epoch_idx, get_last_known_value)
            metrics["train/l0_sparsity"] = l0_val
            metrics["train/dead_features_pct"] = dead_pct

        return metrics

    def _get_slow_metric(
            self,
            values: list[float | None],
            epoch_idx: int,
            get_last_known_value: Any
    ) -> float:
        """Get slow metric value, using last known value if current is None."""
        if epoch_idx < len(values) and values[epoch_idx] is not None:
            return values[epoch_idx]
        return get_last_known_value(values, epoch_idx)

    def _safe_get(self, values: list[float | None], idx: int, default: float) -> float:
        """Safely get value from list, returning default if out of bounds."""
        if idx < len(values) and values[idx] is not None:
            return values[idx]
        return default

    def _log_summary_metrics(
            self,
            history: dict[str, list[float | None]],
            get_last_known_value: Any
    ) -> None:
        """Log final summary metrics to wandb."""
        if self.wandb_run is None:
            return

        # Get last computed values for slow metrics
        final_l0 = get_last_known_value(history["l0"], len(history["l0"]) - 1) if history.get("l0") else 0.0
        final_dead_pct = get_last_known_value(
            history["dead_features_pct"],
            len(history["dead_features_pct"]) - 1
        ) if history.get("dead_features_pct") else 0.0

        final_metrics = {
            "final/loss": history["loss"][-1] if history.get("loss") else 0.0,
            "final/reconstruction_mse": history["recon_mse"][-1] if history.get("recon_mse") else 0.0,
            "final/r2_score": history["r2"][-1] if history.get("r2") else 0.0,
            "final/l1_penalty": history["l1"][-1] if history.get("l1") else 0.0,
            "final/l0_sparsity": final_l0,
            "final/dead_features_pct": final_dead_pct,
            "training/num_epochs": len(history.get("loss", [])),
        }

        # Add best metrics
        if history.get("loss"):
            best_loss_idx = min(range(len(history["loss"])), key=lambda i: history["loss"][i] or float('inf'))
            final_metrics["best/loss"] = history["loss"][best_loss_idx] or 0.0
            final_metrics["best/loss_epoch"] = best_loss_idx + 1

        if history.get("r2"):
            best_r2_idx = max(range(len(history["r2"])), key=lambda i: history["r2"][i] or -float('inf'))
            final_metrics["best/r2_score"] = history["r2"][best_r2_idx] or 0.0
            final_metrics["best/r2_epoch"] = best_r2_idx + 1

        self.wandb_run.summary.update(final_metrics)

    def _log_wandb_url(self) -> None:
        """Log wandb run URL if available."""
        if self.wandb_run is None:
            return

        try:
            url = self.wandb_run.url
            logger.info(f"[WandbLogger] Metrics logged to wandb: {url}")
        except (AttributeError, RuntimeError):
            # Offline mode or URL not available
            logger.info("[WandbLogger] Metrics logged to wandb (offline mode)")

