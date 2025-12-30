"""Training utilities for SAE models using overcomplete's training functions."""

from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING
from datetime import datetime
import json
import logging
import gc

import torch

from mi_crow.store.store_dataloader import StoreDataloader
from mi_crow.utils import get_logger

if TYPE_CHECKING:
    from mi_crow.mechanistic.sae.sae import Sae
    from mi_crow.store.store import Store


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
    memory_efficient: bool = False  # Enable memory-efficient processing (moves tensors to CPU, clears cache)


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
            config: SaeTrainingConfig | None = None,
            training_run_id: str | None = None
    ) -> dict[str, Any]:
        self._ensure_overcomplete_available()
        cfg = config or SaeTrainingConfig()

        wandb_run = self._initialize_wandb(cfg, run_id)
        device = self._setup_device(cfg)
        optimizer = self._create_optimizer(cfg)
        criterion = self._create_criterion(cfg)
        dataloader = self._create_dataloader(store, run_id, layer_signature, cfg, device)
        monitoring = self._configure_logging(cfg, run_id)

        logs = self._run_training(cfg, dataloader, criterion, optimizer, device, monitoring)
        history = self._process_training_logs(logs, cfg)
        if cfg.memory_efficient:
            self._clear_memory()

        if wandb_run is not None:
            self._log_to_wandb(wandb_run, history, cfg)

        if cfg.verbose:
            self.logger.info(f"[SaeTrainer] Training completed, processing {len(history['loss'])} epochs of results")
            self.logger.info("[SaeTrainer] Completed training")

        if training_run_id is None:
            training_run_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if cfg.verbose:
            self.logger.info(f"[SaeTrainer] Saving training outputs to store/runs/{training_run_id}/")

        self._save_training_to_store(
            store=store,
            training_run_id=training_run_id,
            run_id=run_id,
            layer_signature=layer_signature,
            history=history,
            config=cfg
        )

        return {
            "history": history,
            "training_run_id": training_run_id
        }

    def _ensure_overcomplete_available(self) -> None:
        try:
            from overcomplete.sae.train import train_sae, train_sae_amp
        except ImportError:
            raise ImportError("overcomplete.sae.train module not available. Cannot use overcomplete training.")

    def _initialize_wandb(self, cfg: SaeTrainingConfig, run_id: str) -> Any:
        if not cfg.use_wandb:
            return None

        try:
            import wandb
            wandb_project = cfg.wandb_project or "sae-training"
            wandb_name = cfg.wandb_name or run_id
            wandb_mode = cfg.wandb_mode.lower() if cfg.wandb_mode else "online"

            return wandb.init(
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
            return None
        except Exception as e:
            self.logger.warning(f"[SaeTrainer] Unexpected error initializing wandb: {e}")
            self.logger.warning("[SaeTrainer] Continuing training without wandb logging")
            return None

    def _setup_device(self, cfg: SaeTrainingConfig) -> torch.device:
        device_str = str(cfg.device)
        device = torch.device(device_str)
        self.sae.sae_engine.to(device)
        
        if cfg.dtype is not None:
            try:
                self.sae.sae_engine.to(device, dtype=cfg.dtype)
            except (TypeError, AttributeError):
                self.sae.sae_engine.to(device)
        
        return device

    def _create_optimizer(self, cfg: SaeTrainingConfig) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.sae.sae_engine.parameters(), lr=cfg.lr)

    def _create_criterion(self, cfg: SaeTrainingConfig):
        def criterion(x: torch.Tensor, x_hat: torch.Tensor, z_pre: torch.Tensor, z: torch.Tensor,
                      dictionary: torch.Tensor) -> torch.Tensor:
            recon_loss = ((x_hat - x) ** 2).mean()
            l1_penalty = z.abs().mean() * cfg.l1_lambda if cfg.l1_lambda > 0 else torch.tensor(0.0, device=x.device)
            return recon_loss + l1_penalty
        return criterion

    def _create_dataloader(self, store: "Store", run_id: str, layer_signature: str | int, cfg: SaeTrainingConfig, device: torch.device) -> StoreDataloader:
        return StoreDataloader(
            store=store,
            run_id=run_id,
            layer=layer_signature,
            key="activations",
            batch_size=cfg.batch_size,
            dtype=cfg.dtype,
            device=device,
            max_batches=cfg.max_batches_per_epoch,
            logger_instance=self.logger
        )

    def _configure_logging(self, cfg: SaeTrainingConfig, run_id: str) -> int:
        monitoring = cfg.monitoring
        if cfg.verbose and monitoring < 2:
            monitoring = 2

        if cfg.verbose:
            device_str = str(cfg.device)
            self.logger.info(
                f"[SaeTrainer] Starting training run_id={run_id} epochs={cfg.epochs} batch_size={cfg.batch_size} "
                f"device={device_str} use_amp={cfg.use_amp}"
            )

        overcomplete_logger = logging.getLogger("overcomplete")
        if cfg.verbose:
            overcomplete_logger.setLevel(logging.INFO)
            if not overcomplete_logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
                overcomplete_logger.addHandler(handler)
            overcomplete_logger.propagate = True
        else:
            overcomplete_logger.setLevel(logging.WARNING)

        return monitoring

    def _run_training(self, cfg: SaeTrainingConfig, dataloader: StoreDataloader, criterion, optimizer: torch.optim.Optimizer,
                     device: torch.device, monitoring: int) -> dict[str, Any]:
        from overcomplete.sae.train import train_sae, train_sae_amp

        device_str = str(device)
        
        try:
            if cfg.use_amp and device.type in ("cuda", "cpu"):
                if cfg.verbose:
                    self.logger.info(f"[SaeTrainer] Using train_sae_amp with monitoring={monitoring}")
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
                if cfg.verbose:
                    self.logger.info(f"[SaeTrainer] Using train_sae with monitoring={monitoring}")
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
            
            if cfg.verbose:
                self.logger.info(
                    f"[SaeTrainer] Overcomplete training function returned, processing {len(logs.get('avg_loss', []))} epoch results...")
            
            return logs
        except Exception as e:
            self.logger.error(f"[SaeTrainer] Error during training: {e}")
            import traceback
            self.logger.error(f"[SaeTrainer] Traceback: {traceback.format_exc()}")
            raise

    def _process_training_logs(self, logs: dict[str, Any], cfg: SaeTrainingConfig) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {
            "loss": logs.get("avg_loss", []),
            "recon_mse": [],
            "l1": [],
            "r2": [],
            "l0": [],
            "dead_features_pct": [],
        }

        self._extract_r2_and_mse(history, logs)
        self._extract_sparsity_metrics(history, logs, cfg)

        return history

    def _extract_r2_and_mse(self, history: dict[str, list[float]], logs: dict[str, Any]) -> None:
        if "r2" in logs:
            history["r2"] = logs["r2"]
            history["recon_mse"] = [(1.0 - r2) for r2 in logs["r2"]]
        else:
            history["r2"] = [0.0] * len(history["loss"])

    def _extract_sparsity_metrics(self, history: dict[str, list[float]], logs: dict[str, Any], cfg: SaeTrainingConfig) -> None:
        memory_efficient = cfg.memory_efficient
        num_epochs = len(history["loss"])
        
        if "dead_features" in logs and isinstance(logs["dead_features"], list) and len(logs["dead_features"]) == num_epochs:
            history["dead_features_pct"] = logs["dead_features"]
        else:
            history["dead_features_pct"] = [0.0] * num_epochs
        
        history["l0"] = [0.0] * num_epochs
        
        if "z" in logs and logs["z"]:
            first_z = logs["z"][0] if len(logs["z"]) > 0 else None
            is_flat_list = isinstance(first_z, torch.Tensor) if first_z is not None else False
            
            if is_flat_list:
                batches_per_epoch = len(logs["z"]) // num_epochs if num_epochs > 0 else len(logs["z"])
                
                for epoch_idx in range(num_epochs):
                    start_idx = epoch_idx * batches_per_epoch
                    end_idx = start_idx + batches_per_epoch
                    epoch_z_tensors = logs["z"][start_idx:end_idx]
                    
                    if epoch_z_tensors:
                        l1_val = self._compute_l1(epoch_z_tensors, memory_efficient)
                        history["l1"].append(l1_val)
                    else:
                        history["l1"].append(0.0)
            else:
                n_latents = self._get_n_latents()
                slow_metrics_freq = cfg.wandb_slow_metrics_frequency if cfg.use_wandb else 1

                for epoch_idx, z_batch_list in enumerate(logs["z"]):
                    if isinstance(z_batch_list, list) and len(z_batch_list) > 0:
                        l1_val = self._compute_l1(z_batch_list, memory_efficient)
                        history["l1"].append(l1_val)

                        should_compute_slow = (epoch_idx % slow_metrics_freq == 0) or (epoch_idx == len(logs["z"]) - 1)

                        if should_compute_slow:
                            l0, dead_pct = self._compute_slow_metrics(z_batch_list, n_latents, memory_efficient)
                            if len(history["l0"]) <= epoch_idx:
                                history["l0"].extend([None] * (epoch_idx + 1 - len(history["l0"])))
                            history["l0"][epoch_idx] = l0
                            if len(history["dead_features_pct"]) <= epoch_idx:
                                history["dead_features_pct"].extend([None] * (epoch_idx + 1 - len(history["dead_features_pct"])))
                            history["dead_features_pct"][epoch_idx] = dead_pct
                        else:
                            if len(history["l0"]) <= epoch_idx:
                                history["l0"].extend([None] * (epoch_idx + 1 - len(history["l0"])))
                            history["l0"].append(None)
                            if len(history["dead_features_pct"]) <= epoch_idx:
                                history["dead_features_pct"].extend([None] * (epoch_idx + 1 - len(history["dead_features_pct"])))
                            history["dead_features_pct"].append(None)
                        
                        if memory_efficient:
                            del z_batch_list
                            if epoch_idx % 5 == 0:
                                self._clear_memory()
                    else:
                        history["l1"].append(0.0)
        elif "z_sparsity" in logs:
            history["l1"] = logs["z_sparsity"]
        else:
            history["l1"] = [0.0] * num_epochs
        
        if memory_efficient and "z" in logs:
            del logs["z"]
            self._clear_memory()

    def _get_n_latents(self) -> Optional[int]:
        if hasattr(self.sae, 'context') and hasattr(self.sae.context, 'n_latents'):
            return self.sae.context.n_latents
        return None

    def _clear_memory(self) -> None:
        """Clear GPU/MPS memory cache and run garbage collection."""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _compute_l1(self, z_batch_list: list[torch.Tensor], memory_efficient: bool = False) -> float:
        l1_vals = []
        for z in z_batch_list:
            if isinstance(z, torch.Tensor):
                if memory_efficient and z.device.type != "cpu":
                    z_cpu = z.cpu()
                    l1_vals.append(z_cpu.abs().mean().item())
                    del z
                else:
                    l1_vals.append(z.abs().mean().item())
        return sum(l1_vals) / len(l1_vals) if l1_vals else 0.0

    def _compute_slow_metrics(self, z_batch_list: list[torch.Tensor], n_latents: Optional[int], memory_efficient: bool = False) -> tuple[float, float]:
        l0_vals = []
        all_z_epoch = []
        
        for z in z_batch_list:
            if isinstance(z, torch.Tensor):
                if memory_efficient and z.device.type != "cpu":
                    z_cpu = z.cpu()
                    active = (z_cpu.abs() > 1e-6).float()
                    l0_vals.append(active.sum(dim=-1).mean().item())
                    all_z_epoch.append(z_cpu)
                    del z
                else:
                    active = (z.abs() > 1e-6).float()
                    l0_vals.append(active.sum(dim=-1).mean().item())
                    all_z_epoch.append(z)

        l0 = sum(l0_vals) / len(l0_vals) if l0_vals else 0.0

        if all_z_epoch and n_latents is not None:
            z_concatenated = torch.cat(all_z_epoch, dim=0)
            feature_activity = (z_concatenated.abs() > 1e-6).any(dim=0).float()
            dead_count = (feature_activity == 0).sum().item()
            dead_features_pct = dead_count / n_latents * 100.0 if n_latents > 0 else 0.0
            if memory_efficient:
                del z_concatenated, feature_activity
        else:
            dead_features_pct = 0.0

        if memory_efficient:
            del all_z_epoch
        return l0, dead_features_pct

    def _log_to_wandb(self, wandb_run: Any, history: dict[str, list[float]], cfg: SaeTrainingConfig) -> None:
        try:
            num_epochs = len(history["loss"])
            slow_metrics_freq = cfg.wandb_slow_metrics_frequency

            for epoch in range(1, num_epochs + 1):
                epoch_idx = epoch - 1
                should_log_slow = (epoch % slow_metrics_freq == 0) or (epoch == num_epochs)

                metrics = self._build_epoch_metrics(history, epoch, epoch_idx, cfg, should_log_slow)
                wandb_run.log(metrics)

            final_metrics = self._build_final_metrics(history, num_epochs)
            wandb_run.summary.update(final_metrics)

            if cfg.verbose:
                try:
                    url = wandb_run.url
                    self.logger.info(f"[SaeTrainer] Metrics logged to wandb: {url}")
                except (AttributeError, RuntimeError):
                    self.logger.info("[SaeTrainer] Metrics logged to wandb (offline mode)")
        except Exception as e:
            self.logger.warning(f"[SaeTrainer] Failed to log metrics to wandb: {e}")

    def _build_epoch_metrics(self, history: dict[str, list[float]], epoch: int, epoch_idx: int, 
                             cfg: SaeTrainingConfig, should_log_slow: bool) -> dict[str, Any]:
        metrics = {
            "epoch": epoch,
            "train/loss": history["loss"][epoch_idx] if epoch_idx < len(history["loss"]) else 0.0,
            "train/reconstruction_mse": history["recon_mse"][epoch_idx] if epoch_idx < len(history["recon_mse"]) else 0.0,
            "train/r2_score": history["r2"][epoch_idx] if epoch_idx < len(history["r2"]) else 0.0,
            "train/l1_penalty": history["l1"][epoch_idx] if epoch_idx < len(history["l1"]) else 0.0,
            "train/learning_rate": cfg.lr,
        }

        if should_log_slow:
            l0_val = self._get_metric_value(history["l0"], epoch_idx)
            dead_pct = self._get_metric_value(history["dead_features_pct"], epoch_idx)
            metrics["train/l0_sparsity"] = l0_val
            metrics["train/dead_features_pct"] = dead_pct

        return metrics

    def _get_metric_value(self, values: list[float | None], idx: int) -> float:
        if idx < len(values) and values[idx] is not None:
            return values[idx]
        return self._get_last_known_value(values, idx)

    def _get_last_known_value(self, values: list[float | None], idx: int) -> float:
        for i in range(idx, -1, -1):
            if i < len(values) and values[i] is not None:
                return values[i]
        return 0.0

    def _build_final_metrics(self, history: dict[str, list[float]], num_epochs: int) -> dict[str, Any]:
        final_l0 = self._get_last_known_value(history["l0"], len(history["l0"]) - 1) if history["l0"] else 0.0
        final_dead_pct = self._get_last_known_value(history["dead_features_pct"], len(history["dead_features_pct"]) - 1) if history["dead_features_pct"] else 0.0

        final_metrics = {
            "final/loss": history["loss"][-1] if history["loss"] else 0.0,
            "final/reconstruction_mse": history["recon_mse"][-1] if history["recon_mse"] else 0.0,
            "final/r2_score": history["r2"][-1] if history["r2"] else 0.0,
            "final/l1_penalty": history["l1"][-1] if history["l1"] else 0.0,
            "final/l0_sparsity": final_l0,
            "final/dead_features_pct": final_dead_pct,
            "training/num_epochs": num_epochs,
        }

        if history["loss"]:
            best_loss_idx = min(range(len(history["loss"])), key=lambda i: history["loss"][i])
            final_metrics["best/loss"] = history["loss"][best_loss_idx]
            final_metrics["best/loss_epoch"] = best_loss_idx + 1

        if history["r2"]:
            best_r2_idx = max(range(len(history["r2"])), key=lambda i: history["r2"][i])
            final_metrics["best/r2_score"] = history["r2"][best_r2_idx]
            final_metrics["best/r2_epoch"] = best_r2_idx + 1

        return final_metrics

    def _save_training_to_store(
            self,
            store: "Store",
            training_run_id: str,
            run_id: str,
            layer_signature: str | int,
            history: dict[str, list[float]],
            config: SaeTrainingConfig
    ) -> None:
        """Save training outputs (model, history, metadata) to store under training_run_id.
        
        Args:
            store: Store instance
            training_run_id: Training run ID to save under
            run_id: Original activation run ID used for training
            layer_signature: Layer signature used for training
            history: Training history dictionary
            config: Training configuration
        """
        try:
            run_path = store._run_key(training_run_id)
            run_path.mkdir(parents=True, exist_ok=True)

            model_path = run_path / "model.pt"
            history_path = run_path / "history.json"

            sae_state_dict = self.sae.sae_engine.state_dict()

            mi_crow_metadata = {
                "concepts_state": {
                    'multiplication': self.sae.concepts.multiplication.data.cpu().clone(),
                    'bias': self.sae.concepts.bias.data.cpu().clone(),
                },
                "n_latents": self.sae.context.n_latents,
                "n_inputs": self.sae.context.n_inputs,
                "device": self.sae.context.device,
                "layer_signature": self.sae.context.lm_layer_signature,
                "model_id": self.sae.context.model_id,
            }

            if hasattr(self.sae, 'k'):
                mi_crow_metadata["k"] = self.sae.k

            payload = {
                "sae_state_dict": sae_state_dict,
                "mi_crow_metadata": mi_crow_metadata,
            }

            torch.save(payload, model_path)

            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)

            training_metadata = {
                "training_run_id": training_run_id,
                "activation_run_id": run_id,
                "layer_signature": str(layer_signature),
                "model_id": self.sae.context.model_id if hasattr(self.sae.context, 'model_id') else None,
                "sae_type": type(self.sae).__name__,
                "training_config": {
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "lr": config.lr,
                    "l1_lambda": config.l1_lambda,
                    "device": str(config.device),
                    "dtype": str(config.dtype) if config.dtype else None,
                    "use_amp": config.use_amp,
                    "clip_grad": config.clip_grad,
                    "monitoring": config.monitoring,
                },
                "final_metrics": {
                    "loss": history["loss"][-1] if history["loss"] else None,
                    "r2": history["r2"][-1] if history["r2"] else None,
                    "recon_mse": history["recon_mse"][-1] if history["recon_mse"] else None,
                    "l1": history["l1"][-1] if history["l1"] else None,
                },
                "n_epochs": len(history["loss"]),
                "timestamp": datetime.now().isoformat(),
            }

            if history["l0"]:
                final_l0 = [x for x in history["l0"] if x is not None]
                if final_l0:
                    training_metadata["final_metrics"]["l0"] = final_l0[-1]

            if history["dead_features_pct"]:
                final_dead = [x for x in history["dead_features_pct"] if x is not None]
                if final_dead:
                    training_metadata["final_metrics"]["dead_features_pct"] = final_dead[-1]

            store.put_run_metadata(training_run_id, training_metadata)

            if config.verbose:
                self.logger.info(f"[SaeTrainer] Saved model to: {model_path}")
                self.logger.info(f"[SaeTrainer] Saved history to: {history_path}")
                self.logger.info(f"[SaeTrainer] Saved metadata to: runs/{training_run_id}/meta.json")

        except Exception as e:
            self.logger.warning(f"[SaeTrainer] Failed to save training outputs to store: {e}")
            if config.verbose:
                import traceback
                self.logger.warning(f"[SaeTrainer] Traceback: {traceback.format_exc()}")
