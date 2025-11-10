"""Training utilities for SAE models using overcomplete's training functions."""

from dataclasses import dataclass
from typing import Any, Iterator, Optional, TYPE_CHECKING

import torch

from amber.utils import get_logger

if TYPE_CHECKING:
    from amber.mechanistic.autoencoder.sae import Sae
    from amber.store import Store

logger = get_logger(__name__)


class ReusableStoreDataLoader:
    """
    A reusable DataLoader-like class that can be iterated multiple times.
    
    This is needed because overcomplete's train_sae iterates over the dataloader
    once per epoch, so we need a dataloader that can be iterated multiple times.
    """
    
    def __init__(
            self,
            store: "Store",
            run_id: str,
            batch_size: int,
            dtype: Optional[torch.dtype] = None,
            max_batches: Optional[int] = None,
            logger_instance = None
    ):
        """
        Initialize ReusableStoreDataLoader.
        
        Args:
            store: Store instance containing activations
            run_id: Run ID to iterate over
            batch_size: Mini-batch size
            dtype: Optional dtype to cast activations to
            max_batches: Optional limit on number of batches per epoch
            logger_instance: Optional logger instance for debug messages
        """
        self.store = store
        self.run_id = run_id
        self.batch_size = batch_size
        self.dtype = dtype
        self.max_batches = max_batches
        self.logger = logger_instance or logger
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Create a new iterator for each epoch.
        
        This allows the dataloader to be iterated multiple times,
        which is required for multiple epochs.
        """
        batches_yielded = 0
        idx = 0
        for batch in self.store.iter_run_batches(self.run_id):
            idx += 1
            if self.max_batches is not None and batches_yielded >= self.max_batches:
                break
                
            if not isinstance(batch, dict) or "activations" not in batch:
                if self.logger.isEnabledFor(self.logger.level):
                    self.logger.debug(f"Skipping non-dict or missing 'activations' in batch #{idx}")
                continue
            acts = batch["activations"]
            if not isinstance(acts, torch.Tensor):
                if self.logger.isEnabledFor(self.logger.level):
                    self.logger.debug(f"Skipping non-tensor 'activations' in batch #{idx} (type={type(acts)})")
                continue
            # Ensure 2D [N, D]
            if acts.dim() > 2:
                d = acts.shape[-1]
                acts = acts.view(-1, d)
            elif acts.dim() == 1:
                acts = acts.view(1, -1)
            # dtype handling
            if self.dtype is not None:
                acts = acts.to(self.dtype)

            # Yield mini-batches
            bs = max(1, int(self.batch_size))
            n = acts.shape[0]
            for start in range(0, n, bs):
                if self.max_batches is not None and batches_yielded >= self.max_batches:
                    return
                yield acts[start:start + bs]
                batches_yielded += 1


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
        def criterion(x: torch.Tensor, x_hat: torch.Tensor, z_pre: torch.Tensor, z: torch.Tensor, dictionary: torch.Tensor) -> torch.Tensor:
            """Loss function compatible with overcomplete's train_sae."""
            # Reconstruction loss (MSE)
            recon_loss = ((x_hat - x) ** 2).mean()
            
            # L1 sparsity penalty if configured
            l1_penalty = z.abs().mean() * cfg.l1_lambda if cfg.l1_lambda > 0 else torch.tensor(0.0, device=x.device)
            
            return recon_loss + l1_penalty
        
        # Create reusable DataLoader from Store
        # Overcomplete's extract_input can handle tensors directly, tuples, lists, or dicts
        # We need a reusable dataloader that can be iterated multiple times (once per epoch)
        dataloader = ReusableStoreDataLoader(
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
        
        # Use overcomplete's training function
        # The dataloader is reusable and can be iterated multiple times (once per epoch)
        if cfg.use_amp and device.type in ("cuda", "cpu"):
            # Use train_sae_amp for mixed precision training
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
        
        # Convert overcomplete logs format to our history format
        history: dict[str, list[float]] = {
            "loss": logs.get("avg_loss", []),
            "recon_mse": [],  # We'll compute from R2 if available
            "l1": []  # L1 sparsity (we'll compute from z if available, or use z_sparsity as proxy)
        }
        
        # Convert R2 scores to MSE-like metric (1 - R2 gives error)
        if "r2" in logs:
            history["recon_mse"] = [(1.0 - r2) for r2 in logs["r2"]]
        
        # Compute L1 from stored z values if available (monitoring > 1), otherwise use z_sparsity as proxy
        if "z" in logs and logs["z"]:
            # z is a list of tensors, compute mean L1 across all stored z values per epoch
            for z_batch_list in logs["z"]:
                if isinstance(z_batch_list, list) and len(z_batch_list) > 0:
                    # Take mean L1 across all stored z batches for this epoch
                    l1_vals = [z.abs().mean().item() for z in z_batch_list if isinstance(z, torch.Tensor)]
                    history["l1"].append(sum(l1_vals) / len(l1_vals) if l1_vals else 0.0)
                else:
                    history["l1"].append(0.0)
        elif "z_sparsity" in logs:
            # Use L0 sparsity as a proxy for L1 (normalized by number of features)
            # z_sparsity is the sum of L0, so we approximate L1
            history["l1"] = logs["z_sparsity"]
        else:
            # Fill with zeros if no sparsity info available
            history["l1"] = [0.0] * len(history["loss"])
        
        if cfg.verbose:
            self.logger.info("[SaeTrainer] Completed training")
        
        return history

