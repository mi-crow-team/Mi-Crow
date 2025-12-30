from typing import Optional, Iterator, TYPE_CHECKING

import torch

from mi_crow.utils import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from mi_crow.store.store import Store


class StoreDataloader:
    """
    A reusable DataLoader-like class that can be iterated multiple times.

    This is needed because overcomplete's train_sae iterates over the dataloader
    once per epoch, so we need a dataloader that can be iterated multiple times.
    """

    def __init__(
            self,
            store: "Store",
            run_id: str,
            layer: str,
            key: str = "activations",
            batch_size: int = 32,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            max_batches: Optional[int] = None,
            logger_instance=None
    ):
        """
        Initialize StoreDataloader.

        Args:
            store: Store instance containing activations
            run_id: Run ID to iterate over
            layer: Layer signature to load activations from
            key: Tensor key to load (default: "activations")
            batch_size: Mini-batch size
            dtype: Optional dtype to cast activations to
            device: Optional device to move tensors to
            max_batches: Optional limit on number of batches per epoch
            logger_instance: Optional logger instance for debug messages
        """
        self.store = store
        self.run_id = run_id
        self.layer = layer
        self.key = key
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device
        self.max_batches = max_batches
        self.logger = logger_instance or logger

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Create a new iterator for each epoch.

        This allows the dataloader to be iterated multiple times,
        which is required for multiple epochs.
        """
        batches_yielded = 0
        
        # Get list of batch indices
        batch_indices = self.store.list_run_batches(self.run_id)
        
        for batch_index in batch_indices:
            if self.max_batches is not None and batches_yielded >= self.max_batches:
                break
            
            acts = None
            try:
                # Try to load from detector metadata first
                acts = self.store.get_detector_metadata_by_layer_by_key(
                    self.run_id,
                    batch_index,
                    self.layer,
                    self.key
                )
            except FileNotFoundError:
                # Fall back to traditional batch files
                try:
                    batch = self.store.get_run_batch(self.run_id, batch_index)
                    if isinstance(batch, dict) and self.key in batch:
                        acts = batch[self.key]
                    elif isinstance(batch, dict) and "activations" in batch:
                        # For backward compatibility, use "activations" if key not found
                        acts = batch["activations"]
                except Exception:
                    pass
            
            if acts is None:
                if self.logger.isEnabledFor(self.logger.level):
                    self.logger.debug(
                        f"Skipping batch {batch_index}: tensor not found "
                        f"(run_id={self.run_id}, layer={self.layer}, key={self.key})"
                    )
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
            
            # device handling
            if self.device is not None:
                acts = acts.to(self.device)

            # Yield mini-batches
            bs = max(1, int(self.batch_size))
            n = acts.shape[0]
            for start in range(0, n, bs):
                if self.max_batches is not None and batches_yielded >= self.max_batches:
                    return
                yield acts[start:start + bs]
                batches_yielded += 1
