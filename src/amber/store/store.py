from __future__ import annotations

import abc
from pathlib import Path
from typing import Iterator, List, Dict, Any
import torch


class Store(abc.ABC):
    """Abstract store optimized for tensor batches grouped by run_id.

    This interface intentionally excludes generic bytes/JSON APIs.
    Implementations should focus on efficient safetensors-backed IO.
    """

    def __init__(
            self,
            base_path: Path | str = "",
            runs_prefix: str = "runs",
            dataset_prefix: str = "datasets",
            model_prefix: str = "models",
    ):
        self.runs_prefix = runs_prefix
        self.dataset_prefix = dataset_prefix
        self.model_prefix = model_prefix
        self.base_path = Path(base_path)

    # --- Run-oriented batch APIs ---
    def _run_batch_key(self, run_id: str, batch_index: int) -> str:
        return f"{self.runs_prefix}/{run_id}/batch_{batch_index:06d}.safetensors"

    @abc.abstractmethod
    def put_run_batch(self, run_id: str, batch_index: int,
                      tensors: List[torch.Tensor] | Dict[str, torch.Tensor]) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_run_batch(self, run_id: str, batch_index: int) -> List[torch.Tensor] | Dict[
        str, torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def list_run_batches(self, run_id: str) -> List[int]:
        raise NotImplementedError

    def iter_run_batches(self, run_id: str) -> Iterator[List[torch.Tensor] | Dict[str, torch.Tensor]]:
        for idx in self.list_run_batches(run_id):
            yield self.get_run_batch(run_id, idx)

    def iter_run_batch_range(
            self,
            run_id: str,
            *,
            start: int = 0,
            stop: int | None = None,
            step: int = 1,
    ) -> Iterator[List[torch.Tensor] | Dict[str, torch.Tensor]]:
        """Iterate run batches for indices in range(start, stop, step).

        If stop is None, it will be set to max(list_run_batches(run_id)) + 1 (or 0 if none).
        Raises ValueError if step == 0 or start < 0.
        """
        if step == 0:
            raise ValueError("step must not be 0")
        if start < 0:
            raise ValueError("start must be >= 0")
        indices = self.list_run_batches(run_id)
        if not indices:
            return
        max_idx = max(indices)
        if stop is None:
            stop = max_idx + 1
        for idx in range(start, stop, step):
            try:
                yield self.get_run_batch(run_id, idx)
            except FileNotFoundError:
                continue

    @abc.abstractmethod
    def delete_run(self, run_id: str) -> None:
        raise NotImplementedError

    # --- Run metadata (optional helpers) ---
    @abc.abstractmethod
    def put_run_meta(self, run_id: str, meta: Dict[str, Any]) -> str:
        """Persist metadata for a run (e.g., dataset/model identifiers).

        Implementations should store JSON at a stable location, e.g., runs/{run_id}/meta.json.
        Returns the key/path used for store.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_run_meta(self, run_id: str) -> Dict[str, Any]:
        """Load metadata for a run. Should return an empty dict if missing."""
        raise NotImplementedError

    @abc.abstractmethod
    def put_detector_metadata(
            self,
            run_id: str,
            batch_index: int,
            metadata: Dict[str, Any],
            tensor_metadata: Dict[str, Dict[str, torch.Tensor]]
    ) -> str:
        """Save detector metadata with separate JSON and tensor store.
        
        Args:
            run_id: Run ID
            batch_index: Batch index
            metadata: JSON-serializable metadata dictionary (aggregated from all detectors)
            tensor_metadata: Dictionary mapping layer_signature to dict of tensor_key -> tensor (from all detectors)
            
        Returns:
            Full path key used for store (e.g., "runs/run_123/batch_0")
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_detector_metadata_by_layer_by_key(
            self,
            run_id: str,
            batch_index: int,
            layer: str,
            key: str
    ) -> torch.Tensor:
        """Get a specific tensor from detector metadata by layer and key.
        
        Args:
            run_id: Run ID
            batch_index: Batch index
            layer: Layer signature
            key: Tensor key (e.g., "activations")
            
        Returns:
            The requested tensor
            
        Raises:
            FileNotFoundError: If the tensor doesn't exist
        """
        raise NotImplementedError
