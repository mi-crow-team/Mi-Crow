from __future__ import annotations

import abc
from pathlib import Path
from typing import Dict, Any, List, Iterator

import torch


TensorMetadata = Dict[str, Dict[str, torch.Tensor]]


class Store(abc.ABC):
    """Abstract store optimized for tensor batches grouped by run_id.

    This interface intentionally excludes generic bytes/JSON APIs.
    Implementations should focus on efficient safetensors-backed IO.
    
    The store organizes data hierarchically:
    - Runs: Top-level grouping by run_id
    - Batches: Within each run, data is organized by batch_index
    - Layers: Within each batch, tensors are organized by layer_signature
    - Keys: Within each layer, tensors are identified by key (e.g., "activations")
    """

    def __init__(
            self,
            base_path: Path | str = "",
            runs_prefix: str = "runs",
            dataset_prefix: str = "datasets",
            model_prefix: str = "models",
    ):
        """Initialize Store.
        
        Args:
            base_path: Base directory path for the store
            runs_prefix: Prefix for runs directory (default: "runs")
            dataset_prefix: Prefix for datasets directory (default: "datasets")
            model_prefix: Prefix for models directory (default: "models")
        """
        self.runs_prefix = runs_prefix
        self.dataset_prefix = dataset_prefix
        self.model_prefix = model_prefix
        self.base_path = Path(base_path)

    def _run_key(self, run_id: str) -> Path:
        """Get path for a run directory.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Path to run directory
        """
        return self.base_path / self.runs_prefix / run_id

    def _run_batch_key(self, run_id: str, batch_index: int) -> Path:
        """Get path for a batch directory within a run.
        
        Args:
            run_id: Run identifier
            batch_index: Batch index
            
        Returns:
            Path to batch directory
        """
        return self._run_key(run_id) / f"batch_{batch_index}"

    def _run_metadata_key(self, run_id: str) -> Path:
        """Get path for run metadata file.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Path to metadata JSON file
        """
        return self._run_key(run_id) / "meta.json"

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

    @abc.abstractmethod
    def put_run_metadata(self, run_id: str, meta: Dict[str, Any]) -> str:
        """Persist metadata for a run (e.g., dataset/model identifiers).

        Args:
            run_id: Run identifier
            meta: Metadata dictionary to save (must be JSON-serializable)
            
        Returns:
            String path/key where metadata was stored (e.g., "runs/{run_id}/meta.json")
            
        Raises:
            ValueError: If run_id is invalid or meta is not JSON-serializable
            OSError: If file system operations fail
            
        Note:
            Implementations should store JSON at a stable location, e.g., runs/{run_id}/meta.json.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_run_metadata(self, run_id: str) -> Dict[str, Any]:
        """Load metadata for a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Metadata dictionary, or empty dict if not found
            
        Raises:
            ValueError: If run_id is invalid
            json.JSONDecodeError: If metadata file exists but contains invalid JSON
        """
        raise NotImplementedError

    @abc.abstractmethod
    def put_detector_metadata(
            self,
            run_id: str,
            batch_index: int,
            metadata: Dict[str, Any],
            tensor_metadata: TensorMetadata
    ) -> str:
        """Save detector metadata with separate JSON and tensor store.
        
        Args:
            run_id: Run identifier
            batch_index: Batch index (must be non-negative)
            metadata: JSON-serializable metadata dictionary (aggregated from all detectors)
            tensor_metadata: Dictionary mapping layer_signature to dict of tensor_key -> tensor
                           (from all detectors)
            
        Returns:
            Full path key used for store (e.g., "runs/{run_id}/batch_{batch_index}")
            
        Raises:
            ValueError: If parameters are invalid or metadata is not JSON-serializable
            OSError: If file system operations fail
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_detector_metadata(
            self,
            run_id: str,
            batch_index: int
    ) -> tuple[Dict[str, Any], TensorMetadata]:
        """Load detector metadata with separate JSON and tensor store.
        
        Args:
            run_id: Run identifier
            batch_index: Batch index
            
        Returns:
            Tuple of (metadata dict, tensor_metadata dict). Returns empty dicts if not found.
            
        Raises:
            ValueError: If parameters are invalid or metadata format is invalid
            json.JSONDecodeError: If metadata file exists but contains invalid JSON
            OSError: If tensor files exist but cannot be loaded
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
            run_id: Run identifier
            batch_index: Batch index
            layer: Layer signature
            key: Tensor key (e.g., "activations")
            
        Returns:
            The requested tensor
            
        Raises:
            ValueError: If parameters are invalid
            FileNotFoundError: If the tensor doesn't exist
            OSError: If tensor file exists but cannot be loaded
        """
        raise NotImplementedError

    # --- Unified detector metadata for whole runs ---
    @abc.abstractmethod
    def put_run_detector_metadata(
            self,
            run_id: str,
            metadata: Dict[str, Any],
            tensor_metadata: TensorMetadata,
    ) -> str:
        """
        Save detector metadata for a whole run in a unified location.

        This differs from ``put_detector_metadata`` which organises data
        per-batch under ``runs/{run_id}/batch_{batch_index}``.

        ``put_run_detector_metadata`` instead stores everything under
        ``runs/{run_id}/detectors``. Implementations are expected to
        support being called multiple times for the same ``run_id`` and
        append / aggregate new metadata rather than overwrite it.

        Args:
            run_id: Run identifier
            metadata: JSON-serialisable metadata dictionary aggregated
                from all detectors for the current chunk / batch.
            tensor_metadata: Dictionary mapping layer_signature to dict
                of tensor_key -> tensor (from all detectors).

        Returns:
            String path/key where metadata was stored
            (e.g. ``runs/{run_id}/detectors``).

        Raises:
            ValueError: If parameters are invalid or metadata is not
                JSON‑serialisable.
            OSError: If file system operations fail.
        """
        raise NotImplementedError
