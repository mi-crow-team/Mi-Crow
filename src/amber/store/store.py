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

    def _run_key(self, run_id: str) -> Path:
        return self.base_path / self.runs_prefix / run_id

    def _run_batch_key(self, run_id: str, batch_index: int) -> Path:
        return self._run_key(run_id) / f"batch_{batch_index}"

    def _run_metadata_key(self, run_id: str) -> Path:
        return self._run_key(run_id) / "meta.json"

    @abc.abstractmethod
    def put_run_metadata(self, run_id: str, meta: Dict[str, Any]) -> str:
        """Persist metadata for a run (e.g., dataset/model identifiers).

        Implementations should store JSON at a stable location, e.g., runs/{run_id}/meta.json.
        Returns the key/path used for store.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_run_metadata(self, run_id: str) -> Dict[str, Any]:
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
    def get_detector_metadata(self, run_id: str, batch_index: int) -> tuple[
        Dict[str, Any], Dict[str, Dict[str, torch.Tensor]]]:
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
