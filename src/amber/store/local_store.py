import json
from pathlib import Path
from typing import Dict, Any, List
import shutil

import torch

from amber.store.store import Store, TensorMetadata
import safetensors.torch as storch


class LocalStore(Store):
    """Local filesystem implementation of Store interface.
    
    Stores metadata as JSON files and tensors as safetensors files.
    Uses a directory structure organized by run_id and batch_index.
    """

    def __init__(
            self,
            base_path: Path | str = '',
            runs_prefix: str = "runs",
            dataset_prefix: str = "datasets",
            model_prefix: str = "models",
    ):
        """Initialize LocalStore.
        
        Args:
            base_path: Base directory path for the store
            runs_prefix: Prefix for runs directory
            dataset_prefix: Prefix for datasets directory
            model_prefix: Prefix for models directory
        """
        super().__init__(base_path, runs_prefix, dataset_prefix, model_prefix)

    def _full(self, key: str) -> Path:
        """Get full path for a key, creating parent directories if needed."""
        p = self.base_path / key
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def put_tensor(self, key: str, tensor: torch.Tensor) -> None:
        """Save a single tensor to the store."""
        path = self._full(key)
        storch.save_file({"tensor": tensor}, str(path))

    def get_tensor(self, key: str) -> torch.Tensor:
        """Load a single tensor from the store."""
        loaded = storch.load_file(str(self._full(key)))
        return loaded["tensor"]

    def _validate_run_id(self, run_id: str) -> None:
        """Validate run_id parameter.
        
        Args:
            run_id: Run identifier to validate
            
        Raises:
            ValueError: If run_id is empty or None
        """
        if not run_id or not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"run_id must be a non-empty string, got: {run_id!r}")

    def _validate_batch_index(self, batch_index: int) -> None:
        """Validate batch_index parameter.
        
        Args:
            batch_index: Batch index to validate
            
        Raises:
            ValueError: If batch_index is negative
        """
        if batch_index < 0:
            raise ValueError(f"batch_index must be non-negative, got: {batch_index}")

    def _validate_layer_key(self, layer: str, key: str) -> None:
        """Validate layer and key parameters.
        
        Args:
            layer: Layer signature to validate
            key: Tensor key to validate
            
        Raises:
            ValueError: If layer or key is empty
        """
        if not layer or not isinstance(layer, str) or not layer.strip():
            raise ValueError(f"layer must be a non-empty string, got: {layer!r}")
        if not key or not isinstance(key, str) or not key.strip():
            raise ValueError(f"key must be a non-empty string, got: {key!r}")

    def _ensure_directory(self, path: Path) -> None:
        """Ensure directory exists, creating it if necessary.
        
        Args:
            path: Directory path to ensure exists
        """
        path.mkdir(parents=True, exist_ok=True)

    def put_run_batch(self, run_id: str, batch_index: int,
                      tensors: List[torch.Tensor] | Dict[str, torch.Tensor]) -> str:
        """Save a batch of tensors for a run.
        
        Args:
            run_id: Run identifier
            batch_index: Batch index
            tensors: List or dict of tensors to save
            
        Returns:
            Path key where batch was saved
        """
        if isinstance(tensors, dict):
            to_save = tensors
        elif isinstance(tensors, list):
            if len(tensors) == 0:
                to_save = {"_empty_list": torch.tensor([])}
            else:
                to_save = {f"item_{i}": t for i, t in enumerate(tensors)}
        else:
            to_save = {}
        # Use the batch key path but append .safetensors extension
        batch_key = self._run_batch_key(run_id, batch_index)
        batch_path = self.base_path / f"{self.runs_prefix}/{run_id}/batch_{batch_index:06d}.safetensors"
        self._ensure_directory(batch_path.parent)
        storch.save_file(to_save, str(batch_path))
        return f"{self.runs_prefix}/{run_id}/batch_{batch_index:06d}.safetensors"

    def get_run_batch(self, run_id: str, batch_index: int) -> List[torch.Tensor] | Dict[
        str, torch.Tensor]:
        """Load a batch of tensors for a run.
        
        Args:
            run_id: Run identifier
            batch_index: Batch index
            
        Returns:
            List or dict of tensors
            
        Raises:
            FileNotFoundError: If batch doesn't exist
        """
        # Try traditional safetensors file first
        batch_path = self.base_path / f"{self.runs_prefix}/{run_id}/batch_{batch_index:06d}.safetensors"
        if batch_path.exists():
            loaded = storch.load_file(str(batch_path))
            keys = list(loaded.keys())
            if keys == ["_empty_list"]:
                return []
            if keys and all(k.startswith("item_") for k in keys):
                try:
                    items = sorted(((int(k.split("_", 1)[1]), v) for k, v in loaded.items()), key=lambda x: x[0])
                    if [i for i, _ in items] == list(range(len(items))):
                        return [v for _, v in items]
                except Exception:
                    pass
            return loaded

        # Try detector format (directory structure)
        detector_base = self.base_path / self.runs_prefix / run_id / f"batch_{batch_index}"
        if detector_base.exists():
            result: Dict[str, torch.Tensor] = {}

            layer_dirs = [d for d in detector_base.iterdir() if d.is_dir()]
            for layer_dir in layer_dirs:
                activations_path = layer_dir / "activations.safetensors"
                if activations_path.exists():
                    try:
                        loaded_tensor = storch.load_file(str(activations_path))["tensor"]
                        # Use layer_signature as key, or "activations" if only one layer
                        layer_sig = layer_dir.name
                        if len(layer_dirs) == 1:
                            # Only one layer, use simple "activations" key for compatibility
                            result["activations"] = loaded_tensor
                        else:
                            # Multiple layers, use layer-specific key
                            result[f"activations_{layer_sig}"] = loaded_tensor
                    except Exception:
                        pass

            if result:
                return result

        # If neither exists, raise FileNotFoundError
        raise FileNotFoundError(f"Batch {batch_index} not found for run {run_id}")

    def list_run_batches(self, run_id: str) -> List[int]:
        """List all batch indices for a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Sorted list of batch indices
        """
        base = self.base_path / self.runs_prefix / run_id
        if not base.exists():
            return []
        out: set[int] = set()
        
        # Check for traditional batch files
        for p in sorted(base.glob("batch_*.safetensors")):
            name = p.name
            try:
                idx = int(name[len("batch_"): len("batch_") + 6])
                out.add(idx)
            except Exception:
                continue
        
        # Also check for detector metadata batch directories
        for p in sorted(base.glob("batch_*")):
            if p.is_dir():
                name = p.name
                try:
                    idx = int(name[len("batch_"):])
                    out.add(idx)
                except Exception:
                    continue
        
        return sorted(list(out))

    def delete_run(self, run_id: str) -> None:
        """Delete all batches for a run.
        
        Args:
            run_id: Run identifier
        """
        base = self.base_path / self.runs_prefix / run_id
        if not base.exists():
            return
        for p in base.glob("batch_*.safetensors"):
            if p.is_file():
                p.unlink()
        # Also delete detector metadata directories
        for p in base.glob("batch_*"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        # Delete metadata file
        metadata_path = self._run_metadata_key(run_id)
        if metadata_path.exists():
            metadata_path.unlink()

    def put_run_metadata(self, run_id: str, meta: Dict[str, Any]) -> str:
        """Persist metadata for a run.
        
        Args:
            run_id: Run identifier
            meta: Metadata dictionary to save (must be JSON-serializable)
            
        Returns:
            String path where metadata was saved (e.g., "runs/{run_id}/meta.json")
            
        Raises:
            ValueError: If run_id is empty or meta is not JSON-serializable
            OSError: If file system operations fail
        """
        self._validate_run_id(run_id)
        
        metadata_path = self._run_metadata_key(run_id)
        self._ensure_directory(metadata_path.parent)
        
        try:
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Metadata is not JSON-serializable for run_id={run_id!r}. "
                f"Error: {e}"
            ) from e
        except OSError as e:
            raise OSError(
                f"Failed to write metadata file at {metadata_path} for run_id={run_id!r}. "
                f"Error: {e}"
            ) from e
        
        return f"{self.runs_prefix}/{run_id}/meta.json"

    def get_run_metadata(self, run_id: str) -> Dict[str, Any]:
        """Load metadata for a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Metadata dictionary, or empty dict if not found
            
        Raises:
            ValueError: If run_id is empty
            json.JSONDecodeError: If metadata file exists but contains invalid JSON
        """
        self._validate_run_id(run_id)
        
        metadata_path = self._run_metadata_key(run_id)
        if not metadata_path.exists():
            return {}
        
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in metadata file at {metadata_path} for run_id={run_id!r}",
                e.doc,
                e.pos
            ) from e

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
            batch_index: Batch index
            metadata: JSON-serializable metadata dictionary (aggregated from all detectors)
            tensor_metadata: Dictionary mapping layer_signature to dict of tensor_key -> tensor
            
        Returns:
            Full path key used for store (e.g., "runs/{run_id}/batch_{batch_index}")
            
        Raises:
            ValueError: If parameters are invalid or metadata is not JSON-serializable
            OSError: If file system operations fail
        """
        self._validate_run_id(run_id)
        self._validate_batch_index(batch_index)
        
        batch_dir = self._run_batch_key(run_id, batch_index)
        self._ensure_directory(batch_dir)

        tensor_metadata_names = {
            layer_signature: list(detector_tensors.keys())
            for layer_signature, detector_tensors in tensor_metadata.items()
            if detector_tensors
        }
        metadata_with_tensor_names = {
            **metadata,
            "_tensor_metadata_names": tensor_metadata_names
        }

        detector_metadata_path = batch_dir / "metadata.json"
        try:
            with detector_metadata_path.open("w", encoding="utf-8") as f:
                json.dump(metadata_with_tensor_names, f, ensure_ascii=False, indent=2)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Metadata is not JSON-serializable for run_id={run_id!r}, "
                f"batch_index={batch_index}. Error: {e}"
            ) from e
        except OSError as e:
            raise OSError(
                f"Failed to write metadata file at {detector_metadata_path} for "
                f"run_id={run_id!r}, batch_index={batch_index}. Error: {e}"
            ) from e

        # Process each detector's tensor metadata (keyed by layer_signature)
        for layer_signature, detector_tensors in tensor_metadata.items():
            if not detector_tensors:
                continue

            layer_dir = batch_dir / layer_signature
            self._ensure_directory(layer_dir)

            # Save each tensor key (e.g., "activations") as a separate safetensors file
            for tensor_key, tensor in detector_tensors.items():
                tensor_filename = f"{tensor_key}.safetensors"
                tensor_path = layer_dir / tensor_filename
                try:
                    storch.save_file({"tensor": tensor}, str(tensor_path))
                except Exception as e:
                    raise OSError(
                        f"Failed to save tensor at {tensor_path} for run_id={run_id!r}, "
                        f"batch_index={batch_index}, layer={layer_signature!r}, "
                        f"key={tensor_key!r}. Error: {e}"
                    ) from e

        return f"{self.runs_prefix}/{run_id}/batch_{batch_index}"

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
        """
        self._validate_run_id(run_id)
        self._validate_batch_index(batch_index)
        
        batch_dir = self._run_batch_key(run_id, batch_index)
        metadata_path = batch_dir / "metadata.json"
        
        # Return empty dicts if metadata file doesn't exist (per abstract contract)
        if not metadata_path.exists():
            return {}, {}
        
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in metadata file at {metadata_path} for "
                f"run_id={run_id!r}, batch_index={batch_index}",
                e.doc,
                e.pos
            ) from e

        tensor_metadata: Dict[str, Dict[str, torch.Tensor]] = {}
        tensor_metadata_names = metadata.pop("_tensor_metadata_names", None)

        if tensor_metadata_names is not None:
            for layer_signature, tensor_keys in tensor_metadata_names.items():
                layer_dir = batch_dir / layer_signature
                detector_tensors: Dict[str, torch.Tensor] = {}
                for tensor_key in tensor_keys:
                    tensor_filename = f"{tensor_key}.safetensors"
                    tensor_path = layer_dir / tensor_filename
                    if tensor_path.exists():
                        try:
                            detector_tensors[tensor_key] = storch.load_file(str(tensor_path))["tensor"]
                        except Exception as e:
                            raise OSError(
                                f"Failed to load tensor at {tensor_path} for "
                                f"run_id={run_id!r}, batch_index={batch_index}, "
                                f"layer={layer_signature!r}, key={tensor_key!r}. Error: {e}"
                            ) from e
                if detector_tensors:
                    tensor_metadata[layer_signature] = detector_tensors
        else:
            raise ValueError(
                f"Field '_tensor_metadata_names' not found in detector metadata at "
                f"{metadata_path} for run_id={run_id!r}, batch_index={batch_index}. "
                f"Cannot retrieve tensors."
            )

        return metadata, tensor_metadata

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
            FileNotFoundError: If the tensor file doesn't exist
            OSError: If tensor file exists but cannot be loaded
        """
        self._validate_run_id(run_id)
        self._validate_batch_index(batch_index)
        self._validate_layer_key(layer, key)
        
        batch_dir = self._run_batch_key(run_id, batch_index)
        layer_dir = batch_dir / layer
        tensor_path = layer_dir / f"{key}.safetensors"
        
        if not tensor_path.exists():
            raise FileNotFoundError(
                f"Tensor not found at {tensor_path} for run_id={run_id!r}, "
                f"batch_index={batch_index}, layer={layer!r}, key={key!r}"
            )
        
        try:
            return storch.load_file(str(tensor_path))["tensor"]
        except Exception as e:
            raise OSError(
                f"Failed to load tensor at {tensor_path} for run_id={run_id!r}, "
                f"batch_index={batch_index}, layer={layer!r}, key={key!r}. Error: {e}"
            ) from e
