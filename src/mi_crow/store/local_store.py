import json
from pathlib import Path
from typing import Dict, Any, List
import shutil

import torch

from mi_crow.store.store import Store, TensorMetadata
import safetensors.torch as storch


class LocalStore(Store):
    """Local filesystem implementation of Store interface."""

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
        p = self.base_path / key
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def put_tensor(self, key: str, tensor: torch.Tensor) -> None:
        path = self._full(key)
        tensor_copy = tensor.clone().detach()
        storch.save_file({"tensor": tensor_copy}, str(path))
        del tensor_copy

    def get_tensor(self, key: str) -> torch.Tensor:
        loaded = storch.load_file(str(self._full(key)))
        return loaded["tensor"]

    def _validate_run_id(self, run_id: str) -> None:
        if not run_id or not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"run_id must be a non-empty string, got: {run_id!r}")

    def _validate_batch_index(self, batch_index: int) -> None:
        if batch_index < 0:
            raise ValueError(f"batch_index must be non-negative, got: {batch_index}")

    def _validate_layer_key(self, layer: str, key: str) -> None:
        if not layer or not isinstance(layer, str) or not layer.strip():
            raise ValueError(f"layer must be a non-empty string, got: {layer!r}")
        if not key or not isinstance(key, str) or not key.strip():
            raise ValueError(f"key must be a non-empty string, got: {key!r}")

    def _ensure_directory(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def put_run_batch(self, run_id: str, batch_index: int,
                      tensors: List[torch.Tensor] | Dict[str, torch.Tensor]) -> str:
        if isinstance(tensors, dict):
            to_save = tensors
        elif isinstance(tensors, list):
            if len(tensors) == 0:
                to_save = {"_empty_list": torch.tensor([])}
            else:
                to_save = {f"item_{i}": t for i, t in enumerate(tensors)}
        else:
            to_save = {}
        batch_path = self.base_path / f"{self.runs_prefix}/{run_id}/batch_{batch_index:06d}.safetensors"
        self._ensure_directory(batch_path.parent)
        storch.save_file(to_save, str(batch_path))
        return f"{self.runs_prefix}/{run_id}/batch_{batch_index:06d}.safetensors"

    def get_run_batch(self, run_id: str, batch_index: int) -> List[torch.Tensor] | Dict[
        str, torch.Tensor]:

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
        base = self.base_path / self.runs_prefix / run_id
        if not base.exists():
            return []
        out: set[int] = set()
        
        for p in sorted(base.glob("batch_*.safetensors")):
            name = p.name
            try:
                idx = int(name[len("batch_"): len("batch_") + 6])
                out.add(idx)
            except Exception:
                continue
        
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
        base = self.base_path / self.runs_prefix / run_id
        if not base.exists():
            return
        for p in base.glob("batch_*.safetensors"):
            if p.is_file():
                p.unlink()
        for p in base.glob("batch_*"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        metadata_path = self._run_metadata_key(run_id)
        if metadata_path.exists():
            metadata_path.unlink()

    def put_run_metadata(self, run_id: str, meta: Dict[str, Any]) -> str:
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
        self._validate_run_id(run_id)
        self._validate_batch_index(batch_index)
        
        batch_dir = self._run_batch_key(run_id, batch_index)
        self._ensure_directory(batch_dir)

        tensor_metadata_names = {
            str(layer_signature): list(detector_tensors.keys())
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
                    tensor_copy = tensor.clone().detach()
                    storch.save_file({"tensor": tensor_copy}, str(tensor_path))
                    del tensor_copy
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
        self._validate_run_id(run_id)
        self._validate_batch_index(batch_index)
        
        batch_dir = self._run_batch_key(run_id, batch_index)
        metadata_path = batch_dir / "metadata.json"
        
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

    def put_run_detector_metadata(
            self,
            run_id: str,
            metadata: Dict[str, Any],
            tensor_metadata: TensorMetadata,
    ) -> str:
        self._validate_run_id(run_id)

        detectors_dir = self._run_key(run_id) / "detectors"
        self._ensure_directory(detectors_dir)

        metadata_path = detectors_dir / "metadata.json"

        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    aggregated = json.load(f)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON in unified detector metadata at {metadata_path} for run_id={run_id!r}",
                    e.doc,
                    e.pos,
                ) from e
        else:
            aggregated = {"batches": []}

        batches = aggregated.setdefault("batches", [])
        batch_index = len(batches)

        tensor_metadata_names = {
            layer_signature: list(detector_tensors.keys())
            for layer_signature, detector_tensors in tensor_metadata.items()
            if detector_tensors
        }

        batch_entry = {
            **metadata,
            "batch_index": batch_index,
            "_tensor_metadata_names": tensor_metadata_names,
        }

        batches.append(batch_entry)

        try:
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(aggregated, f, ensure_ascii=False, indent=2)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Unified detector metadata is not JSON-serializable for run_id={run_id!r}. "
                f"Error: {e}"
            ) from e
        except OSError as e:
            raise OSError(
                f"Failed to write unified detector metadata at {metadata_path} for run_id={run_id!r}. "
                f"Error: {e}"
            ) from e

        for layer_signature, detector_tensors in tensor_metadata.items():
            if not detector_tensors:
                continue

            layer_dir = detectors_dir / str(layer_signature)
            self._ensure_directory(layer_dir)

            for tensor_key, tensor in detector_tensors.items():
                tensor_filename = f"{tensor_key}.safetensors"
                tensor_path = layer_dir / tensor_filename

                if tensor_path.exists():
                    try:
                        existing = storch.load_file(str(tensor_path))
                    except Exception as e:
                        raise OSError(
                            f"Failed to load existing unified tensor at {tensor_path} for "
                            f"run_id={run_id!r}, layer={layer_signature!r}, key={tensor_key!r}. "
                            f"Error: {e}"
                        ) from e
                else:
                    existing = {}

                batch_key = f"batch_{batch_index}"
                existing[batch_key] = tensor.clone().detach()

                try:
                    storch.save_file(existing, str(tensor_path))
                    del existing[batch_key]
                except Exception as e:
                    tensor_shape = tuple(tensor.shape) if hasattr(tensor, 'shape') else 'unknown'
                    raise OSError(
                        f"Failed to save unified tensor at {tensor_path} for run_id={run_id!r}, "
                        f"layer={layer_signature!r}, key={tensor_key!r}, batch_index={batch_index}, "
                        f"shape={tensor_shape}. Error: {e}"
                    ) from e

        return f"{self.runs_prefix}/{run_id}/detectors"
