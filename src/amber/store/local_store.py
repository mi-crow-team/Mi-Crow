import json
from pathlib import Path
from typing import List, Dict, Any

import torch

from amber.store.store import Store
import safetensors.torch as storch


class LocalStore(Store):
    base_path: Path | str = ''

    def __init__(
            self,
            base_path: Path | str = '',
            runs_prefix: str = "runs",
            dataset_prefix: str = "datasets",
            model_prefix: str = "models",
    ):
        super().__init__(base_path, runs_prefix, dataset_prefix, model_prefix)

    def _full(self, key: str) -> Path:
        p = self.base_path / key
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def put_tensor(self, key: str, tensor: torch.Tensor) -> None:
        path = self._full(key)
        storch.save_file({"tensor": tensor}, str(path))

    def get_tensor(self, key: str) -> torch.Tensor:
        loaded = storch.load_file(str(self._full(key)))
        return loaded["tensor"]

    def put_run_batch(self, run_id: str, batch_index: int,
                      tensors: List[torch.Tensor] | Dict[str, torch.Tensor]) -> str:
        if isinstance(tensors, dict):
            to_save = tensors
        else:
            to_save = {f"item_{i}": t for i, t in enumerate(tensors)}
        key = self._run_batch_key(run_id, batch_index)
        storch.save_file(to_save, str(self._full(key)))
        return key

    def get_run_batch(self, run_id: str, batch_index: int) -> List[torch.Tensor] | Dict[
        str, torch.Tensor]:
        key = self._run_batch_key(run_id, batch_index)
        batch_path = self._full(key)

        if batch_path.exists():
            loaded = storch.load_file(str(batch_path))
            keys = list(loaded.keys())
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

    def get_run_meta(self, run_id: str) -> Dict[str, Any]:
        metadata_path = self.base_path / self.runs_prefix / run_id / "meta.json"
        if not metadata_path.exists():
            return {}
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def list_run_batches(self, run_id: str) -> List[int]:
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
        base = self.base_path / self.runs_prefix / run_id
        if not base.exists():
            return
        for p in base.glob("batch_*.safetensors"):
            if p.is_file():
                p.unlink()
        # Also delete detector metadata directories
        for p in base.glob("batch_*"):
            if p.is_dir():
                import shutil
                shutil.rmtree(p, ignore_errors=True)

    def put_run_meta(self, run_id: str, meta: Dict[str, Any]) -> str:
        base = self.base_path / self.runs_prefix / run_id
        base.mkdir(parents=True, exist_ok=True)
        metadata_path = base / "meta.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return f"{self.runs_prefix}/{run_id}/meta.json"

    def put_detector_metadata(
            self,
            run_id: str,
            batch_index: int,
            metadata: Dict[str, Any],
            tensor_metadata: Dict[str, Dict[str, torch.Tensor]]
    ) -> str:
        base = self.base_path / self.runs_prefix / run_id / f"batch_{batch_index}"
        base.mkdir(parents=True, exist_ok=True)

        # Save JSON metadata (aggregated from all detectors)
        metadata_path = base / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Process each detector's tensor metadata (keyed by layer_signature)
        for layer_signature, detector_tensors in tensor_metadata.items():
            if not detector_tensors:
                continue

            # Create subdirectory for this detector/layer
            layer_dir = base / layer_signature
            layer_dir.mkdir(parents=True, exist_ok=True)

            # Save each tensor key (e.g., "activations") as a separate safetensors file
            for tensor_key, tensor in detector_tensors.items():
                tensor_filename = f"{tensor_key}.safetensors"
                tensor_path = layer_dir / tensor_filename
                storch.save_file({"tensor": tensor}, str(tensor_path))

        return f"{self.runs_prefix}/{run_id}/batch_{batch_index}"

    def get_detector_metadata(
            self,
            run_id: str,
            batch_index: int,
    ) -> tuple[Any, dict[Any, Any]]:
        base = self.base_path / self.runs_prefix / run_id / f"batch_{batch_index}"
        metadata = {}
        tensor_metadata = {}
        metadata_path = base / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        layer_dirs = [d for d in base.iterdir() if d.is_dir()]
        for layer_dir in layer_dirs:
            layer_signature = layer_dir.name
            detector_tensors = {}
            for tensor_path in layer_dir.iterdir():
                tensor_key = tensor_path.stem
                if tensor_key.startswith("activations_"):
                    tensor_key = tensor_key.split("_", 1)[1]
                detector_tensors[tensor_key] = storch.load_file(str(tensor_path))["tensor"]
            tensor_metadata[layer_signature] = detector_tensors

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
            run_id: Run ID
            batch_index: Batch index
            layer: Layer signature
            key: Tensor key (e.g., "activations")
            
        Returns:
            The requested tensor
            
        Raises:
            FileNotFoundError: If the tensor doesn't exist
        """
        base = self.base_path / self.runs_prefix / run_id / f"batch_{batch_index}"
        layer_dir = base / layer
        tensor_path = layer_dir / f"{key}.safetensors"
        if not tensor_path.exists():
            raise FileNotFoundError(
                f"Tensor not found: run_id={run_id}, batch_index={batch_index}, "
                f"layer={layer}, key={key}"
            )
        return storch.load_file(str(tensor_path))["tensor"]
