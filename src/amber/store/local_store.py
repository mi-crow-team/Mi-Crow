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

    def put_run_metadata(self, run_id: str, meta: Dict[str, Any]) -> None:
        base = self._run_metadata_key(run_id)
        base.mkdir(parents=True, exist_ok=True)
        with self._run_metadata_key(run_id).open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def get_run_metadata(self, run_id: str) -> Dict[str, Any]:
        base = self._run_metadata_key(run_id)
        if not base.exists():
            return {}
        with base.open("r", encoding="utf-8") as f:
            return json.load(f)

    def put_detector_metadata(
            self,
            run_id: str,
            batch_index: int,
            metadata: Dict[str, Any],
            tensor_metadata: Dict[str, Dict[str, torch.Tensor]]
    ) -> str:
        base = self._run_batch_key(run_id, batch_index)
        base.mkdir(parents=True, exist_ok=True)

        tensor_metadata_names = {
            layer_signature: list(detector_tensors.keys())
            for layer_signature, detector_tensors in tensor_metadata.items()
            if detector_tensors
        }
        metadata_with_tensor_names = {
            **metadata,
            "_tensor_metadata_names": tensor_metadata_names
        }

        detector_metadata_path = base / "metadata.json"
        with detector_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata_with_tensor_names, f, ensure_ascii=False, indent=2)

        for layer_signature, detector_tensors in tensor_metadata.items():
            if not detector_tensors:
                continue

            layer_dir = base / layer_signature
            layer_dir.mkdir(parents=True, exist_ok=True)

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
        base = self._run_batch_key(run_id, batch_index)
        metadata = {}
        tensor_metadata = {}
        metadata_path = base / "metadata.json"

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        tensor_metadata_names = metadata.pop("_tensor_metadata_names", None)

        if tensor_metadata_names is not None:
            for layer_signature, tensor_keys in tensor_metadata_names.items():
                layer_dir = base / layer_signature
                detector_tensors = {}
                for tensor_key in tensor_keys:
                    tensor_filename = f"{tensor_key}.safetensors"
                    tensor_path = layer_dir / tensor_filename
                    if tensor_path.exists():
                        detector_tensors[tensor_key] = storch.load_file(str(tensor_path))["tensor"]
                if detector_tensors:
                    tensor_metadata[layer_signature] = detector_tensors
        else:
            raise ValueError(
                "Field _tensor_metadata_names not found in detector metadata. Cannot retrieve tensors."
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
            run_id: Run ID
            batch_index: Batch index
            layer: Layer signature
            key: Tensor key (e.g., "activations")
            
        Returns:
            The requested tensor
            
        Raises:
            FileNotFoundError: If the tensor doesn't exist
        """
        base = self._run_batch_key(run_id, batch_index)
        layer_dir = base / layer
        tensor_path = layer_dir / f"{key}.safetensors"
        if not tensor_path.exists():
            raise FileNotFoundError(
                f"Tensor not found: run_id={run_id}, batch_index={batch_index}, "
                f"layer={layer}, key={key}"
            )
        return storch.load_file(str(tensor_path))["tensor"]
