from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from server.activation_extractor import ActivationExtractor, iter_hf_dataset, iter_local_files
from server.config import Settings
from server.model_manager import ModelManager
from server.schemas import ActivationRunInfo, ActivationRunListResponse, SaveActivationsRequest, SaveActivationsResponse
from server.storage import activation_dir
from server.utils import generate_id, write_json
from amber.store.local_store import LocalStore

logger = logging.getLogger(__name__)


class ActivationService:
    """Service for managing activation extraction and storage."""

    def __init__(self, settings: Settings):
        self._settings = settings


    def save_activations(
        self, manager: ModelManager, payload: SaveActivationsRequest
    ) -> SaveActivationsResponse:
        """Save activations from a dataset."""
        lm = manager.get_model(payload.model_id)
        
        # Validation
        if payload.batch_size <= 0 or payload.batch_size > 1024:
            raise ValueError("batch_size must be between 1 and 1024")
        if payload.sample_limit is not None and payload.sample_limit <= 0:
            raise ValueError("sample_limit must be positive when provided")

        run_id = payload.run_id or generate_id()
        folder = activation_dir(self._settings.artifact_base_path, payload.model_id, run_id)
        store = LocalStore(base_path=folder)

        # Prepare dataset iterator
        dataset_cfg = payload.dataset
        source = dataset_cfg.get("type")
        if source == "hf":
            name = dataset_cfg.get("name")
            field = dataset_cfg.get("text_field")
            if not name or not field:
                raise ValueError("hf dataset requires 'name' and 'text_field'")
            split = dataset_cfg.get("split", "train")
            iterator = iter_hf_dataset(name=name, split=split, text_field=field)
            dataset_meta = {"type": "hf", "name": name, "split": split, "text_field": field}
        elif source == "local":
            paths = dataset_cfg.get("paths", [])
            if not paths:
                raise ValueError("local dataset requires non-empty 'paths'")
            iterator = iter_local_files(paths)
            dataset_meta = {"type": "local", "paths": paths}
        else:
            raise ValueError("dataset.type must be 'hf' or 'local'")

        # Validate layers
        missing_layers = [layer for layer in payload.layers if layer not in lm.layers.name_to_layer]
        if missing_layers:
            raise ValueError(f"layers not found: {missing_layers}")

        # Extract activations
        extractor = ActivationExtractor(
            lm=lm,
            layers=payload.layers,
            batch_size=payload.batch_size,
            shard_size=payload.shard_size,
        )
        manifest = extractor.extract(
            texts=iterator,
            out_dir=folder,
            limit=payload.sample_limit,
            store=store,
            run_id=run_id,
        )

        # Create manifest
        created_at = datetime.now(timezone.utc).isoformat()
        manifest_data = {
            "model_id": payload.model_id,
            "layers": payload.layers,
            "dataset": dataset_meta,
            "samples": manifest["samples"],
            "tokens": manifest.get("tokens", 0),
            "batches": manifest.get("batches", []),
            "shards": manifest.get("shards", []),
            "run_id": run_id,
            "store_path": str(folder),
            "created_at": created_at,
            "status": "done",
        }
        manifest_path = folder / "manifest.json"
        write_json(manifest_path, manifest_data)
        
        try:
            store.put_run_metadata(run_id, manifest_data)
        except Exception:
            logger.debug("failed to persist store metadata", exc_info=True)
        
        logger.info(
            "activations_saved",
            extra={"model_id": payload.model_id, "run_id": run_id, "samples": manifest["samples"]},
        )

        return SaveActivationsResponse(
            path=str(manifest_path),
            manifest_path=str(manifest_path),
            run_id=run_id,
            samples=manifest["samples"],
            tokens=manifest.get("tokens", 0),
            layers=payload.layers,
            batches=manifest.get("batches", []),
            dataset=dataset_meta,
            status="done",
            created_at=created_at,
        )

    def list_activation_runs(self, model_id: str) -> ActivationRunListResponse:
        """List all activation runs for a model."""
        root = self._settings.artifact_base_path / "activations" / model_id
        runs: List[ActivationRunInfo] = []
        
        if root.exists():
            for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
                # Prefer per-run meta.json written by the store; fall back to manifest.json for older runs.
                meta_path = run_dir / "meta.json"
                manifest_path = run_dir / "manifest.json"
                meta: dict = {}
                data_path: Path | None = None
                
                if meta_path.exists():
                    data_path = meta_path
                    try:
                        meta = json.loads(meta_path.read_text())
                    except Exception:
                        meta = {}
                elif manifest_path.exists():
                    data_path = manifest_path
                    try:
                        meta = json.loads(manifest_path.read_text())
                    except Exception:
                        meta = {}
                
                runs.append(
                    ActivationRunInfo(
                        model_id=model_id,
                        run_id=run_dir.name,
                        manifest_path=str(data_path) if data_path and data_path.exists() else None,
                        samples=meta.get("samples"),
                        tokens=meta.get("tokens"),
                        layers=meta.get("layers", []),
                        dataset=meta.get("dataset", {}),
                        created_at=meta.get("created_at"),
                        status=meta.get("status") or "done",
                    )
                )
        
        return ActivationRunListResponse(model_id=model_id, runs=runs)

    def delete_activation_run(self, model_id: str, run_id: str) -> bool:
        """Delete an activation run."""
        import shutil
        folder = self._settings.artifact_base_path / "activations" / model_id / run_id
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
            return True
        return False

    def get_layer_size(self, activations_path: str, layer: str) -> dict[str, int]:
        """Get the hidden dimension (size) of a layer from an activation run."""
        activations_path_obj = Path(activations_path)
        if not activations_path_obj.exists():
            raise ValueError(f"activations_path '{activations_path}' does not exist")
        
        manifest = json.loads(activations_path_obj.read_text())
        store_path = Path(manifest.get("store_path", activations_path_obj.parent))
        run_id = manifest.get("run_id") or activations_path_obj.parent.name
        store = LocalStore(base_path=store_path)
        
        batch_indices = [b.get("batch_index") for b in manifest.get("batches", []) if "batch_index" in b]
        if not batch_indices:
            batch_indices = store.list_run_batches(run_id)
        if not batch_indices:
            raise ValueError("no activation batches found")
        
        first_batch = batch_indices[0]
        activations = store.get_detector_metadata_by_layer_by_key(run_id, first_batch, layer, "activations")
        hidden_dim = activations.shape[-1]
        
        return {"layer": layer, "hidden_dim": int(hidden_dim)}

