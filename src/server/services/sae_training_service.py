from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Type

from server.config import Settings
from server.job_manager import JobManager
from server.model_manager import ModelManager
from server.schemas import TrainSAERequest, TrainSAEResponse, TrainStatusResponse
from server.storage import sae_dir
from server.utils import SAERegistry, generate_id, write_json
from mi_crow.store.local_store import LocalStore
from mi_crow.mechanistic.sae.sae import Sae
from mi_crow.mechanistic.sae.sae_trainer import SaeTrainingConfig
from mi_crow.mechanistic.sae.modules.topk_sae import TopKSaeTrainingConfig
import os

logger = logging.getLogger(__name__)


class SAETrainingService:
    """Service for managing SAE training."""

    def __init__(self, settings: Settings, job_manager: JobManager):
        self._settings = settings
        self._job_manager = job_manager
        self._sae_registry: Dict[str, Path] = {}
        self._job_id_map: Dict[str, str] = {}  # idempotency_key -> job_id
        self._sae_registry_class = SAERegistry()

    def _get_sae_class(self, name: str) -> Type[Sae]:
        """Get SAE class by name."""
        return self._sae_registry_class.get_class(name)

    def _build_training_config(self, payload: TrainSAERequest, sae_class: str, sae_kwargs: dict) -> SaeTrainingConfig:
        """Build training config from payload.
        
        Adds wandb API key from settings or environment if available.
        For TopKSae, uses TopKSaeTrainingConfig and includes k parameter.
        """
        merged = {**payload.hyperparams, **payload.training_config}
        
        if self._settings.wandb_api_key and "wandb_api_key" not in merged:
            merged["wandb_api_key"] = self._settings.wandb_api_key
        elif "wandb_api_key" not in merged:
            env_wandb_key = os.getenv("WANDB_API_KEY")
            if env_wandb_key:
                merged["wandb_api_key"] = env_wandb_key
        
        if "wandb_project" not in merged or not merged.get("wandb_project"):
            if self._settings.wandb_project:
                merged["wandb_project"] = self._settings.wandb_project
            else:
                env_wandb_project = os.getenv("WANDB_PROJECT") or os.getenv("SERVER_WANDB_PROJECT")
                if env_wandb_project:
                    merged["wandb_project"] = env_wandb_project
        
        if sae_class == "TopKSae":
            if "k" not in sae_kwargs:
                raise ValueError("TopKSae requires 'k' parameter in sae_kwargs")
            merged["k"] = sae_kwargs["k"]
            try:
                return TopKSaeTrainingConfig(**merged)
            except TypeError as exc:
                from server.exceptions import ValidationError
                raise ValidationError(f"Invalid TopKSae training config: {exc}") from exc
        
        try:
            return SaeTrainingConfig(**merged)
        except TypeError as exc:
            from server.exceptions import ValidationError
            raise ValidationError(f"Invalid training config: {exc}") from exc

    def train_sae(self, manager: ModelManager, payload: TrainSAERequest) -> TrainSAEResponse:
        """Start SAE training asynchronously."""
        manager.get_model(payload.model_id)
        activations_path = Path(payload.activations_path)
        if not activations_path.exists():
            raise ValueError(f"activations_path '{activations_path}' does not exist")
        
        manifest = json.loads(activations_path.read_text())
        manifest_layers = manifest.get("layers") or []
        store_path = Path(manifest.get("store_path", activations_path.parent))
        run_id = payload.run_id or manifest.get("run_id") or generate_id()
        
        if not manifest_layers:
            raise ValueError("activations manifest missing layers")
        
        layer = payload.layer or (manifest_layers[0] if len(manifest_layers) == 1 else None)
        if not layer:
            raise ValueError("layer is required when multiple layers are present")
        
        sae_class = payload.sae_class or "TopKSae"
        store = LocalStore(base_path=store_path)
        
        batch_indices = [b.get("batch_index") for b in manifest.get("batches", []) if "batch_index" in b]
        if not batch_indices:
            batch_indices = store.list_run_batches(run_id)
        if not batch_indices:
            raise ValueError("no activation batches found for training")
        
        first_batch = batch_indices[0]
        activations = store.get_detector_metadata_by_layer_by_key(run_id, first_batch, layer, "activations")
        hidden_dim = activations.shape[-1]
        n_latents = payload.n_latents or hidden_dim
        sae_kwargs = dict(payload.sae_kwargs)
        sae_kwargs.pop("n_latents", None)
        sae_kwargs.pop("n_inputs", None)
        
        if sae_class == "TopKSae":
            if "k" not in sae_kwargs:
                raise ValueError("TopKSae requires 'k' parameter in sae_kwargs")
        elif sae_class == "L1Sae":
            pass
        
        config = self._build_training_config(payload, sae_class, sae_kwargs)
        
        idempotency_key = f"{payload.model_id}:{activations_path}:{payload.hyperparams}:{layer}:{sae_class}"

        def _create_training_func(job_id: str):
            """Create training function with job_id in closure."""
            def _run():
                start = time.time()
                folder = sae_dir(self._settings.artifact_base_path, payload.model_id, run_id)
                sae = self._get_sae_class(sae_class)(
                    n_latents=n_latents,
                    n_inputs=hidden_dim,
                    **sae_kwargs,
                )
                sae.context.model_id = payload.model_id
                sae.context.lm_layer_signature = layer
                
                self._job_manager.append_log(job_id, "training_started")
                self._job_manager.set_progress(job_id, 0.0)
                
                logger.info(
                    "sae_train_start",
                    extra={"model_id": payload.model_id, "sae_id": run_id, "layer": layer, "sae_class": sae_class},
                )
                
                train_result = sae.train(
                    store=store,
                    run_id=run_id,
                    layer_signature=layer,
                    config=config,
                    training_run_id=run_id,
                )
                
                sae_name = "sae"
                # Get k from config for TopKSae
                k_value = None
                if sae_class == "TopKSae" and hasattr(config, 'k'):
                    k_value = config.k
                sae.save(name=sae_name, path=folder, k=k_value)
                sae_path = folder / f"{sae_name}.pt"
                metadata = {
                    "sae_id": run_id,
                    "sae_class": sae_class,
                    "sae_kwargs": sae_kwargs,
                    "layer": layer,
                    "model_id": payload.model_id,
                    "activations_path": str(activations_path),
                    "manifest": manifest,
                    "training": {
                        "result": train_result,
                        "config": config.__dict__,
                        "duration_sec": time.time() - start,
                        "wandb_url": train_result.get("wandb_url"),
                    },
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "sae_path": str(sae_path),
                }
                metadata_path = folder / "metadata.json"
                write_json(metadata_path, metadata)
                self._sae_registry[run_id] = sae_path
                self._job_manager.set_progress(job_id, 1.0)
                self._job_manager.append_log(job_id, "training_completed")
                
                logger.info(
                    "sae_train_complete",
                    extra={
                        "model_id": payload.model_id,
                        "sae_id": run_id,
                        "layer": layer,
                        "duration": metadata["training"]["duration_sec"],
                    },
                )
                
                return {"sae_id": run_id, "sae_path": str(sae_path), "metadata_path": str(metadata_path)}
            return _run

        if idempotency_key and idempotency_key in self._job_id_map:
            existing_job_id = self._job_id_map[idempotency_key]
            return TrainSAEResponse(job_id=existing_job_id, status="pending")

        def _run_wrapper():
            """Wrapper that gets job_id from idempotency mapping."""
            job_id = self._job_id_map.get(idempotency_key) if idempotency_key else None
            if not job_id and idempotency_key:
                idempotency_dict = getattr(self._job_manager, "_idempotency", {})
                job_id = idempotency_dict.get(idempotency_key)
            if not job_id:
                raise RuntimeError("job_id not available in training closure")
            
            training_func = _create_training_func(job_id)
            return training_func()

        job_id = self._job_manager.submit(
            job_type="sae_train",
            func=_run_wrapper,
            idempotency_key=idempotency_key,
            timeout_sec=3600,
        )
        
        if idempotency_key:
            self._job_id_map[idempotency_key] = job_id
        
        logger.info("train_job_submitted", extra={"job_id": job_id, "model_id": payload.model_id})
        return TrainSAEResponse(job_id=job_id, status="pending")

    def train_status(self, job_id: str) -> TrainStatusResponse:
        """Get training job status."""
        job = self._job_manager.status(job_id)
        result = job.get("result") or {}
        return TrainStatusResponse(
            job_id=job_id,
            status=str(job.get("status")),
            sae_id=result.get("sae_id"),
            sae_path=result.get("sae_path"),
            metadata_path=result.get("metadata_path"),
            progress=job.get("progress"),
            logs=job.get("logs", []),
            error=job.get("error"),
        )

    def cancel_train(self, job_id: str) -> TrainStatusResponse:
        """Cancel a training job."""
        job = self._job_manager.cancel(job_id)
        result = job.get("result") or {}
        return TrainStatusResponse(
            job_id=job_id,
            status=str(job.get("status")),
            sae_id=result.get("sae_id"),
            sae_path=result.get("sae_path"),
            metadata_path=result.get("metadata_path"),
            progress=job.get("progress"),
            logs=job.get("logs", []),
            error=job.get("error"),
        )

    def get_sae_registry(self) -> Dict[str, Path]:
        """Get the SAE registry."""
        return self._sae_registry

    def register_sae(self, sae_id: str, sae_path: Path) -> None:
        """Register an SAE in the registry."""
        self._sae_registry[sae_id] = sae_path

