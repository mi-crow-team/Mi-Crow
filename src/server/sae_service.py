from __future__ import annotations

import json
import uuid
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type, List
import shutil

import torch

from server.activation_extractor import ActivationExtractor, iter_hf_dataset, iter_local_files
from server.config import Settings
from server.inference_service import InferenceService
from server.job_manager import JobManager
from server.model_manager import ModelManager
from amber.store.local_store import LocalStore
from amber.mechanistic.sae.modules.topk_sae import TopKSae
from amber.mechanistic.sae.modules.l1_sae import L1Sae
from amber.mechanistic.sae.sae import Sae
from amber.mechanistic.sae.sae_trainer import SaeTrainingConfig
from amber.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary
from server.schemas import (
    ActivationRunInfo,
    ActivationRunListResponse,
    ConceptListResponse,
    ConceptLoadRequest,
    ConceptLoadResponse,
    ConceptConfigListResponse,
    ConceptConfigInfo,
    ConceptPreviewRequest,
    ConceptPreviewResponse,
    ConceptManipulationRequest,
    ConceptManipulationResponse,
    LoadSAERequest,
    LoadSAEResponse,
    SAEInferenceRequest,
    SAEInferenceResponse,
    SaeRunInfo,
    SaeRunListResponse,
    SaveActivationsRequest,
    SaveActivationsResponse,
    TrainSAERequest,
    TrainSAEResponse,
    TrainStatusResponse,
)
from server.storage import activation_dir, concepts_dir, sae_dir, top_texts_dir

logger = logging.getLogger(__name__)


def _write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


class SAEService:
    def __init__(self, settings: Settings, inference_service: InferenceService, job_manager: JobManager):
        self._settings = settings
        self._inference_service = inference_service
        self._job_manager = job_manager
        self._sae_registry: Dict[str, Path] = {}
        self._concept_configs: Dict[str, Path] = {}
        self._sae_classes: Dict[str, Type[Sae]] = {
            "TopKSae": TopKSae,
            "L1Sae": L1Sae,
        }

    def _generate_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def _get_sae_class(self, name: str) -> Type[Sae]:
        if name not in self._sae_classes:
            raise ValueError(f"Unsupported SAE class '{name}'. Available: {sorted(self._sae_classes.keys())}")
        return self._sae_classes[name]

    def _load_sae(self, sae_class: str, path: Path) -> Sae:
        cls = self._get_sae_class(sae_class)
        if not hasattr(cls, "load"):
            raise ValueError(f"SAE class '{sae_class}' does not implement load()")
        return cls.load(path)

    def _build_training_config(self, payload: TrainSAERequest) -> SaeTrainingConfig:
        merged = {**payload.hyperparams, **payload.training_config}
        try:
            return SaeTrainingConfig(**merged)
        except TypeError as exc:
            raise ValueError(f"Invalid training config: {exc}") from exc

    def list_sae_classes(self) -> Dict[str, str]:
        """
        Return the available SAE classes as a name -> class dotted path mapping.

        This can be used by the UI to present a fixed enum of supported SAE types.
        """
        out: Dict[str, str] = {}
        for name, cls in self._sae_classes.items():
            out[name] = f"{cls.__module__}.{cls.__name__}"
        return out

    def save_activations(self, manager: ModelManager, payload: SaveActivationsRequest) -> SaveActivationsResponse:
        lm = manager.get_model(payload.model_id)
        if payload.batch_size <= 0 or payload.batch_size > 1024:
            raise ValueError("batch_size must be between 1 and 1024")
        if payload.sample_limit is not None and payload.sample_limit <= 0:
            raise ValueError("sample_limit must be positive when provided")
        run_id = payload.run_id or self._generate_id()
        folder = activation_dir(self._settings.artifact_base_path, payload.model_id, run_id)
        store = LocalStore(base_path=folder)

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

        missing_layers = [layer for layer in payload.layers if layer not in lm.layers.name_to_layer]
        if missing_layers:
            raise ValueError(f"layers not found: {missing_layers}")

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
        created_at = datetime.utcnow().isoformat()
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
        _write_json(manifest_path, manifest_data)
        try:
            store.put_run_metadata(run_id, manifest_data)
        except Exception:
            logger.debug("failed to persist store metadata", exc_info=True)
        logger.info("activations_saved", extra={"model_id": payload.model_id, "run_id": run_id, "samples": manifest["samples"]})

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

    def train_sae(self, manager: ModelManager, payload: TrainSAERequest) -> TrainSAEResponse:
        manager.get_model(payload.model_id)
        activations_path = Path(payload.activations_path)
        if not activations_path.exists():
            raise ValueError(f"activations_path '{activations_path}' does not exist")
        manifest = json.loads(activations_path.read_text())
        manifest_layers = manifest.get("layers") or []
        store_path = Path(manifest.get("store_path", activations_path.parent))
        run_id = payload.run_id or manifest.get("run_id") or self._generate_id()
        if not manifest_layers:
            raise ValueError("activations manifest missing layers")
        layer = payload.layer or (manifest_layers[0] if len(manifest_layers) == 1 else None)
        if not layer:
            raise ValueError("layer is required when multiple layers are present")
        sae_class = payload.sae_class or "TopKSae"
        config = self._build_training_config(payload)
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
            # L1Sae doesn't require any special kwargs for now
            pass
        idempotency_key = f"{payload.model_id}:{activations_path}:{payload.hyperparams}:{layer}:{sae_class}"

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
            sae.save(name=sae_name, path=folder)
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
                },
                "created_at": datetime.utcnow().isoformat(),
                "sae_path": str(sae_path),
            }
            metadata_path = folder / "metadata.json"
            _write_json(metadata_path, metadata)
            self._sae_registry[run_id] = sae_path
            self._job_manager.set_progress(job_id, 1.0)
            self._job_manager.append_log(job_id, "training_completed")
            logger.info(
                "sae_train_complete",
                extra={"model_id": payload.model_id, "sae_id": run_id, "layer": layer, "duration": metadata["training"]["duration_sec"]},
            )
            return {"sae_id": run_id, "sae_path": str(sae_path), "metadata_path": str(metadata_path)}

        job_id = self._job_manager.submit(
            job_type="sae_train",
            func=_run,
            idempotency_key=idempotency_key,
            timeout_sec=3600,
        )
        logger.info("train_job_submitted", extra={"job_id": job_id, "model_id": payload.model_id})
        return TrainSAEResponse(job_id=job_id, status="pending")

    def train_status(self, job_id: str) -> TrainStatusResponse:
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

    def load_sae(self, payload: LoadSAERequest) -> LoadSAEResponse:
        sae_path = Path(payload.sae_path)
        if not sae_path.exists():
            raise ValueError(f"sae_path '{sae_path}' does not exist")
        sae_id = sae_path.parent.name or sae_path.stem
        self._sae_registry[sae_id] = sae_path
        return LoadSAEResponse(sae_id=sae_id)

    def _resolve_sae_path(self, sae_id: Optional[str], sae_path: Optional[str]) -> Path:
        if sae_path:
            return Path(sae_path)
        if sae_id and sae_id in self._sae_registry:
            return self._sae_registry[sae_id]
        raise ValueError("SAE not loaded; provide sae_path or load first")

    def _apply_concept_config(self, sae: Sae, concept_config: Dict[str, any]) -> None:
        weights = concept_config.get("weights") or concept_config.get("edits") or {}
        bias_weights = concept_config.get("bias") or {}
        if weights:
            mult = torch.ones_like(sae.concepts.multiplication.data)
            for idx, val in weights.items():
                try:
                    i = int(idx)
                except Exception:
                    continue
                if 0 <= i < mult.numel():
                    mult[i] = float(val)
            sae.concepts.multiplication.data = mult
        if bias_weights:
            bias_tensor = torch.zeros_like(sae.concepts.bias.data)
            for idx, val in bias_weights.items():
                try:
                    i = int(idx)
                except Exception:
                    continue
                if 0 <= i < bias_tensor.numel():
                    bias_tensor[i] = float(val)
            sae.concepts.bias.data = bias_tensor

    def list_activation_runs(self, model_id: str) -> ActivationRunListResponse:
        root = self._settings.artifact_base_path / "activations" / model_id
        runs: List[ActivationRunInfo] = []
        if root.exists():
            for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
                # Prefer per-run meta.json written by the store; fall back to manifest.json for older runs.
                meta_path = run_dir / "meta.json"
                manifest_path = run_dir / "manifest.json"
                meta: Dict[str, Any] = {}
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

    def list_sae_runs(self, model_id: str) -> SaeRunListResponse:
        root = self._settings.artifact_base_path / "sae" / model_id
        saes: List[SaeRunInfo] = []
        if root.exists():
            for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
                metadata_path = run_dir / "metadata.json"
                sae_file = next(run_dir.glob("*.pt"), None)
                meta: Dict[str, Any] = {}
                if metadata_path.exists():
                    try:
                        meta = json.loads(metadata_path.read_text())
                    except Exception:
                        meta = {}
                saes.append(
                    SaeRunInfo(
                        model_id=model_id,
                        sae_id=run_dir.name,
                        sae_path=str(meta.get("sae_path") or sae_file) if (meta.get("sae_path") or sae_file) else None,
                        metadata_path=str(metadata_path) if metadata_path.exists() else None,
                        sae_class=meta.get("sae_class"),
                        layer=meta.get("layer"),
                        created_at=meta.get("created_at"),
                    )
                )
        return SaeRunListResponse(model_id=model_id, saes=saes)

    def get_sae_metadata(self, model_id: str, sae_id: str) -> Dict[str, Any]:
        root = self._settings.artifact_base_path / "sae" / model_id / sae_id
        metadata_path = root / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Metadata not found for SAE {sae_id}")
        try:
            return json.loads(metadata_path.read_text())
        except Exception as e:
            raise ValueError(f"Failed to read metadata: {e}") from e

    def infer(self, manager: ModelManager, payload: SAEInferenceRequest) -> SAEInferenceResponse:
        lm = manager.get_model(payload.model_id)
        sae_path = self._resolve_sae_path(payload.sae_id, payload.sae_path)
        resolved_sae_id = payload.sae_id or sae_path.parent.name
        metadata_path = sae_path.parent / "metadata.json"
        meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        sae_class = meta.get("sae_class", "TopKSae")
        sae_layer = payload.layer or meta.get("layer")
        if not sae_layer:
            raise ValueError("layer must be specified for SAE inference")
        concept_config = {}
        concept_path = None
        if payload.concept_config_path:
            concept_path = Path(payload.concept_config_path)
            if not concept_path.exists():
                raise ValueError(f"concept_config_path '{concept_path}' does not exist")
            concept_config = json.loads(concept_path.read_text())
        elif payload.sae_id and payload.sae_id in self._concept_configs:
            concept_path = self._concept_configs[payload.sae_id]
            concept_config = json.loads(concept_path.read_text())

        sae_hook = self._load_sae(sae_class, sae_path)
        sae_hook.context.lm = lm
        sae_hook.context.lm_layer_signature = sae_layer

        if concept_config:
            self._apply_concept_config(sae_hook, concept_config)

        hook_id = lm.layers.register_hook(sae_layer, sae_hook)
        start = time.time()
        try:
            if payload.track_texts or payload.save_top_texts:
                sae_hook.context.text_tracking_enabled = True
                sae_hook.context.text_tracking_k = payload.top_k_neurons
                sae_hook.context.text_tracking_negative = False
                sae_hook.concepts.enable_text_tracking()
            outputs = self._inference_service.run(lm, payload.inputs)
        finally:
            try:
                lm.layers.unregister_hook(hook_id)
            except Exception:
                pass
        duration = (time.time() - start) * 1000
        logger.info(
            "sae_infer_complete",
            extra={"model_id": payload.model_id, "sae_id": payload.sae_id, "layer": sae_layer, "duration_ms": duration},
        )

        top_neurons: List[Dict[str, Any]] = []
        token_latents: List[Any] = []
        activations_tensor = sae_hook.tensor_metadata.get("activations")
        if isinstance(activations_tensor, torch.Tensor):
            act = activations_tensor
            if act.dim() == 3:
                per_prompt = act.max(dim=1).values
                if payload.return_token_latents:
                    seq_len = act.shape[1]
                    k_tok = max(1, min(payload.top_k_neurons, act.shape[-1]))
                    for idx in range(min(len(payload.inputs), act.shape[0])):
                        per_token = []
                        for t in range(seq_len):
                            row = act[idx, t]
                            topk_tok = torch.topk(row, k_tok)
                            per_token.append(
                                {
                                    "token_index": t,
                                    "neuron_ids": [int(i) for i in topk_tok.indices],
                                    "activations": [float(v) for v in topk_tok.values],
                                }
                            )
                        token_latents.append({"prompt_index": idx, "tokens": per_token})
            elif act.dim() == 2:
                per_prompt = act
            else:
                per_prompt = act.view(act.shape[0], -1)
            k = max(1, min(payload.top_k_neurons, per_prompt.shape[-1]))
            for idx in range(min(len(payload.inputs), per_prompt.shape[0])):
                row = per_prompt[idx]
                topk = torch.topk(row, k)
                top_neurons.append(
                    {
                        "prompt_index": idx,
                        "prompt": payload.inputs[idx].prompt,
                        "neuron_ids": [int(i) for i in topk.indices],
                        "activations": [float(v) for v in topk.values],
                    }
                )

        top_texts_path: str | None = None
        if payload.save_top_texts:
            sae_id = resolved_sae_id or self._generate_id()
            folder = top_texts_dir(self._settings.artifact_base_path, payload.model_id, sae_id)
            folder.mkdir(parents=True, exist_ok=True)
            neuron_texts = {}
            try:
                for idx, texts in enumerate(sae_hook.concepts.get_all_top_texts()):
                    neuron_texts[idx] = [
                        {
                            "text": t.text,
                            "score": t.score,
                            "token_idx": t.token_idx,
                            "token_str": t.token_str,
                        }
                        for t in texts
                    ]
            except Exception:
                neuron_texts = {}
            top_path = folder / "top_texts.json"
            _write_json(
                top_path,
                {
                    "sae_path": str(sae_path),
                    "sae_id": sae_id,
                    "layer": sae_layer,
                    "prompts": [inp.prompt for inp in payload.inputs],
                    "outputs": [out.model_dump() for out in outputs],
                    "top_neurons": top_neurons,
                    "neuron_texts": neuron_texts,
                    "concept_config_path": str(concept_path) if concept_path else None,
                },
            )
            top_texts_path = str(top_path)
        return SAEInferenceResponse(
            outputs=outputs,
            sae_id=resolved_sae_id,
            sae_path=str(sae_path),
            metadata_path=str(metadata_path) if metadata_path.exists() else None,
            top_neurons=top_neurons,
            token_latents=token_latents,
            top_texts_path=top_texts_path,
        )

    def list_concepts(self, payload_model_id: str, sae_id: str | None = None) -> ConceptListResponse:
        folder = concepts_dir(self._settings.artifact_base_path, payload_model_id, sae_id)
        concepts = []
        for item in folder.glob("*.json"):
            concepts.append(item.name)
        return ConceptListResponse(base_path=str(folder), concepts=sorted(concepts))

    def get_concept_dictionary(self, model_id: str, sae_id: str | None = None) -> "ConceptDictionaryResponse":
        from server.schemas import ConceptDictionaryResponse
        folder = concepts_dir(self._settings.artifact_base_path, model_id, sae_id)
        concepts_path = folder / "concepts.json"
        if not concepts_path.exists():
            raise ValueError(f"Concepts file not found at {concepts_path}")
        with concepts_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return ConceptDictionaryResponse(
            n_size=data.get("n_size", 0),
            concepts=data.get("concepts", {})
        )

    def list_concept_configs(self, model_id: str, sae_id: str | None = None) -> ConceptConfigListResponse:
        folder = concepts_dir(self._settings.artifact_base_path, model_id, sae_id)
        configs: List[ConceptConfigInfo] = []
        for item in folder.glob("concepts_*.json"):
            meta: Dict[str, Any] = {}
            try:
                meta = json.loads(item.read_text())
            except Exception:
                meta = {}
            configs.append(
                ConceptConfigInfo(
                    name=item.name,
                    path=str(item),
                    created_at=meta.get("created_at"),
                    sae_id=sae_id,
                    layer=meta.get("layer"),
                )
            )
        return ConceptConfigListResponse(model_id=model_id, sae_id=sae_id, configs=sorted(configs, key=lambda x: x.name))

    def preview_concepts(self, payload: ConceptPreviewRequest) -> ConceptPreviewResponse:
        sae_path = self._resolve_sae_path(payload.sae_id, payload.sae_path)
        sae_id = payload.sae_id or sae_path.parent.name or self._generate_id()
        sae_meta_path = sae_path.parent / "metadata.json"
        sae_meta = json.loads(sae_meta_path.read_text()) if sae_meta_path.exists() else {}
        sae_class = sae_meta.get("sae_class", "TopKSae")
        sae_hook = self._load_sae(sae_class, sae_path)
        n_latents = sae_hook.context.n_latents
        validated_weights: Dict[str, float] = {}
        for idx, val in payload.edits.items():
            i = int(idx)
            if i < 0 or i >= n_latents:
                raise ValueError(f"edit index {idx} out of bounds for n_latents={n_latents}")
            validated_weights[str(i)] = float(val)
        validated_bias: Dict[str, float] = {}
        for idx, val in payload.bias.items():
            i = int(idx)
            if i < 0 or i >= n_latents:
                raise ValueError(f"bias index {idx} out of bounds for n_latents={n_latents}")
            validated_bias[str(i)] = float(val)
        return ConceptPreviewResponse(
            sae_id=sae_id,
            layer=sae_meta.get("layer"),
            n_latents=n_latents,
            weights=validated_weights,
            bias=validated_bias,
        )

    def delete_activation_run(self, model_id: str, run_id: str) -> bool:
        folder = self._settings.artifact_base_path / "activations" / model_id / run_id
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
            return True
        return False

    def delete_sae_run(self, model_id: str, sae_id: str) -> bool:
        folder = self._settings.artifact_base_path / "sae" / model_id / sae_id
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
            self._sae_registry.pop(sae_id, None)
            return True
        return False

    def load_concepts(self, payload: ConceptLoadRequest) -> ConceptLoadResponse:
        src = Path(payload.source_path)
        if not src.exists():
            raise ValueError(f"source_path '{src}' does not exist")
        sae_path = self._resolve_sae_path(payload.sae_id, None)
        sae_meta_path = sae_path.parent / "metadata.json"
        sae_meta = json.loads(sae_meta_path.read_text()) if sae_meta_path.exists() else {}
        sae_class = sae_meta.get("sae_class", "TopKSae")
        sae_hook = self._load_sae(sae_class, sae_path)
        n_latents = sae_hook.context.n_latents
        if src.suffix.lower() == ".csv":
            concept_dict = ConceptDictionary.from_csv(src, n_size=n_latents)
        else:
            concept_dict = ConceptDictionary.from_json(src, n_size=n_latents)
        sae_id = payload.sae_id or sae_path.parent.name or self._generate_id()
        dst_dir = concepts_dir(self._settings.artifact_base_path, payload.model_id, sae_id)
        concept_dict.set_directory(dst_dir)
        saved_path = concept_dict.save()
        concept_id = self._generate_id()
        return ConceptLoadResponse(concept_id=concept_id, stored_path=str(saved_path))

    def manipulate_concepts(self, payload: ConceptManipulationRequest) -> ConceptManipulationResponse:
        sae_path = self._resolve_sae_path(payload.sae_id, payload.sae_path)
        sae_id = payload.sae_id or sae_path.parent.name or self._generate_id()
        sae_meta_path = sae_path.parent / "metadata.json"
        sae_meta = json.loads(sae_meta_path.read_text()) if sae_meta_path.exists() else {}
        sae_class = sae_meta.get("sae_class", "TopKSae")
        sae_hook = self._load_sae(sae_class, sae_path)
        n_latents = sae_hook.context.n_latents
        for idx in payload.edits.keys():
            if int(idx) < 0 or int(idx) >= n_latents:
                raise ValueError(f"edit index {idx} out of bounds for n_latents={n_latents}")
        for idx in payload.bias.keys():
            if int(idx) < 0 or int(idx) >= n_latents:
                raise ValueError(f"bias index {idx} out of bounds for n_latents={n_latents}")
        folder = concepts_dir(self._settings.artifact_base_path, payload.model_id, sae_id)
        config_path = folder / f"concepts_{self._generate_id()}.json"
        _write_json(
            config_path,
            {
                "sae_path": str(sae_path),
                "sae_class": sae_class,
                "layer": sae_meta.get("layer"),
                "weights": payload.edits,
                "bias": payload.bias,
                "created_at": datetime.utcnow().isoformat(),
            },
        )
        self._concept_configs[sae_id] = config_path
        return ConceptManipulationResponse(concept_config_path=str(config_path), sae_id=sae_id)

    def get_layer_size(self, activations_path: str, layer: str) -> Dict[str, Any]:
        """Get the hidden dimension (size) of a layer from an activation run."""
        from server.schemas import LayerSizeInfo
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
