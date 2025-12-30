from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import torch

from server.config import Settings
from server.inference_service import InferenceService
from server.job_manager import JobManager
from server.model_manager import ModelManager
from server.schemas import (
    ConceptConfigListResponse,
    ConceptDictionaryResponse,
    ConceptListResponse,
    ConceptLoadRequest,
    ConceptLoadResponse,
    ConceptManipulationRequest,
    ConceptManipulationResponse,
    ConceptPreviewRequest,
    ConceptPreviewResponse,
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
from server.services.activation_service import ActivationService
from server.services.concept_service import ConceptService
from server.services.sae_training_service import SAETrainingService
from server.storage import top_texts_dir
from server.utils import SAERegistry, generate_id, resolve_sae_path, write_json
from amber.mechanistic.sae.sae import Sae

logger = logging.getLogger(__name__)


class SAEService:
    """
    Facade service for SAE operations.
    
    Delegates to specialized services for activations, training, and concepts,
    while maintaining orchestration logic for inference and SAE management.
    """

    def __init__(
        self,
        settings: Settings,
        inference_service: InferenceService,
        job_manager: JobManager,
        activation_service: ActivationService,
        training_service: SAETrainingService,
        concept_service: ConceptService,
    ):
        self._settings = settings
        self._inference_service = inference_service
        self._job_manager = job_manager
        self._activation_service = activation_service
        self._training_service = training_service
        self._concept_service = concept_service
        self._sae_registry: Dict[str, Path] = {}
        self._sae_registry_class = SAERegistry()

    def _get_sae_class(self, name: str) -> Type[Sae]:
        """Get SAE class by name."""
        return self._sae_registry_class.get_class(name)

    def _load_sae(self, sae_class: str, path: Path) -> Sae:
        """Load an SAE from disk."""
        cls = self._get_sae_class(sae_class)
        if not hasattr(cls, "load"):
            raise ValueError(f"SAE class '{sae_class}' does not implement load()")
        return cls.load(path)

    def _resolve_sae_path(self, sae_id: Optional[str], sae_path: Optional[str]) -> Path:
        """Resolve SAE path using shared utility."""
        return resolve_sae_path(self._settings, sae_id, sae_path, self._sae_registry)

    def _apply_concept_config(self, sae: Sae, concept_config: Dict[str, Any]) -> None:
        """Apply concept configuration to an SAE."""
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

    # Delegate activation methods
    def save_activations(self, manager: ModelManager, payload: SaveActivationsRequest) -> SaveActivationsResponse:
        """Save activations from a dataset."""
        return self._activation_service.save_activations(manager, payload)

    def list_activation_runs(self, model_id: str) -> "ActivationRunListResponse":
        """List all activation runs for a model."""
        from server.schemas import ActivationRunListResponse
        return self._activation_service.list_activation_runs(model_id)

    def delete_activation_run(self, model_id: str, run_id: str) -> bool:
        """Delete an activation run."""
        return self._activation_service.delete_activation_run(model_id, run_id)

    def get_layer_size(self, activations_path: str, layer: str) -> Dict[str, Any]:
        """Get the hidden dimension of a layer from an activation run."""
        return self._activation_service.get_layer_size(activations_path, layer)

    # Delegate training methods
    def train_sae(self, manager: ModelManager, payload: TrainSAERequest) -> TrainSAEResponse:
        """Start SAE training asynchronously."""
        return self._training_service.train_sae(manager, payload)

    def train_status(self, job_id: str) -> TrainStatusResponse:
        """Get training job status."""
        return self._training_service.train_status(job_id)

    def cancel_train(self, job_id: str) -> TrainStatusResponse:
        """Cancel a training job."""
        return self._training_service.cancel_train(job_id)

    # Delegate concept methods
    def list_concepts(self, model_id: str, sae_id: str | None = None) -> ConceptListResponse:
        """List available concept files."""
        return self._concept_service.list_concepts(model_id, sae_id)

    def get_concept_dictionary(self, model_id: str, sae_id: str | None = None) -> ConceptDictionaryResponse:
        """Get the concept dictionary for an SAE."""
        return self._concept_service.get_concept_dictionary(model_id, sae_id)

    def list_concept_configs(self, model_id: str, sae_id: str | None = None) -> ConceptConfigListResponse:
        """List concept configuration files."""
        return self._concept_service.list_concept_configs(model_id, sae_id)

    def preview_concepts(self, payload: ConceptPreviewRequest) -> ConceptPreviewResponse:
        """Preview concept manipulations without saving."""
        return self._concept_service.preview_concepts(payload, self._sae_registry)

    def load_concepts(self, payload: ConceptLoadRequest) -> ConceptLoadResponse:
        """Load concepts from a file."""
        return self._concept_service.load_concepts(payload, self._sae_registry)

    def manipulate_concepts(self, payload: ConceptManipulationRequest) -> ConceptManipulationResponse:
        """Create a concept manipulation configuration."""
        result = self._concept_service.manipulate_concepts(payload, self._sae_registry)
        # Update concept configs registry
        concept_configs = self._concept_service.get_concept_configs()
        if result.sae_id and result.sae_id in concept_configs:
            # Concept service already stored it
            pass
        return result

    # SAE management methods (stay in this service)
    def list_sae_classes(self) -> Dict[str, str]:
        """Return the available SAE classes as a name -> class dotted path mapping."""
        return self._sae_registry_class.list_classes()

    def load_sae(self, payload: LoadSAERequest) -> LoadSAEResponse:
        """Load an SAE into the registry."""
        sae_path = Path(payload.sae_path)
        if not sae_path.exists():
            raise ValueError(f"sae_path '{sae_path}' does not exist")
        sae_id = sae_path.parent.name or sae_path.stem
        self._sae_registry[sae_id] = sae_path
        # Also register in training service
        self._training_service.register_sae(sae_id, sae_path)
        return LoadSAEResponse(sae_id=sae_id)

    def list_sae_runs(self, model_id: str) -> SaeRunListResponse:
        """List all SAE runs for a model."""
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
        """Get metadata for an SAE."""
        root = self._settings.artifact_base_path / "sae" / model_id / sae_id
        metadata_path = root / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Metadata not found for SAE {sae_id}")
        try:
            return json.loads(metadata_path.read_text())
        except Exception as e:
            raise ValueError(f"Failed to read metadata: {e}") from e

    def delete_sae_run(self, model_id: str, sae_id: str) -> bool:
        """Delete an SAE run."""
        import shutil
        folder = self._settings.artifact_base_path / "sae" / model_id / sae_id
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
            self._sae_registry.pop(sae_id, None)
            return True
        return False

    def _load_concept_config(self, payload: SAEInferenceRequest) -> tuple[Dict[str, Any], Path | None]:
        """Load concept configuration from path or registry."""
        concept_config = {}
        concept_path = None
        if payload.concept_config_path:
            concept_path = Path(payload.concept_config_path)
            if not concept_path.exists():
                raise ValueError(f"concept_config_path '{concept_path}' does not exist")
            concept_config = json.loads(concept_path.read_text())
        else:
            # Check concept service registry
            concept_configs = self._concept_service.get_concept_configs()
            if payload.sae_id and payload.sae_id in concept_configs:
                concept_path = concept_configs[payload.sae_id]
                concept_config = json.loads(concept_path.read_text())
        return concept_config, concept_path

    def _load_and_configure_sae(
        self, sae_path: Path, sae_class: str, sae_layer: str, lm, concept_config: Dict[str, Any]
    ) -> tuple[Sae, str]:
        """Load SAE and apply concept configuration."""
        sae_hook = self._load_sae(sae_class, sae_path)
        sae_hook.context.lm = lm
        sae_hook.context.lm_layer_signature = sae_layer

        if concept_config:
            self._apply_concept_config(sae_hook, concept_config)

        hook_id = lm.layers.register_hook(sae_layer, sae_hook)
        return sae_hook, hook_id

    def _extract_token_latents(
        self, act: torch.Tensor, payload: SAEInferenceRequest
    ) -> List[Dict[str, Any]]:
        """Extract token-level latents from activations tensor."""
        token_latents: List[Dict[str, Any]] = []
        if act.dim() == 3 and payload.return_token_latents:
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
        return token_latents

    def _extract_top_neurons(
        self, activations_tensor: torch.Tensor, payload: SAEInferenceRequest
    ) -> List[Dict[str, Any]]:
        """Extract top neurons from activations tensor."""
        top_neurons: List[Dict[str, Any]] = []
        if isinstance(activations_tensor, torch.Tensor):
            act = activations_tensor
            if act.dim() == 3:
                per_prompt = act.max(dim=1).values
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
        return top_neurons

    def _save_top_texts(
        self,
        sae_hook: Sae,
        sae_path: Path,
        sae_id: str,
        sae_layer: str,
        payload: SAEInferenceRequest,
        outputs: List,
        top_neurons: List[Dict[str, Any]],
        concept_path: Path | None,
    ) -> str | None:
        """Save top texts to disk if requested."""
        if not payload.save_top_texts:
            return None

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
        write_json(
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
        return str(top_path)

    # Inference orchestration (stays in this service)
    def infer(self, manager: ModelManager, payload: SAEInferenceRequest) -> SAEInferenceResponse:
        """Run inference with an SAE."""
        lm = manager.get_model(payload.model_id)
        sae_path = self._resolve_sae_path(payload.sae_id, payload.sae_path)
        resolved_sae_id = payload.sae_id or sae_path.parent.name
        metadata_path = sae_path.parent / "metadata.json"
        meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        sae_class = meta.get("sae_class", "TopKSae")
        sae_layer = payload.layer or meta.get("layer")
        if not sae_layer:
            raise ValueError("layer must be specified for SAE inference")
        
        # Load concept config and configure SAE
        concept_config, concept_path = self._load_concept_config(payload)
        sae_hook, hook_id = self._load_and_configure_sae(sae_path, sae_class, sae_layer, lm, concept_config)
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

        # Extract top neurons and token latents
        activations_tensor = sae_hook.tensor_metadata.get("activations")
        top_neurons = self._extract_top_neurons(activations_tensor, payload) if isinstance(activations_tensor, torch.Tensor) else []
        token_latents = self._extract_token_latents(activations_tensor, payload) if isinstance(activations_tensor, torch.Tensor) else []

        # Save top texts if requested
        sae_id_for_texts = resolved_sae_id or generate_id()
        top_texts_path = self._save_top_texts(
            sae_hook, sae_path, sae_id_for_texts, sae_layer, payload, outputs, top_neurons, concept_path
        )
        
        return SAEInferenceResponse(
            outputs=outputs,
            sae_id=resolved_sae_id,
            sae_path=str(sae_path),
            metadata_path=str(metadata_path) if metadata_path.exists() else None,
            top_neurons=top_neurons,
            token_latents=token_latents,
            top_texts_path=top_texts_path,
        )
