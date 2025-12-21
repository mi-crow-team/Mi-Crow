from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Type

from server.config import Settings
from server.schemas import (
    ConceptConfigInfo,
    ConceptConfigListResponse,
    ConceptDictionaryResponse,
    ConceptListResponse,
    ConceptLoadRequest,
    ConceptLoadResponse,
    ConceptManipulationRequest,
    ConceptManipulationResponse,
    ConceptPreviewRequest,
    ConceptPreviewResponse,
)
from server.storage import concepts_dir
from server.utils import SAERegistry, generate_id, write_json
from amber.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary
from amber.mechanistic.sae.sae import Sae

logger = logging.getLogger(__name__)


class ConceptService:
    """Service for managing SAE concepts."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._concept_configs: Dict[str, Path] = {}
        self._sae_registry = SAERegistry()

    def _get_sae_class(self, name: str) -> Type[Sae]:
        """Get SAE class by name."""
        return self._sae_registry.get_class(name)

    def _load_sae(self, sae_class: str, path: Path) -> Sae:
        """Load an SAE from disk."""
        cls = self._get_sae_class(sae_class)
        if not hasattr(cls, "load"):
            raise ValueError(f"SAE class '{sae_class}' does not implement load()")
        return cls.load(path)

    def _resolve_sae_path(self, sae_id: Optional[str], sae_path: Optional[str], sae_registry: Dict[str, Path]) -> Path:
        """Resolve SAE path from ID or direct path."""
        if sae_path:
            return Path(sae_path)
        if sae_id and sae_id in sae_registry:
            return sae_registry[sae_id]
        raise ValueError("SAE not loaded; provide sae_path or load first")

    def list_concepts(self, model_id: str, sae_id: str | None = None) -> ConceptListResponse:
        """List available concept files."""
        folder = concepts_dir(self._settings.artifact_base_path, model_id, sae_id)
        concepts = []
        for item in folder.glob("*.json"):
            concepts.append(item.name)
        return ConceptListResponse(base_path=str(folder), concepts=sorted(concepts))

    def get_concept_dictionary(self, model_id: str, sae_id: str | None = None) -> ConceptDictionaryResponse:
        """Get the concept dictionary for an SAE."""
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
        """List concept configuration files."""
        folder = concepts_dir(self._settings.artifact_base_path, model_id, sae_id)
        configs: List[ConceptConfigInfo] = []
        for item in folder.glob("concepts_*.json"):
            meta: Dict = {}
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

    def preview_concepts(
        self,
        payload: ConceptPreviewRequest,
        sae_registry: Dict[str, Path],
    ) -> ConceptPreviewResponse:
        """Preview concept manipulations without saving."""
        sae_path = self._resolve_sae_path(payload.sae_id, payload.sae_path, sae_registry)
        sae_id = payload.sae_id or sae_path.parent.name or generate_id()
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

    def load_concepts(
        self,
        payload: ConceptLoadRequest,
        sae_registry: Dict[str, Path],
    ) -> ConceptLoadResponse:
        """Load concepts from a file."""
        src = Path(payload.source_path)
        if not src.exists():
            raise ValueError(f"source_path '{src}' does not exist")
        
        sae_path = self._resolve_sae_path(payload.sae_id, None, sae_registry)
        sae_meta_path = sae_path.parent / "metadata.json"
        sae_meta = json.loads(sae_meta_path.read_text()) if sae_meta_path.exists() else {}
        sae_class = sae_meta.get("sae_class", "TopKSae")
        sae_hook = self._load_sae(sae_class, sae_path)
        n_latents = sae_hook.context.n_latents
        
        if src.suffix.lower() == ".csv":
            concept_dict = ConceptDictionary.from_csv(src, n_size=n_latents)
        else:
            concept_dict = ConceptDictionary.from_json(src, n_size=n_latents)
        
        sae_id = payload.sae_id or sae_path.parent.name or generate_id()
        dst_dir = concepts_dir(self._settings.artifact_base_path, payload.model_id, sae_id)
        concept_dict.set_directory(dst_dir)
        saved_path = concept_dict.save()
        concept_id = generate_id()
        
        return ConceptLoadResponse(concept_id=concept_id, stored_path=str(saved_path))

    def manipulate_concepts(
        self,
        payload: ConceptManipulationRequest,
        sae_registry: Dict[str, Path],
    ) -> ConceptManipulationResponse:
        """Create a concept manipulation configuration."""
        sae_path = self._resolve_sae_path(payload.sae_id, payload.sae_path, sae_registry)
        sae_id = payload.sae_id or sae_path.parent.name or generate_id()
        sae_meta_path = sae_path.parent / "metadata.json"
        sae_meta = json.loads(sae_meta_path.read_text()) if sae_meta_path.exists() else {}
        sae_class = sae_meta.get("sae_class", "TopKSae")
        sae_hook = self._load_sae(sae_class, sae_path)
        n_latents = sae_hook.context.n_latents
        
        # Validate indices
        for idx in payload.edits.keys():
            if int(idx) < 0 or int(idx) >= n_latents:
                raise ValueError(f"edit index {idx} out of bounds for n_latents={n_latents}")
        for idx in payload.bias.keys():
            if int(idx) < 0 or int(idx) >= n_latents:
                raise ValueError(f"bias index {idx} out of bounds for n_latents={n_latents}")
        
        folder = concepts_dir(self._settings.artifact_base_path, payload.model_id, sae_id)
        config_path = folder / f"concepts_{generate_id()}.json"
        write_json(
            config_path,
            {
                "sae_path": str(sae_path),
                "sae_class": sae_class,
                "layer": sae_meta.get("layer"),
                "weights": payload.edits,
                "bias": payload.bias,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self._concept_configs[sae_id] = config_path
        
        return ConceptManipulationResponse(concept_config_path=str(config_path), sae_id=sae_id)

    def get_concept_configs(self) -> Dict[str, Path]:
        """Get the concept configs registry."""
        return self._concept_configs

