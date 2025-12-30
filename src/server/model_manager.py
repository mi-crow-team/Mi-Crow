from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from mi_crow.language_model.language_model import LanguageModel
from mi_crow.store.local_store import LocalStore

from server.config import Settings
from server.schemas import LayerInfo, ModelInfo


@dataclass
class ManagedModel:
    model_id: str
    hf_id: str
    language_model: LanguageModel


LoaderFn = Callable[[str, str, Optional[str]], LanguageModel]


def default_loader(model_id: str, hf_id: str, hf_token: str | None) -> LanguageModel:
    store_path = Path.home() / ".cache" / "mi_crow_store" / model_id
    store_path.mkdir(parents=True, exist_ok=True)
    store = LocalStore(base_path=store_path)
    tokenizer_params = {"token": hf_token} if hf_token else {}
    model_params: Dict = {"token": hf_token} if hf_token else {}
    return LanguageModel.from_huggingface(
        model_name=hf_id,
        store=store,
        tokenizer_params=tokenizer_params,
        model_params=model_params,
    )


class ModelManager:
    """Manage a limited set of HF-hosted models."""

    def __init__(self, settings: Settings, loader: LoaderFn | None = None):
        self._settings = settings
        self._loader: LoaderFn = loader or default_loader
        self._models: Dict[str, ManagedModel] = {}
        self._lock = threading.Lock()

    def list_models(self) -> List[ModelInfo]:
        with self._lock:
            return [
                ModelInfo(
                    id=model_id,
                    name=hf_id,
                    status="loaded" if model_id in self._models else "available",
                )
                for model_id, hf_id in self._settings.allowed_models.items()
            ]

    def load_model(self, model_id: str) -> ModelInfo:
        with self._lock:
            if model_id in self._models:
                lm = self._models[model_id].language_model
                return ModelInfo(id=model_id, name=lm.model_id, status="loaded")

            if model_id not in self._settings.allowed_models:
                raise ValueError(f"model_id '{model_id}' not allowed")

            hf_id = self._settings.allowed_models[model_id]
            lm = self._loader(model_id, hf_id, self._settings.hf_token)
            self._models[model_id] = ManagedModel(model_id=model_id, hf_id=hf_id, language_model=lm)
            return ModelInfo(id=model_id, name=hf_id, status="loaded")

    def unload_model(self, model_id: str) -> None:
        with self._lock:
            self._models.pop(model_id, None)

    def get_model(self, model_id: str) -> LanguageModel:
        with self._lock:
            if model_id not in self._models:
                raise ValueError(f"model_id '{model_id}' is not loaded")
            return self._models[model_id].language_model

    def get_layer_tree(self, model_id: str) -> List[LayerInfo]:
        lm = self.get_model(model_id)
        layers = lm.layers
        layer_infos: List[LayerInfo] = []

        for name, layer in layers.name_to_layer.items():
            path = name.split("_")
            layer_infos.append(
                LayerInfo(
                    layer_id=name,
                    name=name,
                    type=layer.__class__.__name__,
                    path=path,
                )
            )

        return layer_infos

    def available_hook_names(self, hook_classes: Iterable[type]) -> List[str]:
        return sorted({cls.__name__ for cls in hook_classes})
