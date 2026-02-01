from __future__ import annotations
import pytest
pytest.importorskip("fastapi")


import json
from pathlib import Path
from typing import List
import torch
from fastapi.testclient import TestClient
from torch import nn
from mi_crow.store.local_store import LocalStore
from server.config import Settings
from server.dependencies import (
    get_activation_service,
    get_concept_service,
    get_hook_factory,
    get_inference_service,
    get_job_manager,
    get_model_manager,
    get_sae_service,
    get_sae_training_service,
    get_settings,
)
from server.inference_service import InferenceService
from server.main import create_app
from server.sae_service import SAEService


class DummyTokenizer:
    def __call__(self, texts: List[str], return_tensors: str = "pt", padding: bool = True, truncation: bool = True):
        input_ids = []
        for text in texts:
            ids = list(range(1, len(text.split()) + 2))
            input_ids.append(ids)

        tensor = torch.tensor(input_ids, dtype=torch.long)
        return {"input_ids": tensor}

    def batch_decode(self, outputs, skip_special_tokens: bool = True):
        return [" ".join(self.decode([tid]) for tid in row.tolist()) for row in outputs]

    def decode(self, token_ids):
        token_ids = token_ids if isinstance(token_ids, list) else [token_ids]
        return "tok" + "-".join(str(tid) for tid in token_ids)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            return torch.zeros((1, 1))

        return input_ids.float()

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        tail = torch.tensor([[999]], device=input_ids.device)
        return torch.cat([input_ids, tail], dim=1)


class DummyInference:
    def extract_logits(self, outputs):
        batch, seq_len = outputs.shape
        return torch.ones((batch, seq_len, 4), dtype=torch.float32)


class DummyLayers:
    def __init__(self):
        self.name_to_layer = {"dummy_root": object()}
        self._registry = {}

    def register_hook(self, layer_signature, hook):
        hook_id = f"hook-{len(self._registry) + 1}"
        hooks = self._registry.setdefault(layer_signature, [])
        hooks.append((hook_id, hook))
        return hook_id

    def unregister_hook(self, hook_id):
        for key, hooks in list(self._registry.items()):
            filtered = [entry for entry in hooks if entry[0] != hook_id]
            if filtered:
                self._registry[key] = filtered

            else:
                self._registry.pop(key, None)

    def get_hooks(self, layer_signature, hook_type=None):
        return [entry[1] for entry in self._registry.get(layer_signature, [])]


class DummyLanguageModel:
    def __init__(self):
        self.model = DummyModel()
        self.tokenizer = DummyTokenizer()
        self.layers = DummyLayers()
        self.inference = DummyInference()
        self.model_id = "dummy-model"


class FakeModelManager:
    def __init__(self):
        self.allowed = {"bielik": "bielik/bielik-7b"}
        self._loaded = {}

    def get_model(self, model_id: str):
        if model_id not in self._loaded:
            if model_id not in self.allowed:
                raise ValueError("not allowed")

            self._loaded[model_id] = DummyLanguageModel()

        return self._loaded[model_id]


def setup_app(tmp_path: Path):
    get_settings.cache_clear()
    get_model_manager.cache_clear()
    get_inference_service.cache_clear()
    get_job_manager.cache_clear()
    get_activation_service.cache_clear()
    get_sae_training_service.cache_clear()
    get_concept_service.cache_clear()
    get_sae_service.cache_clear()
    settings = Settings(artifact_base_path=tmp_path)
    app = create_app()
    manager = FakeModelManager()
    hook_factory = get_hook_factory()
    inference_service = InferenceService(hook_factory=hook_factory)
    job_manager = get_job_manager()
    from server.services.activation_service import ActivationService
    from server.services.concept_service import ConceptService
    from server.services.sae_training_service import SAETrainingService
    activation_service = ActivationService(settings=settings)
    training_service = SAETrainingService(settings=settings, job_manager=job_manager)
    concept_service = ConceptService(settings=settings)
    sae_service = SAEService(
        settings=settings,
        inference_service=inference_service,
        job_manager=job_manager,
        activation_service=activation_service,
        training_service=training_service,
        concept_service=concept_service,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_model_manager] = lambda: manager
    app.dependency_overrides[get_inference_service] = lambda: inference_service
    app.dependency_overrides[get_activation_service] = lambda: activation_service
    app.dependency_overrides[get_sae_training_service] = lambda: training_service
    app.dependency_overrides[get_concept_service] = lambda: concept_service
    app.dependency_overrides[get_sae_service] = lambda: sae_service
    return app, manager


def test_local_activation_save(tmp_path: Path, monkeypatch):
    data_file = tmp_path / "data.txt"
    data_file.write_text("hello world\nsecond line\n")
    app, _ = setup_app(tmp_path)
    from server.activation_extractor import ActivationExtractor
    def _fake_extract(self, texts, out_dir, limit=None, *, store=None, run_id=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        batches = []
        if store is not None and run_id is not None:
            activations = torch.ones((2, 1, 4), dtype=torch.float32)
            tensor_metadata = {"dummy_root": {"activations": activations}}
            metadata = {
                "texts": ["hello world", "second line"],
                "token_counts": [2, 2],
                "layers": self.layers,
            }
            store_key = store.put_detector_metadata(
                run_id=run_id,
                batch_index=0,
                metadata=metadata,
                tensor_metadata=tensor_metadata,
            )
            batches.append(
                {
                    "batch_index": 0,
                    "size": 2,
                    "token_counts": [2, 2],
                    "store_key": store_key,
                }
            )

        manifest_path = out_dir / "manifest.json"
        return {
            "samples": 2,
            "tokens": 4,
            "shards": [],
            "batches": batches,
            "manifest_path": str(manifest_path),
        }

    monkeypatch.setattr(
        ActivationExtractor,
        "extract",
        _fake_extract,
    )
    client = TestClient(app)
    resp = client.post(
        "/sae/activations/save",
        json={
            "model_id": "bielik",
            "layers": ["dummy_root"],
            "dataset": {"type": "local", "paths": [str(data_file)]},
            "sample_limit": 2,
            "batch_size": 1,
            "shard_size": 2,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    manifest_path = Path(payload["path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["samples"] == 2
    assert payload["run_id"]
    store = LocalStore(base_path=manifest_path.parent)
    batches = store.list_run_batches(payload["run_id"])
    assert batches
    meta, tensor_meta = store.get_detector_metadata(payload["run_id"], batches[0])
    assert "token_counts" in meta
    assert tensor_meta
