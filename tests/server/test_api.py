from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from fastapi.testclient import TestClient
from torch import nn

from server.dependencies import get_hook_factory, get_inference_service, get_model_manager, get_job_manager, get_settings, get_sae_service
from server.main import create_app
from server.schemas import HookPayload, InferenceInput, InferenceRequest, LayerInfo, ModelInfo


class DummyTokenizer:
    def __call__(self, texts: List[str], return_tensors: str = "pt"):
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
        self.name_to_layer = {"dummy_root": object(), "dummy_leaf": object()}
        self._registry: Dict[str, List] = {}

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
        self.allowed = {
            "bielik": "bielik/bielik-7b",
            "plum": "plum-2b",
            "tinylm": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        }
        self._loaded: Dict[str, DummyLanguageModel] = {}

    def list_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                id=model_id,
                name=hf_id,
                status="loaded" if model_id in self._loaded else "available",
            )
            for model_id, hf_id in self.allowed.items()
        ]

    def load_model(self, model_id: str) -> ModelInfo:
        if model_id not in self.allowed:
            raise ValueError(f"model_id '{model_id}' not allowed")
        self._loaded.setdefault(model_id, DummyLanguageModel())
        return ModelInfo(id=model_id, name=self.allowed[model_id], status="loaded")

    def unload_model(self, model_id: str) -> None:
        self._loaded.pop(model_id, None)

    def get_model(self, model_id: str) -> DummyLanguageModel:
        if model_id not in self._loaded:
            raise ValueError(f"model_id '{model_id}' is not loaded")
        return self._loaded[model_id]

    def get_layer_tree(self, model_id: str) -> List[LayerInfo]:
        if model_id not in self._loaded:
            raise ValueError(f"model_id '{model_id}' is not loaded")
        return [
            LayerInfo(layer_id="dummy_root", name="dummy_root", type="Dummy", path=["dummy", "root"]),
            LayerInfo(layer_id="dummy_leaf", name="dummy_leaf", type="Dummy", path=["dummy", "leaf"]),
        ]


def setup_test_app():
    get_settings.cache_clear()
    get_model_manager.cache_clear()
    get_inference_service.cache_clear()
    get_job_manager.cache_clear()
    get_sae_service.cache_clear()
    app = create_app()
    fake_manager = FakeModelManager()
    hook_factory = get_hook_factory()
    inference_service = get_inference_service()
    app.dependency_overrides[get_model_manager] = lambda: fake_manager
    app.dependency_overrides[get_hook_factory] = lambda: hook_factory
    app.dependency_overrides[get_inference_service] = lambda: inference_service
    return app, fake_manager


def build_inference_payload(model_id: str, hooks: Iterable[HookPayload] | None = None):
    hooks_list = list(hooks) if hooks else []
    inputs = [
        InferenceInput(
            prompt="hello world",
            hooks=hooks_list,
        )
    ]
    return InferenceRequest(model_id=model_id, inputs=inputs)


def test_list_and_load_models():
    app, manager = setup_test_app()
    client = TestClient(app)

    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert {item["id"] for item in data} == set(manager.allowed.keys())

    load_resp = client.post("/models/load", json={"model_id": "bielik"})
    assert load_resp.status_code == 200
    assert load_resp.json()["status"] == "loaded"


def test_get_layers_requires_loaded_model():
    app, _ = setup_test_app()
    client = TestClient(app)

    resp = client.get("/models/bielik/layers")
    assert resp.status_code == 400

    client.post("/models/load", json={"model_id": "bielik"})
    resp_ok = client.get("/models/bielik/layers")
    assert resp_ok.status_code == 200
    assert len(resp_ok.json()) > 0


def test_inference_with_hooks_and_logits():
    app, _ = setup_test_app()
    client = TestClient(app)
    client.post("/models/load", json={"model_id": "bielik"})

    payload = build_inference_payload(
        "bielik",
        hooks=[
            HookPayload(hook_name="LayerActivationDetector", layer_id="dummy_root"),
            HookPayload(hook_name="NeuronMultiplierController", layer_id="dummy_root", config={"weights": {"0": 2.0}}),
        ],
    )
    resp = client.post("/inference", json=payload.model_dump())
    assert resp.status_code == 200
    output = resp.json()["outputs"][0]
    assert output["text"]
    assert output["hooks"]


def test_inference_compare_mode():
    app, _ = setup_test_app()
    client = TestClient(app)
    client.post("/models/load", json={"model_id": "bielik"})

    inputs = [
        InferenceInput(prompt="one").model_dump(),
        InferenceInput(prompt="two").model_dump(),
    ]
    resp = client.post("/inference", json={"model_id": "bielik", "inputs": inputs})
    assert resp.status_code == 200
    outputs = resp.json()["outputs"]
    assert len(outputs) == 2


def test_invalid_hook_name_returns_error():
    app, _ = setup_test_app()
    client = TestClient(app)
    client.post("/models/load", json={"model_id": "bielik"})

    payload = build_inference_payload(
        "bielik",
        hooks=[HookPayload(hook_name="MissingHook", layer_id="dummy_root")],
    )
    resp = client.post("/inference", json=payload.model_dump())
    assert resp.status_code == 400
