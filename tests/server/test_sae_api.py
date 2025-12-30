from __future__ import annotations

import pytest

# Skip entire module if server dependencies are not installed
pytest.importorskip("fastapi")

from pathlib import Path
from typing import Dict, List
import json

import torch
from fastapi.testclient import TestClient
from torch import nn
from mi_crow.mechanistic.sae.sae import Sae
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
from server.schemas import InferenceInput


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
        """
        Minimal forward pass that also simulates execution of registered
        activation hooks so that the activation extraction + training flow
        in tests closely mirrors the real system.

        The real models integrate with mi_crow's hook system; in tests we
        emulate this by manually calling `process_activations` on any
        registered detectors.
        """
        if input_ids is None:
            outputs = torch.zeros((1, 1))
        else:
            outputs = input_ids.float()

        # Simulate hook invocation for any registered layer detectors so
        # that `ActivationExtractor` can collect activations and persist
        # batches to the `LocalStore`.
        layers = getattr(self, "layers", None)
        if layers is not None:
            for hooks in getattr(layers, "_registry", {}).values():
                for _, hook in hooks:
                    process = getattr(hook, "process_activations", None)
                    if callable(process):
                        # Match the signature expected by LayerActivationDetector
                        process(self, (input_ids,), outputs)

        return outputs

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        tail = torch.tensor([[999]], device=input_ids.device)
        outputs = torch.cat([input_ids, tail], dim=1)

        # Mirror the forward-path behaviour: invoke any registered hooks so that
        # SAE hooks can capture activations during generation, which is what the
        # inference service uses.
        layers = getattr(self, "layers", None)
        if layers is not None:
            for hooks in getattr(layers, "_registry", {}).values():
                for _, hook in hooks:
                    process = getattr(hook, "process_activations", None)
                    if callable(process):
                        process(self, (input_ids,), outputs)

        return outputs


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
        # Expose layers on the model so that the dummy forward pass can
        # find and invoke any registered hooks. This mirrors how the real
        # language model wires hooks into underlying PyTorch modules.
        self.model.layers = self.layers
        self._input_tracker = None
        self.inference = DummyInference()
        self.model_id = "dummy-model"

    def _ensure_input_tracker(self):
        """
        Minimal stand-in for the real LanguageModel._ensure_input_tracker.

        Tests only require that this exists and returns an object with an
        .enable() method so that SAE text tracking can be toggled without
        touching the full tracking pipeline.
        """

        class _DummyTracker:
            def __init__(self):
                self.enabled = False

            def enable(self):
                self.enabled = True

            def disable(self):
                self.enabled = False

        if self._input_tracker is None:
            self._input_tracker = _DummyTracker()
        return self._input_tracker


class DummySae(Sae):
    def __init__(self, n_latents: int = 4, n_inputs: int = 4, *args, **kwargs):
        self._n_latents_init = n_latents
        super().__init__(n_latents=n_latents, n_inputs=n_inputs, *args, **kwargs)

    def _initialize_sae_engine(self):
        return nn.Linear(self.context.n_inputs, self.context.n_inputs, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def save(self, name: str, path: str | Path | None = None):
        path = Path(path or Path.cwd())
        path.mkdir(parents=True, exist_ok=True)
        save_path = path / f"{name}.pt"
        torch.save({"n_latents": self.context.n_latents, "n_inputs": self.context.n_inputs}, save_path)
        return save_path

    @staticmethod
    def load(path: Path):
        payload = torch.load(path, map_location="cpu")
        return DummySae(n_latents=int(payload.get("n_latents", 4)), n_inputs=int(payload.get("n_inputs", 4)))

    def process_activations(self, module, input, output):
        # For tests we just record a simple activations tensor so that the SAE
        # inference pipeline can compute top neurons without depending on real
        # model behaviour.
        self.tensor_metadata["activations"] = torch.ones((1, self.context.n_latents))
        return None

    def modify_activations(self, module, inputs: torch.Tensor | None, output: torch.Tensor | None):
        # Produce simple activations tensor for summary
        self.tensor_metadata["activations"] = torch.ones((1, self.context.n_latents))
        return output

    def train(self, store: LocalStore, run_id: str, layer_signature: str | int, config=None, training_run_id: str | None = None):
        return {"history": {"loss": [0.0]}, "training_run_id": training_run_id or "dummy"}


class FakeModelManager:
    def __init__(self):
        self.allowed = {
            "bielik": "bielik/bielik-7b",
            "plum": "plum-2b",
            "tinylm": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        }
        self._loaded: Dict[str, DummyLanguageModel] = {}

    def list_models(self):
        return []

    def load_model(self, model_id: str):
        if model_id not in self.allowed:
            raise ValueError(f"model_id '{model_id}' not allowed")
        self._loaded.setdefault(model_id, DummyLanguageModel())
        return {"id": model_id, "name": self.allowed[model_id], "status": "loaded"}

    def unload_model(self, model_id: str):
        self._loaded.pop(model_id, None)

    def get_model(self, model_id: str) -> DummyLanguageModel:
        if model_id not in self._loaded:
            raise ValueError(f"model_id '{model_id}' is not loaded")
        return self._loaded[model_id]

    def get_layer_tree(self, model_id: str):
        return []


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
    fake_manager = FakeModelManager()
    hook_factory = get_hook_factory()
    inference_service = InferenceService(hook_factory=hook_factory)
    job_manager = get_job_manager()
    
    # Create the new services
    from server.services.activation_service import ActivationService
    from server.services.sae_training_service import SAETrainingService
    from server.services.concept_service import ConceptService
    
    activation_service = ActivationService(settings=settings)
    training_service = SAETrainingService(settings=settings, job_manager=job_manager)
    concept_service = ConceptService(settings=settings)
    
    # Create SAE service with all dependencies
    sae_service = SAEService(
        settings=settings,
        inference_service=inference_service,
        job_manager=job_manager,
        activation_service=activation_service,
        training_service=training_service,
        concept_service=concept_service,
    )
    sae_service._sae_registry_class.register("DummySae", DummySae)
    # Also register in training service
    training_service._sae_registry_class.register("DummySae", DummySae)
    concept_service._sae_registry.register("DummySae", DummySae)
    
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_model_manager] = lambda: fake_manager
    app.dependency_overrides[get_inference_service] = lambda: inference_service
    app.dependency_overrides[get_activation_service] = lambda: activation_service
    app.dependency_overrides[get_sae_training_service] = lambda: training_service
    app.dependency_overrides[get_concept_service] = lambda: concept_service
    app.dependency_overrides[get_sae_service] = lambda: sae_service
    return app, fake_manager, settings


@pytest.fixture()
def client(tmp_path):
    app, manager, settings = setup_app(tmp_path)
    client = TestClient(app)
    client.manager = manager
    client.settings = settings
    return client


def test_sae_flow(client: TestClient, tmp_path: Path):
    client.post("/models/load", json={"model_id": "bielik"})
    data_file = tmp_path / "data.txt"
    data_file.write_text("hello world\nsecond prompt\n")
    save_resp = client.post(
        "/sae/activations/save",
        json={
            "model_id": "bielik",
            "layers": ["dummy_root"],
            "dataset": {"type": "local", "paths": [str(data_file)]},
            "sample_limit": 2,
            "batch_size": 1,
        },
    )
    assert save_resp.status_code == 200
    save_payload = save_resp.json()
    path = Path(save_payload["path"])
    assert path.exists()
    run_id = save_payload["run_id"]

    train_resp = client.post(
        "/sae/train",
        json={
            "model_id": "bielik",
            "activations_path": str(path),
            "layer": "dummy_root",
            "sae_class": "DummySae",
            "n_latents": 4,
            "hyperparams": {"epochs": 1},
            "run_id": run_id,
        },
    )
    assert train_resp.status_code == 200
    job_id = train_resp.json()["job_id"]

    status = client.get(f"/sae/train/status/{job_id}").json()
    assert status["status"] in {"pending", "running", "completed"}
    # jobs are fast (stub); fetch final status
    status = client.get(f"/sae/train/status/{job_id}").json()
    assert status["status"] == "completed"
    sae_id = status["sae_id"]
    sae_path = Path(status["sae_path"])
    assert sae_path.exists()
    if status.get("metadata_path"):
        assert Path(status["metadata_path"]).exists()

    load_resp = client.post("/sae/load", json={"model_id": "bielik", "sae_path": str(sae_path)})
    assert load_resp.status_code == 200
    loaded_id = load_resp.json()["sae_id"]
    assert loaded_id == sae_id

    concepts_resp = client.get("/sae/concepts", params={"model_id": "bielik", "sae_id": sae_id})
    assert concepts_resp.status_code == 200

    concept_file = tmp_path / "concepts.json"
    concept_file.write_text(json.dumps({"0": {"name": "x", "score": 1.0}}))
    load_concept = client.post(
        "/sae/concepts/load",
        json={"model_id": "bielik", "sae_id": sae_id, "source_path": str(concept_file)},
    )
    assert load_concept.status_code == 200

    manip_resp = client.post(
        "/sae/concepts/manipulate",
        json={"model_id": "bielik", "sae_id": sae_id, "edits": {"0": 1.5}},
    )
    assert manip_resp.status_code == 200
    config_payload = manip_resp.json()
    config_path = Path(config_payload["concept_config_path"])
    assert config_path.exists()

    infer_resp = client.post(
        "/sae/infer",
        json={
            "model_id": "bielik",
            "sae_id": sae_id,
            "save_top_texts": True,
            "concept_config_path": str(config_path),
            "track_texts": True,
            "top_k_neurons": 3,
            "inputs": [InferenceInput(prompt="hi").model_dump()],
        },
    )
    # Include response body on failure to aid debugging when this high-level
    # integration flow breaks.
    assert infer_resp.status_code == 200, infer_resp.text
    infer_json = infer_resp.json()
    assert len(infer_json["outputs"]) == 1
    assert infer_json["sae_id"] == sae_id
    assert infer_json["top_neurons"]
    top_path = infer_json["top_texts_path"]
    assert top_path is not None
    assert Path(top_path).exists()
