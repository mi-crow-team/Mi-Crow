from __future__ import annotations

from typing import Any, Dict, List

import torch

from experiments.baselines.guard_model import GuardModel
from experiments.predictors.baseline_model import BaselineModel
from experiments.predictors.predictor import Predictor


class _DummyPredictor(Predictor):
    def get_config(self) -> Dict[str, Any]:
        return dict(self.config)


class _DummyBaseline(BaselineModel):
    def __init__(self):
        super().__init__(model_id="dummy", config={})

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [{"predicted_label": 0} for _ in texts]

    def get_config(self) -> Dict[str, Any]:
        return dict(self.config)


class _FakeGuardAdapter:
    adapter_id = "fake_guard"

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [
            {
                "predicted_label": 1,
                "score_safe": None,
                "score_unsafe": None,
                "threat_category": "S3",
                "raw_output": "UNSAFE S3",
                "extra_json": "{}",
            }
            for _ in texts
        ]


def test_predictor_save_and_load_json(temp_store):
    p = _DummyPredictor(model_id="dummy", config={"a": 1})
    p.clear_predictions()
    p.add_predictions([{"predicted_label": 1}, {"predicted_label": 0}])

    path = p.save_predictions(run_id="unit_dummy", store=temp_store, format="json")
    assert path.name == "predictions.json"

    preds, meta = Predictor.load_predictions(run_id="unit_dummy", store=temp_store, format="json")
    assert len(preds) == 2
    assert meta["model_id"] == "dummy"


def test_baseline_model_attaches_sample_index(sample_classification_dataset):
    m = _DummyBaseline()
    m.predict_dataset(sample_classification_dataset, batch_size=2, verbose=False, text_field="text")
    assert len(m.predictions) == len(sample_classification_dataset)
    assert m.predictions[0]["sample_index"] == 0


def test_guard_model_normalizes_adapter_output():
    guard = GuardModel(adapter=_FakeGuardAdapter())
    out = guard.predict_batch(["x", "y"])
    assert out[0]["predicted_label"] == 1
    assert out[0]["threat_category"] == "S3"


def test_lpm_predictor_accumulates_binary_predictions():
    from experiments.models.lpm.lpm import LPM

    lpm = LPM(layer_signature="x", distance_metric="euclidean", device="cpu", positive_label="harmful")

    # Provide simple prototypes
    hidden = 4
    lpm.lpm_context.prototypes = {
        "safe": torch.zeros(hidden),
        "harmful": torch.ones(hidden),
    }

    lpm.clear_predictions()

    # Two vectors closer to harmful
    output = torch.ones(2, hidden) * 0.9
    lpm.process_activations(module=torch.nn.Identity(), input=(), output=output)

    assert len(lpm.predictions) == 2
    assert all(p["predicted_label"] == 1 for p in lpm.predictions)
    assert lpm.predictions[0]["sample_index"] == 0
