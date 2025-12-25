from __future__ import annotations

from typing import Any, Dict, List, Optional

from experiments.baselines.guard_adapters import GuardAdapter
from experiments.predictors.baseline_model import BaselineModel


class GuardModel(BaselineModel):
    """Baseline wrapper around a `GuardAdapter`.

    Provides a unified binary classification output across different guard families.
    """

    def __init__(
        self,
        adapter: GuardAdapter,
        model_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.adapter = adapter
        merged_config = {"adapter_id": adapter.adapter_id}
        if config:
            merged_config.update(config)

        super().__init__(model_id=model_id or adapter.adapter_id.replace(":", "_"), config=merged_config)

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        preds = self.adapter.predict_batch(texts)

        # Ensure minimal schema fields exist
        normalized: List[Dict[str, Any]] = []
        for p in preds:
            normalized.append(
                {
                    "predicted_label": int(p.get("predicted_label")),
                    "score_safe": p.get("score_safe"),
                    "score_unsafe": p.get("score_unsafe"),
                    "threat_category": p.get("threat_category"),
                    "raw_output": p.get("raw_output"),
                    "extra_json": p.get("extra_json"),
                }
            )
        return normalized

    def get_config(self) -> Dict[str, Any]:
        return dict(self.config)
