from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List

from tqdm.auto import tqdm

from .predictor import Predictor


class BaselineModel(Predictor):
    """Predictor for baselines that run directly on text (no hooks)."""

    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def predict_dataset(self, dataset, batch_size: int = 32, verbose: bool = True, text_field: str = "text") -> None:
        self.clear_predictions()

        indices = range(0, len(dataset), batch_size)
        iterator = tqdm(indices, desc=f"Predicting ({self.model_id})") if verbose else indices

        for start in iterator:
            end = min(start + batch_size, len(dataset))
            texts = [dataset[i][text_field] for i in range(start, end)]
            batch_predictions = self.predict_batch(texts)

            # Attach stable indices for later evaluation joins.
            for offset, pred in enumerate(batch_predictions):
                if isinstance(pred, dict) and "sample_index" not in pred:
                    pred["sample_index"] = start + offset

            self.add_predictions(batch_predictions)
