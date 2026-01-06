from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mi_crow.store.store import Store
from mi_crow.utils import get_logger

logger = get_logger(__name__)


JsonDict = Dict[str, Any]


class Predictor(ABC):
    """Experiment-layer abstraction for *storing* predictions.

    This is intentionally not part of `src/mi_crow`.

    - Accumulates batch predictions in-memory.
    - Saves a single artifact per run under: `store.base_path / "runs" / run_id / predictions.*`

    Notes on schema:
    - Keep a stable set of top-level keys for analysis.
    - Put model-specific detail into `extra_json` (string) to avoid Parquet schema issues.
    """

    def __init__(self, model_id: Optional[str] = None, config: Optional[JsonDict] = None):
        self.model_id = model_id or self.__class__.__name__
        self.config: JsonDict = config or {}

        self.predictions: List[JsonDict] = []
        self.run_metadata: JsonDict = {
            "model_id": self.model_id,
            "config": self.config,
            "timestamp_start": None,
            "timestamp_end": None,
            "num_samples": 0,
        }

    def clear_predictions(self) -> None:
        self.predictions = []
        self.run_metadata["timestamp_start"] = datetime.now().isoformat()
        self.run_metadata["timestamp_end"] = None
        self.run_metadata["num_samples"] = 0

    def add_predictions(self, batch_predictions: Iterable[JsonDict]) -> None:
        batch_list = list(batch_predictions)
        self.predictions.extend(batch_list)
        self.run_metadata["num_samples"] = len(self.predictions)

    def finalize_predictions(self) -> None:
        self.run_metadata["timestamp_end"] = datetime.now().isoformat()

    def save_predictions(self, run_id: str, store: Store, format: str = "parquet") -> Path:
        if not self.predictions:
            raise ValueError("No predictions to save. Call predict first.")

        self.finalize_predictions()

        output_dir = Path(store.base_path) / "runs" / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        meta_path = output_dir / "meta.json"
        meta_path.write_text(json.dumps(self.run_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        if format == "parquet":
            try:
                output_path = self._save_parquet(output_dir)
            except ImportError:
                logger.warning("pyarrow not installed; falling back to JSON predictions.")
                output_path = self._save_json(output_dir)
        elif format == "json":
            output_path = self._save_json(output_dir)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved {len(self.predictions)} predictions to {output_path}")
        return output_path

    @staticmethod
    def load_predictions(run_id: str, store: Store, format: str = "auto") -> Tuple[List[JsonDict], JsonDict]:
        run_dir = Path(store.base_path) / "runs" / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")

        meta_path = run_dir / "meta.json"
        metadata: JsonDict = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        if format == "auto":
            if (run_dir / "predictions.parquet").exists():
                format = "parquet"
            elif (run_dir / "predictions.json").exists():
                format = "json"
            else:
                raise FileNotFoundError(f"No predictions file found in {run_dir}")

        if format == "parquet":
            predictions = Predictor._load_parquet(run_dir)
        elif format == "json":
            predictions = Predictor._load_json(run_dir)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return predictions, metadata

    def _save_parquet(self, output_dir: Path) -> Path:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError("pyarrow is required for Parquet format") from e

        table = pa.Table.from_pylist(self.predictions)
        output_path = output_dir / "predictions.parquet"
        pq.write_table(table, output_path, compression="snappy")
        return output_path

    @staticmethod
    def _load_parquet(run_dir: Path) -> List[JsonDict]:
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError("pyarrow is required for Parquet format") from e

        parquet_path = run_dir / "predictions.parquet"
        table = pq.read_table(parquet_path)
        return table.to_pylist()

    def _save_json(self, output_dir: Path) -> Path:
        output_path = output_dir / "predictions.json"
        output_path.write_text(
            json.dumps({"predictions": self.predictions, "metadata": self.run_metadata}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    @staticmethod
    def _load_json(run_dir: Path) -> List[JsonDict]:
        json_path = run_dir / "predictions.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "predictions" in data:
            return data["predictions"]
        if isinstance(data, list):
            return data
        raise ValueError(f"Unexpected JSON structure in {json_path}")

    @abstractmethod
    def get_config(self) -> JsonDict:
        raise NotImplementedError
