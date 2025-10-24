from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Dict, Any
import json

import safetensors.torch as storch
import torch


class Store:
    """Abstract store optimized for tensor batches grouped by run_id.

    This interface intentionally excludes generic bytes/JSON APIs.
    Implementations should focus on efficient safetensors-backed IO.
    """

    # Configurable base prefix for run storage (e.g., "runs" by default)
    runs_prefix: str = "activations"

    # --- Run-oriented batch APIs ---
    def _run_batch_key(self, run_id: str, batch_index: int) -> str:
        return f"{self.runs_prefix}/{run_id}/batch_{batch_index:06d}.safetensors"

    def put_run_batch(self, run_id: str, batch_index: int,
                      tensors: List[torch.Tensor] | Dict[str, torch.Tensor]) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def get_run_batch(self, run_id: str, batch_index: int) -> List[torch.Tensor] | Dict[
        str, torch.Tensor]:  # pragma: no cover - abstract
        raise NotImplementedError

    def list_run_batches(self, run_id: str) -> List[int]:  # pragma: no cover - abstract
        raise NotImplementedError

    def iter_run_batches(self, run_id: str) -> Iterator[List[torch.Tensor] | Dict[str, torch.Tensor]]:
        for idx in self.list_run_batches(run_id):
            yield self.get_run_batch(run_id, idx)

    def iter_run_batch_range(
            self,
            run_id: str,
            *,
            start: int = 0,
            stop: int | None = None,
            step: int = 1,
    ) -> Iterator[List[torch.Tensor] | Dict[str, torch.Tensor]]:
        """Iterate run batches for indices in range(start, stop, step).

        If stop is None, it will be set to max(list_run_batches(run_id)) + 1 (or 0 if none).
        Raises ValueError if step == 0 or start < 0.
        """
        if step == 0:
            raise ValueError("step must not be 0")
        if start < 0:
            raise ValueError("start must be >= 0")
        indices = self.list_run_batches(run_id)
        if not indices:
            return
            yield  # pragma: no cover - make this a generator even when empty
        max_idx = max(indices)
        if stop is None:
            stop = max_idx + 1
        # Iterate using numeric range; this will attempt to load each index even if some are missing.
        # Implementations may raise on missing indices (e.g., LocalStore) or return empty/dicts (FakeMemStore tests).
        for idx in range(start, stop, step):
            try:
                yield self.get_run_batch(run_id, idx)
            except FileNotFoundError:
                # Skip missing batch files (e.g., gaps when some batches weren't saved)
                continue

    def delete_run(self, run_id: str) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    # --- Run metadata (optional helpers) ---
    def put_run_meta(self, run_id: str, meta: Dict[str, Any]) -> str:  # pragma: no cover - abstract
        """Persist metadata for a run (e.g., dataset/model identifiers).

        Implementations should store JSON at a stable location, e.g., runs/{run_id}/meta.json.
        Returns the key/path used for storage.
        """
        raise NotImplementedError

    def get_run_meta(self, run_id: str) -> Dict[str, Any]:  # pragma: no cover - abstract
        """Load metadata for a run. Should return an empty dict if missing."""
        raise NotImplementedError


@dataclass
class LocalStore(Store):
    base_path: Path | str = ''

    def __init__(self, base_path: Path | str = '', runs_prefix: str = "activations"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.runs_prefix = runs_prefix

    def _full(self, key: str) -> Path:
        p = self.base_path / key
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # Single-tensor IO using safetensors files (mmap-friendly)
    def put_tensor(self, key: str, tensor: torch.Tensor) -> None:  # type: ignore[override]
        path = self._full(key)
        storch.save_file({"tensor": tensor}, str(path))

    def get_tensor(self, key: str) -> torch.Tensor:  # type: ignore[override]
        loaded = storch.load_file(str(self._full(key)))
        return loaded["tensor"]

    # Run-batch IO
    def put_run_batch(self, run_id: str, batch_index: int,
                      tensors: List[torch.Tensor] | Dict[str, torch.Tensor]) -> str:  # type: ignore[override]
        if isinstance(tensors, dict):
            to_save = tensors
        else:
            to_save = {f"item_{i}": t for i, t in enumerate(tensors)}
        key = self._run_batch_key(run_id, batch_index)
        storch.save_file(to_save, str(self._full(key)))
        return key

    def get_run_batch(self, run_id: str, batch_index: int) -> List[torch.Tensor] | Dict[
        str, torch.Tensor]:  # type: ignore[override]
        key = self._run_batch_key(run_id, batch_index)
        loaded = storch.load_file(str(self._full(key)))
        keys = list(loaded.keys())
        if keys and all(k.startswith("item_") for k in keys):
            try:
                items = sorted(((int(k.split("_", 1)[1]), v) for k, v in loaded.items()), key=lambda x: x[0])
                if [i for i, _ in items] == list(range(len(items))):
                    return [v for _, v in items]
            except Exception:
                pass
        return loaded

    def put_run_meta(self, run_id: str, meta: Dict[str, Any]) -> str:  # type: ignore[override]
        base = self.base_path / self.runs_prefix / run_id
        base.mkdir(parents=True, exist_ok=True)
        key = f"{self.runs_prefix}/{run_id}/meta.json"
        path = self._full(key)
        with path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return key

    def get_run_meta(self, run_id: str) -> Dict[str, Any]:  # type: ignore[override]
        path = self._full(f"{self.runs_prefix}/{run_id}/meta.json")
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def list_run_batches(self, run_id: str) -> List[int]:  # type: ignore[override]
        base = self.base_path / self.runs_prefix / run_id
        if not base.exists():
            return []
        out: List[int] = []
        for p in sorted(base.glob("batch_*.safetensors")):
            name = p.name
            try:
                idx = int(name[len("batch_"): len("batch_") + 6])
                out.append(idx)
            except Exception:
                continue
        return out

    def delete_run(self, run_id: str) -> None:  # type: ignore[override]
        base = self.base_path / self.runs_prefix / run_id
        if not base.exists():
            return
        for p in base.glob("batch_*.safetensors"):
            if p.is_file():
                p.unlink()
