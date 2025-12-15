from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def activation_dir(base: Path, model_id: str, run_id: str | None = None) -> Path:
    folder = base / "activations" / model_id
    if run_id:
        folder = folder / run_id
    return ensure_dir(folder)


def sae_dir(base: Path, model_id: str, run_id: str | None = None) -> Path:
    folder = base / "sae" / model_id
    if run_id:
        folder = folder / run_id
    return ensure_dir(folder)


def concepts_dir(base: Path, model_id: str, sae_id: str | None = None) -> Path:
    folder = base / "concepts" / model_id
    if sae_id:
        folder = folder / sae_id
    return ensure_dir(folder)


def top_texts_dir(base: Path, model_id: str, sae_id: str | None = None) -> Path:
    folder = base / "top_texts" / model_id
    if sae_id:
        folder = folder / sae_id
    return ensure_dir(folder)
