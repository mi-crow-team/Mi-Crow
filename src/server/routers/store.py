from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from server.config import Settings
from server.config.storage import get_config_manager
from server.dependencies import get_sae_service, get_settings
from server.sae_service import SAEService
from server.schemas import ActivationRunInfo


class StoreInfo(BaseModel):
    """Information about the current store/artifact configuration."""

    artifact_base_path: str
    activation_datasets: Dict[str, List[ActivationRunInfo]] = Field(default_factory=dict)


class StorePathUpdate(BaseModel):
    """Request payload for updating the store/artifact base path."""

    artifact_base_path: str


router = APIRouter(prefix="/store", tags=["store"])


def _gather_activation_datasets(
    settings: Settings,
    service: SAEService,
) -> Dict[str, List[ActivationRunInfo]]:
    """Scan the artifact tree and collect activation runs per model."""
    base = settings.artifact_base_path
    activation_root = base / "activations"
    by_model: Dict[str, List[ActivationRunInfo]] = {}

    if not activation_root.exists():
        return by_model

    for model_dir in sorted(p for p in activation_root.iterdir() if p.is_dir()):
        model_id = model_dir.name
        runs = service.list_activation_runs(model_id)
        if runs.runs:
            by_model[model_id] = runs.runs

    return by_model


@router.get("/info", response_model=StoreInfo)
def get_store_info(
    settings: Settings = Depends(get_settings),
    service: SAEService = Depends(get_sae_service),
) -> StoreInfo:
    """Return the current artifact base path and discovered local datasets."""
    activation_datasets = _gather_activation_datasets(settings, service)
    return StoreInfo(
        artifact_base_path=str(settings.artifact_base_path),
        activation_datasets=activation_datasets,
    )


@router.post("/path", response_model=StoreInfo)
def set_store_path(
    payload: StorePathUpdate,
    settings: Settings = Depends(get_settings),
    service: SAEService = Depends(get_sae_service),
) -> StoreInfo:
    """
    Update the artifact base path used for activations, SAEs and concepts.

    This only affects new runs; existing data will continue to live at the old path.
    The path is persisted to a config file and will be loaded on server restart.
    """
    new_path = Path(payload.artifact_base_path).expanduser()
    new_path.mkdir(parents=True, exist_ok=True)
    settings.artifact_base_path = new_path

    # Save the path to config file for persistence
    config_manager = get_config_manager()
    config_manager.save_artifact_path(new_path)

    activation_datasets = _gather_activation_datasets(settings, service)
    return StoreInfo(
        artifact_base_path=str(settings.artifact_base_path),
        activation_datasets=activation_datasets,
    )












