from __future__ import annotations

from fastapi import APIRouter, Depends

from server.dependencies import get_model_manager
from server.middleware.error_handler import handle_errors
from server.model_manager import ModelManager
from server.schemas import ModelInfo, ModelLoadRequest

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
def list_models(manager: ModelManager = Depends(get_model_manager)) -> list[ModelInfo]:
    return manager.list_models()


@router.post("/load", response_model=ModelInfo)
@handle_errors
def load_model(payload: ModelLoadRequest, manager: ModelManager = Depends(get_model_manager)) -> ModelInfo:
    if payload.action == "unload":
        manager.unload_model(payload.model_id)
        return ModelInfo(id=payload.model_id, name=payload.model_id, status="unloaded")
    return manager.load_model(payload.model_id)
