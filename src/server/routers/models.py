from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from server.dependencies import get_model_manager
from server.model_manager import ModelManager
from server.schemas import ModelInfo, ModelLoadRequest

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
def list_models(manager: ModelManager = Depends(get_model_manager)) -> list[ModelInfo]:
    return manager.list_models()


@router.post("/load", response_model=ModelInfo)
def load_model(payload: ModelLoadRequest, manager: ModelManager = Depends(get_model_manager)) -> ModelInfo:
    try:
        if payload.action == "unload":
            manager.unload_model(payload.model_id)
            return ModelInfo(id=payload.model_id, name=payload.model_id, status="unloaded")
        return manager.load_model(payload.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
