from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from server.dependencies import get_model_manager
from server.model_manager import ModelManager
from server.schemas import LayerInfo

router = APIRouter(prefix="/models", tags=["layers"])


@router.get("/{model_id}/layers", response_model=list[LayerInfo])
def get_layers(model_id: str, manager: ModelManager = Depends(get_model_manager)) -> list[LayerInfo]:
    try:
        return manager.get_layer_tree(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
