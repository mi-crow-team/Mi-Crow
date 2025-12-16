from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from server.dependencies import get_inference_service, get_model_manager
from server.inference_service import InferenceService
from server.model_manager import ModelManager
from server.schemas import InferenceRequest, InferenceResponse

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("", response_model=InferenceResponse)
def run_inference(
    payload: InferenceRequest,
    manager: ModelManager = Depends(get_model_manager),
    service: InferenceService = Depends(get_inference_service),
) -> InferenceResponse:
    try:
        lm = manager.get_model(payload.model_id)
        outputs = service.run(lm, payload.inputs)
        return InferenceResponse(outputs=outputs)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
