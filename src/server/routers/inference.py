from __future__ import annotations

from fastapi import APIRouter, Depends

from server.dependencies import get_inference_service, get_model_manager
from server.inference_service import InferenceService
from server.middleware.error_handler import handle_errors
from server.model_manager import ModelManager
from server.schemas import InferenceRequest, InferenceResponse

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("", response_model=InferenceResponse)
@handle_errors
def run_inference(
    payload: InferenceRequest,
    manager: ModelManager = Depends(get_model_manager),
    service: InferenceService = Depends(get_inference_service),
) -> InferenceResponse:
    lm = manager.get_model(payload.model_id)
    outputs = service.run(lm, payload.inputs)
    return InferenceResponse(outputs=outputs)
