from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from server.dependencies import get_model_manager, get_sae_service, verify_api_key
from server.model_manager import ModelManager
from server.sae_service import SAEService
from server.schemas import (
    ConceptListResponse,
    ConceptLoadRequest,
    ConceptLoadResponse,
    ConceptManipulationRequest,
    ConceptManipulationResponse,
    ConceptConfigListResponse,
    ConceptPreviewRequest,
    ConceptPreviewResponse,
    ActivationRunListResponse,
    SaeRunListResponse,
    LoadSAERequest,
    LoadSAEResponse,
    SAEInferenceRequest,
    SAEInferenceResponse,
    SaveActivationsRequest,
    SaveActivationsResponse,
    TrainSAERequest,
    TrainSAEResponse,
    TrainStatusResponse,
)

router = APIRouter(prefix="/sae", tags=["sae"])


@router.post("/activations/save", response_model=SaveActivationsResponse)
def save_activations(
    payload: SaveActivationsRequest,
    manager: ModelManager = Depends(get_model_manager),
    service: SAEService = Depends(get_sae_service),
) -> SaveActivationsResponse:
    try:
        return service.save_activations(manager, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/activations", response_model=ActivationRunListResponse)
def list_activations(
    model_id: str = Query(...),
    service: SAEService = Depends(get_sae_service),
) -> ActivationRunListResponse:
    return service.list_activation_runs(model_id)


@router.delete("/activations/{run_id}")
def delete_activation_run(
    run_id: str,
    model_id: str = Query(...),
    service: SAEService = Depends(get_sae_service),
    _: None = Depends(verify_api_key),
) -> dict:
    deleted = service.delete_activation_run(model_id, run_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="activation run not found")
    return {"deleted": True, "run_id": run_id}


@router.post("/train", response_model=TrainSAEResponse)
def train_sae(
    payload: TrainSAERequest,
    manager: ModelManager = Depends(get_model_manager),
    service: SAEService = Depends(get_sae_service),
) -> TrainSAEResponse:
    try:
        return service.train_sae(manager, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/train/status/{job_id}", response_model=TrainStatusResponse)
def train_status(
    job_id: str,
    service: SAEService = Depends(get_sae_service),
) -> TrainStatusResponse:
    try:
        return service.train_status(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/train/cancel/{job_id}", response_model=TrainStatusResponse)
def cancel_train(
    job_id: str,
    service: SAEService = Depends(get_sae_service),
) -> TrainStatusResponse:
    try:
        return service.cancel_train(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/load", response_model=LoadSAEResponse)
def load_sae(
    payload: LoadSAERequest,
    service: SAEService = Depends(get_sae_service),
) -> LoadSAEResponse:
    try:
        return service.load_sae(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/saes", response_model=SaeRunListResponse)
def list_saes(
    model_id: str = Query(...),
    service: SAEService = Depends(get_sae_service),
) -> SaeRunListResponse:
    return service.list_sae_runs(model_id)


@router.delete("/saes/{sae_id}")
def delete_sae(
    sae_id: str,
    model_id: str = Query(...),
    service: SAEService = Depends(get_sae_service),
    _: None = Depends(verify_api_key),
) -> dict:
    deleted = service.delete_sae_run(model_id, sae_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="sae run not found")
    return {"deleted": True, "sae_id": sae_id}


@router.post("/infer", response_model=SAEInferenceResponse)
def sae_infer(
    payload: SAEInferenceRequest,
    manager: ModelManager = Depends(get_model_manager),
    service: SAEService = Depends(get_sae_service),
) -> SAEInferenceResponse:
    try:
        return service.infer(manager, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.get("/concepts", response_model=ConceptListResponse)
def list_concepts(
    model_id: str = Query(...),
    sae_id: str | None = Query(None),
    service: SAEService = Depends(get_sae_service),
) -> ConceptListResponse:
    return service.list_concepts(model_id, sae_id)


@router.get("/concepts/configs", response_model=ConceptConfigListResponse)
def list_concept_configs(
    model_id: str = Query(...),
    sae_id: str | None = Query(None),
    service: SAEService = Depends(get_sae_service),
) -> ConceptConfigListResponse:
    return service.list_concept_configs(model_id, sae_id)


@router.post("/concepts/load", response_model=ConceptLoadResponse)
def load_concepts(
    payload: ConceptLoadRequest,
    service: SAEService = Depends(get_sae_service),
) -> ConceptLoadResponse:
    try:
        return service.load_concepts(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/concepts/manipulate", response_model=ConceptManipulationResponse)
def manipulate_concepts(
    payload: ConceptManipulationRequest,
    service: SAEService = Depends(get_sae_service),
) -> ConceptManipulationResponse:
    try:
        return service.manipulate_concepts(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/concepts/preview", response_model=ConceptPreviewResponse)
def preview_concepts(
    payload: ConceptPreviewRequest,
    service: SAEService = Depends(get_sae_service),
) -> ConceptPreviewResponse:
    try:
        return service.preview_concepts(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/classes", tags=["sae"])
def list_sae_classes(
    service: SAEService = Depends(get_sae_service),
) -> dict[str, str]:
    """Return the available SAE classes as a simple mapping."""
    return service.list_sae_classes()
