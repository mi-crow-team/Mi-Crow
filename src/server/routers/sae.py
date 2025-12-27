from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from server.dependencies import get_model_manager, get_sae_service, verify_api_key
from server.middleware.error_handler import handle_errors
from server.model_manager import ModelManager
from server.sae_service import SAEService
from server.schemas import (
    ConceptListResponse,
    ConceptDictionaryResponse,
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
    LayerSizeInfo,
)

router = APIRouter(prefix="/sae", tags=["sae"])


@router.post("/activations/save", response_model=SaveActivationsResponse)
@handle_errors
def save_activations(
    payload: SaveActivationsRequest,
    manager: ModelManager = Depends(get_model_manager),
    service: SAEService = Depends(get_sae_service),
) -> SaveActivationsResponse:
    return service.save_activations(manager, payload)


@router.get("/activations", response_model=ActivationRunListResponse)
def list_activations(
    model_id: str = Query(...),
    service: SAEService = Depends(get_sae_service),
) -> ActivationRunListResponse:
    return service.list_activation_runs(model_id)


@router.delete("/activations/{run_id}")
@handle_errors
def delete_activation_run(
    run_id: str,
    model_id: str = Query(...),
    service: SAEService = Depends(get_sae_service),
    _: None = Depends(verify_api_key),
) -> dict:
    from server.exceptions import NotFoundError

    deleted = service.delete_activation_run(model_id, run_id)
    if not deleted:
        raise NotFoundError("activation run not found")
    return {"deleted": True, "run_id": run_id}


@router.post("/train", response_model=TrainSAEResponse)
@handle_errors
def train_sae(
    payload: TrainSAERequest,
    manager: ModelManager = Depends(get_model_manager),
    service: SAEService = Depends(get_sae_service),
) -> TrainSAEResponse:
    return service.train_sae(manager, payload)


@router.get("/train/status/{job_id}", response_model=TrainStatusResponse)
@handle_errors
def train_status(
    job_id: str,
    service: SAEService = Depends(get_sae_service),
) -> TrainStatusResponse:
    return service.train_status(job_id)


@router.post("/train/cancel/{job_id}", response_model=TrainStatusResponse)
@handle_errors
def cancel_train(
    job_id: str,
    service: SAEService = Depends(get_sae_service),
) -> TrainStatusResponse:
    return service.cancel_train(job_id)


@router.get("/train/layer-size", response_model=LayerSizeInfo)
@handle_errors
def get_layer_size(
    activations_path: str = Query(...),
    layer: str = Query(...),
    service: SAEService = Depends(get_sae_service),
) -> LayerSizeInfo:
    result = service.get_layer_size(activations_path, layer)
    return LayerSizeInfo(**result)


@router.post("/load", response_model=LoadSAEResponse)
@handle_errors
def load_sae(
    payload: LoadSAERequest,
    service: SAEService = Depends(get_sae_service),
) -> LoadSAEResponse:
    return service.load_sae(payload)


@router.get("/saes", response_model=SaeRunListResponse)
def list_saes(
    model_id: str = Query(...),
    service: SAEService = Depends(get_sae_service),
) -> SaeRunListResponse:
    return service.list_sae_runs(model_id)


@router.get("/saes/{sae_id}/metadata")
@handle_errors
def get_sae_metadata(
    model_id: str = Query(...),
    sae_id: str = ...,
    service: SAEService = Depends(get_sae_service),
) -> dict:
    return service.get_sae_metadata(model_id, sae_id)


@router.delete("/saes/{sae_id}")
@handle_errors
def delete_sae(
    sae_id: str,
    model_id: str = Query(...),
    service: SAEService = Depends(get_sae_service),
    _: None = Depends(verify_api_key),
) -> dict:
    from server.exceptions import NotFoundError

    deleted = service.delete_sae_run(model_id, sae_id)
    if not deleted:
        raise NotFoundError("sae run not found")
    return {"deleted": True, "sae_id": sae_id}


@router.post("/infer", response_model=SAEInferenceResponse)
@handle_errors
def sae_infer(
    payload: SAEInferenceRequest,
    manager: ModelManager = Depends(get_model_manager),
    service: SAEService = Depends(get_sae_service),
) -> SAEInferenceResponse:
    return service.infer(manager, payload)


@router.get("/concepts", response_model=ConceptListResponse)
def list_concepts(
    model_id: str = Query(...),
    sae_id: str | None = Query(None),
    service: SAEService = Depends(get_sae_service),
) -> ConceptListResponse:
    return service.list_concepts(model_id, sae_id)


@router.get("/concepts/dictionary", response_model=ConceptDictionaryResponse)
@handle_errors
def get_concept_dictionary(
    model_id: str = Query(...),
    sae_id: str | None = Query(None),
    service: SAEService = Depends(get_sae_service),
) -> ConceptDictionaryResponse:
    return service.get_concept_dictionary(model_id, sae_id)


@router.get("/concepts/configs", response_model=ConceptConfigListResponse)
def list_concept_configs(
    model_id: str = Query(...),
    sae_id: str | None = Query(None),
    service: SAEService = Depends(get_sae_service),
) -> ConceptConfigListResponse:
    return service.list_concept_configs(model_id, sae_id)


@router.post("/concepts/load", response_model=ConceptLoadResponse)
@handle_errors
def load_concepts(
    payload: ConceptLoadRequest,
    service: SAEService = Depends(get_sae_service),
) -> ConceptLoadResponse:
    return service.load_concepts(payload)


@router.post("/concepts/manipulate", response_model=ConceptManipulationResponse)
@handle_errors
def manipulate_concepts(
    payload: ConceptManipulationRequest,
    service: SAEService = Depends(get_sae_service),
) -> ConceptManipulationResponse:
    return service.manipulate_concepts(payload)


@router.post("/concepts/preview", response_model=ConceptPreviewResponse)
@handle_errors
def preview_concepts(
    payload: ConceptPreviewRequest,
    service: SAEService = Depends(get_sae_service),
) -> ConceptPreviewResponse:
    return service.preview_concepts(payload)


@router.get("/classes", tags=["sae"])
def list_sae_classes(
    service: SAEService = Depends(get_sae_service),
) -> dict[str, str]:
    """Return the available SAE classes as a simple mapping."""
    return service.list_sae_classes()
