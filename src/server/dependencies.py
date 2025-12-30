from __future__ import annotations

from functools import lru_cache
from fastapi import Depends, Header, HTTPException, status

from mi_crow.hooks.implementations.layer_activation_detector import LayerActivationDetector
from mi_crow.hooks.implementations.model_input_detector import ModelInputDetector
from mi_crow.hooks.implementations.model_output_detector import ModelOutputDetector
from server.hooks import NeuronMultiplierController

from server.config import Settings
from server.hook_factory import HookFactory
from server.inference_service import InferenceService
from server.job_manager import JobManager
from server.model_manager import ModelManager
from server.sae_service import SAEService
from server.services.activation_service import ActivationService
from server.services.concept_service import ConceptService
from server.services.sae_training_service import SAETrainingService


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_model_manager() -> ModelManager:
    return ModelManager(settings=get_settings())


@lru_cache
def get_hook_factory() -> HookFactory:
    hook_classes = [
        LayerActivationDetector,
        ModelInputDetector,
        ModelOutputDetector,
        NeuronMultiplierController,
    ]
    return HookFactory.from_modules(hook_classes)


@lru_cache
def get_inference_service() -> InferenceService:
    return InferenceService(hook_factory=get_hook_factory())


@lru_cache
def get_job_manager() -> JobManager:
    return JobManager()


@lru_cache
def get_activation_service() -> ActivationService:
    return ActivationService(settings=get_settings())


@lru_cache
def get_sae_training_service() -> SAETrainingService:
    return SAETrainingService(settings=get_settings(), job_manager=get_job_manager())


@lru_cache
def get_concept_service() -> ConceptService:
    return ConceptService(settings=get_settings())


@lru_cache
def get_sae_service() -> SAEService:
    return SAEService(
        settings=get_settings(),
        inference_service=get_inference_service(),
        job_manager=get_job_manager(),
        activation_service=get_activation_service(),
        training_service=get_sae_training_service(),
        concept_service=get_concept_service(),
    )


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """Verify API key for protected endpoints."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")
