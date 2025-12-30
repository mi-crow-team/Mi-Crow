from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ModelInfo(BaseModel):
    id: str
    name: str
    status: str


class ModelLoadRequest(BaseModel):
    model_id: str
    action: str = Field("load", description="load or unload")

    @field_validator("action")
    @classmethod
    def validate_action(cls, value: str) -> str:
        allowed = {"load", "unload"}
        if value not in allowed:
            raise ValueError(f"action must be one of {allowed}")
        return value


class LayerInfo(BaseModel):
    layer_id: str
    name: str
    type: str
    path: List[str]


class HookPayload(BaseModel):
    hook_name: str
    layer_id: str
    config: Dict[str, Any] = Field(default_factory=dict)


class ReturnOptions(BaseModel):
    logits: bool = False
    tokens: bool = True
    activations: bool = False
    probabilities: bool = False


class GenerationConfig(BaseModel):
    max_new_tokens: int = 16
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool = True
    repetition_penalty: float | None = None
    stop: List[str] | None = None
    seed: int | None = None


class InferenceInput(BaseModel):
    prompt: str
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    return_options: ReturnOptions = Field(default_factory=ReturnOptions)
    hooks: List[HookPayload] = Field(default_factory=list)


class InferenceRequest(BaseModel):
    model_id: str
    inputs: List[InferenceInput]

    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, value: List[InferenceInput]) -> List[InferenceInput]:
        if not value:
            raise ValueError("inputs must contain at least one item")
        if len(value) > 2:
            raise ValueError("compare mode supports at most two inputs")
        return value


class HookResult(BaseModel):
    hook_name: str
    layer_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tensors: Dict[str, Any] = Field(default_factory=dict)


class InferenceOutput(BaseModel):
    text: str
    tokens: List[str] = Field(default_factory=list)
    logits: Optional[List[float]] = None
    probabilities: Optional[List[float]] = None
    hooks: List[HookResult] = Field(default_factory=list)
    timing_ms: float | None = None


class InferenceResponse(BaseModel):
    outputs: List[InferenceOutput]


class SaveActivationsRequest(BaseModel):
    model_id: str
    layers: List[str]
    dataset: Dict[str, Any]
    sample_limit: int | None = None
    batch_size: int = 4
    shard_size: int = 64
    run_id: str | None = None


class SaveActivationsResponse(BaseModel):
    path: str
    run_id: str
    samples: int
    tokens: int
    layers: List[str]
    batches: List[Dict[str, Any]] = Field(default_factory=list)
    dataset: Dict[str, Any] = Field(default_factory=dict)
    manifest_path: str | None = None
    status: str | None = None
    created_at: str | None = None


class ActivationRunInfo(BaseModel):
    model_id: str
    run_id: str
    manifest_path: str | None = None
    samples: int | None = None
    tokens: int | None = None
    layers: List[str] = Field(default_factory=list)
    dataset: Dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    status: str | None = None


class ActivationRunListResponse(BaseModel):
    model_id: str
    runs: List[ActivationRunInfo] = Field(default_factory=list)


class TrainSAERequest(BaseModel):
    model_id: str
    activations_path: str
    layer: str | None = None
    sae_class: str = "TopKSae"
    sae_kwargs: Dict[str, Any] = Field(default_factory=dict)
    hyperparams: Dict[str, Any] = Field(default_factory=dict)
    training_config: Dict[str, Any] = Field(default_factory=dict)
    n_latents: int | None = None
    run_id: str | None = None


class TrainSAEResponse(BaseModel):
    job_id: str
    status: str


class LayerSizeInfo(BaseModel):
    layer: str
    hidden_dim: int


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str
    sae_id: str | None = None
    sae_path: str | None = None
    metadata_path: str | None = None
    progress: float | None = None
    logs: List[str] = Field(default_factory=list)
    error: str | None = None


class LoadSAERequest(BaseModel):
    model_id: str
    sae_path: str


class LoadSAEResponse(BaseModel):
    sae_id: str


class SaeRunInfo(BaseModel):
    model_id: str
    sae_id: str
    sae_path: str | None = None
    metadata_path: str | None = None
    sae_class: str | None = None
    layer: str | None = None
    created_at: str | None = None


class SaeRunListResponse(BaseModel):
    model_id: str
    saes: List[SaeRunInfo] = Field(default_factory=list)


class SAEInferenceRequest(BaseModel):
    model_id: str
    sae_id: str | None = None
    sae_path: str | None = None
    layer: str | None = None
    inputs: List[InferenceInput]
    save_top_texts: bool = False
    concept_config_path: str | None = None
    top_k_neurons: int = 5
    track_texts: bool = False
    return_token_latents: bool = False

    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, value: List[InferenceInput]) -> List[InferenceInput]:
        if not value:
            raise ValueError("inputs must contain at least one item")
        return value

    @field_validator("sae_path", mode="after")
    @classmethod
    def ensure_sae_source(cls, value, info):
        sae_id = info.data.get("sae_id")
        if not value and not sae_id:
            raise ValueError("either sae_id or sae_path must be provided")
        return value


class SAEInferenceResponse(BaseModel):
    outputs: List[InferenceOutput]
    sae_id: str | None = None
    sae_path: str | None = None
    metadata_path: str | None = None
    top_neurons: List[Dict[str, Any]] = Field(default_factory=list)
    token_latents: List[Any] = Field(default_factory=list)
    top_texts_path: str | None = None


class ConceptListResponse(BaseModel):
    base_path: str
    concepts: List[str] = Field(default_factory=list)


class ConceptDictionaryResponse(BaseModel):
    n_size: int
    concepts: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class ConceptLoadRequest(BaseModel):
    model_id: str
    sae_id: str
    source_path: str


class ConceptLoadResponse(BaseModel):
    concept_id: str
    stored_path: str


class ConceptManipulationRequest(BaseModel):
    model_id: str
    sae_id: str | None = None
    sae_path: str | None = None
    edits: Dict[str, float] = Field(default_factory=dict)
    bias: Dict[str, float] = Field(default_factory=dict)

    @field_validator("edits")
    @classmethod
    def validate_edits(cls, value: Dict[str, float]) -> Dict[str, float]:
        if not value:
            raise ValueError("edits cannot be empty")
        return {k: float(v) for k, v in value.items()}

    @field_validator("sae_path", mode="after")
    @classmethod
    def ensure_sae_source(cls, value, info):
        sae_id = info.data.get("sae_id")
        if not value and not sae_id:
            raise ValueError("either sae_id or sae_path must be provided")
        return value


class ConceptManipulationResponse(BaseModel):
    concept_config_path: str
    sae_id: str | None = None


class ConceptConfigInfo(BaseModel):
    name: str
    path: str
    created_at: str | None = None
    sae_id: str | None = None
    layer: str | None = None


class ConceptConfigListResponse(BaseModel):
    model_id: str
    sae_id: str | None = None
    configs: List[ConceptConfigInfo] = Field(default_factory=list)


class ConceptPreviewRequest(ConceptManipulationRequest):
    pass


class ConceptPreviewResponse(BaseModel):
    sae_id: str | None = None
    layer: str | None = None
    n_latents: int | None = None
    weights: Dict[str, float] = Field(default_factory=dict)
    bias: Dict[str, float] = Field(default_factory=dict)
