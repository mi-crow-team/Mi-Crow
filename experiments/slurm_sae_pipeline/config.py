"""Pydantic models for pipeline configuration."""
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration."""
    model_id: str = Field(..., description="HuggingFace model ID")


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    hf_dataset: str = Field(..., description="HuggingFace dataset name")
    data_split: str = Field(default="train", description="Dataset split to use")
    text_field: str = Field(default="text", description="Field name containing text")
    data_limit: int = Field(..., description="Maximum number of samples to use")
    max_length: int = Field(..., description="Maximum sequence length")
    batch_size_save: int = Field(..., description="Batch size for saving activations")


class LayerConfig(BaseModel):
    """Layer configuration."""
    layer_num: int = Field(..., description="Layer number")
    layer_signature: Optional[str] = Field(default=None, description="Layer signature (auto-generated if None)")


class StorageConfig(BaseModel):
    """Storage configuration."""
    store_dir: Optional[str] = Field(default=None, description="Store directory path")
    device: Optional[str] = Field(default=None, description="Device to use (cuda/cpu)")


class SaeConfig(BaseModel):
    """SAE configuration."""
    n_latents_multiplier: int = Field(..., description="Overcompleteness factor")
    top_k: int = Field(..., description="Top-K sparsity parameter")


class TrainingConfig(BaseModel):
    """Training configuration."""
    epochs: int = Field(..., description="Number of training epochs")
    batch_size_train: int = Field(..., description="Training batch size")
    lr: float = Field(..., description="Learning rate")
    l1_lambda: float = Field(..., description="L1 regularization weight")
    snapshot_every_n_epochs: Optional[int] = Field(default=None, description="Save model snapshot every N epochs (None = disabled)")
    snapshot_base_path: Optional[str] = Field(default=None, description="Base path for snapshots (defaults to training_run_id/snapshots)")


class PipelineConfig(BaseModel):
    """Root pipeline configuration."""
    model: ModelConfig
    dataset: DatasetConfig
    layer: LayerConfig
    storage: StorageConfig
    sae: SaeConfig
    training: TrainingConfig

    @classmethod
    def from_json_file(cls, config_path: Path) -> "PipelineConfig":
        """Load configuration from JSON file."""
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found at {config_path}")
        
        with open(config_path, "r") as f:
            data = json.load(f)
        
        return cls(**data)
