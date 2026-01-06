from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from server.config.storage import get_config_manager


def get_default_artifact_path() -> Path:
    """Get the default artifact path from config manager."""
    return get_config_manager().get_default_artifact_path()


class Settings(BaseSettings):
    """
    Application settings.

    Only Hugging Face hosted models are supported for now.
    """

    allowed_models: dict[str, str] = Field(
        default_factory=lambda: {
            "bielik": "speakleash/Bielik-1.5B-v3.0-Instruct",
            "plum": "plum-2b",
            "tinylm": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        }
    )
    hf_token: str | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    artifact_base_path: Path = Field(default_factory=get_default_artifact_path)
    api_key: str | None = Field(default=None, description="Optional API key for protected endpoints")
    wandb_api_key: str | None = Field(default=None, description="Wandb API key for experiment tracking")
    wandb_project: str | None = Field(default=None, description="Default wandb project name for training")

    model_config = SettingsConfigDict(env_prefix="SERVER_", case_sensitive=False)

    @field_validator("artifact_base_path", mode="before")
    @classmethod
    def load_artifact_path(cls, v: Path | str | None) -> Path:
        """Load artifact path from env var, saved config, or use default."""
        # If explicitly set (e.g., from env var), use it
        if v is not None:
            return Path(v).expanduser() if isinstance(v, str) else v

        # Otherwise, try to load from saved config file
        config_manager = get_config_manager()
        saved_path = config_manager.load_artifact_path()
        if saved_path is not None:
            return saved_path

        # Fall back to default
        return Path.home() / ".cache" / "mi_crow_server"

