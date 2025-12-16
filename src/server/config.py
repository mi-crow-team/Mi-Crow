from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    artifact_base_path: Path = Field(default_factory=lambda: Path.home() / ".cache" / "amber_server")
    api_key: str | None = Field(default=None, description="Optional API key for protected endpoints")

    model_config = SettingsConfigDict(env_prefix="SERVER_", case_sensitive=False)
