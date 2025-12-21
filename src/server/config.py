from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_config_file_path() -> Path:
    """Get the path to the server config file."""
    return Path.home() / ".config" / "amber_server" / "config.json"


def load_saved_artifact_path() -> Path | None:
    """Load the saved artifact base path from config file, if it exists."""
    config_file = get_config_file_path()
    if not config_file.exists():
        return None
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
            if "artifact_base_path" in config:
                return Path(config["artifact_base_path"]).expanduser()
    except (json.JSONDecodeError, KeyError, OSError):
        return None
    return None


def save_artifact_path(path: Path) -> None:
    """Save the artifact base path to config file."""
    config_file = get_config_file_path()
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config = {"artifact_base_path": str(path)}
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


def get_default_artifact_path() -> Path:
    """Get the default artifact path, checking saved config first."""
    saved_path = load_saved_artifact_path()
    if saved_path is not None:
        return saved_path
    return Path.home() / ".cache" / "amber_server"


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

    model_config = SettingsConfigDict(env_prefix="SERVER_", case_sensitive=False)

    @field_validator("artifact_base_path", mode="before")
    @classmethod
    def load_artifact_path(cls, v: Path | str | None) -> Path:
        """Load artifact path from env var, saved config, or use default."""
        # If explicitly set (e.g., from env var), use it
        if v is not None:
            return Path(v).expanduser() if isinstance(v, str) else v

        # Otherwise, try to load from saved config file
        saved_path = load_saved_artifact_path()
        if saved_path is not None:
            return saved_path

        # Fall back to default
        return Path.home() / ".cache" / "amber_server"
