from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache


def get_config_file_path() -> Path:
    """Get the path to the server config file."""
    return Path.home() / ".config" / "amber_server" / "config.json"


class ConfigManager:
    """Manages configuration file persistence."""

    def __init__(self, config_file: Path | None = None):
        self._config_file = config_file or get_config_file_path()

    def load_artifact_path(self) -> Path | None:
        """Load the saved artifact base path from config file, if it exists."""
        if not self._config_file.exists():
            return None
        try:
            with open(self._config_file, "r") as f:
                config = json.load(f)
                if "artifact_base_path" in config:
                    return Path(config["artifact_base_path"]).expanduser()
        except (json.JSONDecodeError, KeyError, OSError):
            return None
        return None

    def save_artifact_path(self, path: Path) -> None:
        """Save the artifact base path to config file."""
        self._config_file.parent.mkdir(parents=True, exist_ok=True)
        config = {"artifact_base_path": str(path)}
        with open(self._config_file, "w") as f:
            json.dump(config, f, indent=2)

    def get_default_artifact_path(self) -> Path:
        """Get the default artifact path, checking saved config first."""
        saved_path = self.load_artifact_path()
        if saved_path is not None:
            return saved_path
        return Path.home() / ".cache" / "amber_server"


@lru_cache
def get_config_manager() -> ConfigManager:
    """Get the global config manager instance."""
    return ConfigManager()

