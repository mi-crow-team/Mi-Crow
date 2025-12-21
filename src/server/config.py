"""Legacy config module - imports from new config package for backward compatibility."""

from server.config import Settings
from server.config.storage import ConfigManager, get_config_manager, save_artifact_path

__all__ = ["Settings", "ConfigManager", "get_config_manager", "save_artifact_path"]
