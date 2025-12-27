"""Path resolution utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from server.config import Settings


def resolve_sae_path(
    settings: Settings,
    sae_id: Optional[str] = None,
    sae_path: Optional[str] = None,
    sae_registry: Optional[dict[str, Path]] = None,
) -> Path:
    """
    Resolve SAE path from sae_id, sae_path, or registry.
    
    Args:
        settings: Application settings
        sae_id: Optional SAE ID to look up in registry
        sae_path: Optional explicit SAE path
        sae_registry: Optional registry mapping sae_id to Path
        
    Returns:
        Resolved Path to SAE file
        
    Raises:
        ValueError: If path cannot be resolved
    """
    if sae_path:
        path = Path(sae_path)
        if path.exists():
            return path
        raise ValueError(f"sae_path '{sae_path}' does not exist")
    
    if sae_id:
        if sae_registry and sae_id in sae_registry:
            return sae_registry[sae_id]
        
        # Try to find in artifact directory
        base = settings.artifact_base_path / "sae"
        for model_dir in base.iterdir():
            if not model_dir.is_dir():
                continue
            for run_dir in model_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                if run_dir.name == sae_id:
                    sae_file = next(run_dir.glob("*.pt"), None)
                    if sae_file and sae_file.exists():
                        return sae_file
                    metadata_path = run_dir / "metadata.json"
                    if metadata_path.exists():
                        import json
                        meta = json.loads(metadata_path.read_text())
                        if "sae_path" in meta:
                            path = Path(meta["sae_path"])
                            if path.exists():
                                return path
    
    raise ValueError("sae_id or sae_path must be provided")

