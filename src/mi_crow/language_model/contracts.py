from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class ModelMetadata:
    """Metadata for a saved language model."""
    model_id: str
    hooks: Dict[str, List[Dict[str, Any]]]
    model_path: str

