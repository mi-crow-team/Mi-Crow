"""JSON utility functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON payload to file."""
    path.write_text(json.dumps(payload, indent=2))

