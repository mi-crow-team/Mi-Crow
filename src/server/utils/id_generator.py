"""ID generation utilities."""

from __future__ import annotations

import uuid


def generate_id() -> str:
    """Generate a unique ID (8 hex characters)."""
    return uuid.uuid4().hex[:8]

