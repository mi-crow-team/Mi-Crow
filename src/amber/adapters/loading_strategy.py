from __future__ import annotations

from enum import Enum
from typing import Union, Sequence, TypeAlias


class LoadingStrategy(Enum):
    """Strategy for loading dataset data."""
    STREAM = "stream"  # Stream batches from file (lazy loading)
    MEMORY = "memory"  # Load all into memory (eager loading)


IndexLike: TypeAlias = Union[int, slice, Sequence[int]]

