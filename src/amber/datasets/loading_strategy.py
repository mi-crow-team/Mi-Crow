from __future__ import annotations

from enum import Enum
from typing import Union, Sequence, TypeAlias


class LoadingStrategy(Enum):
    """
    Strategy for loading dataset data.
    
    Choose the best strategy for your use case:
    
    - MEMORY: Load entire dataset into memory (fastest random access, highest memory usage)
      Best for: Small datasets that fit in memory, when you need fast random access
    
    - DISK: Save to disk, read dynamically via memory-mapped Arrow files
      (supports len/getitem, lower memory usage)
      Best for: Large datasets that don't fit in memory, when you need random access
    
    - STREAMING: True streaming mode using IterableDataset (lowest memory, no len/getitem support)
      Best for: Very large datasets, when you only need sequential iteration
    """
    MEMORY = "memory"  # Load all into memory (fastest random access, highest memory usage)
    DISK = "disk"  # Save to disk, read dynamically via memory-mapped Arrow files (supports len/getitem, lower memory usage)
    STREAMING = "streaming"  # True streaming mode using IterableDataset (lowest memory, no len/getitem support)


IndexLike: TypeAlias = Union[int, slice, Sequence[int]]

