from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set seeds for python, numpy, and torch.

    Args:
        seed: Seed value.
        deterministic: If True, tries to make torch deterministic where possible.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def get_logger(name: str = "mi_crow", level: int | str = logging.INFO) -> logging.Logger:
    """Get a configured logger with a simple format. Idempotent."""
    logger = logging.getLogger(name)
    if isinstance(level, str):
        level = logging.getLevelName(level)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        # Allow propagation so pytest's caplog can capture logs
        logger.propagate = True
    else:
        # Even if a handler exists (e.g., configured by the app), ensure propagation is enabled
        logger.propagate = True
    return logger
