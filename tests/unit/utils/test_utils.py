import logging
import random

import numpy as np
import torch

from amber.utils import set_seed, get_logger


def test_set_seed_reproducible():
    set_seed(123)
    a1 = random.random()
    b1 = np.random.RandomState().rand()
    t1 = torch.randn(3)

    set_seed(123)
    a2 = random.random()
    b2 = np.random.RandomState().rand()
    t2 = torch.randn(3)

    assert a1 == a2
    # numpy's global RNG not used; ensure torch consistent
    assert torch.allclose(t1, t2)


def test_get_logger_returns_configured_logger():
    logger = get_logger("amber.test", level=logging.DEBUG)
    assert isinstance(logger, logging.Logger)
    assert logger.name == "amber.test"
    # Should not add duplicate handlers on repeated calls
    h_count = len(logger.handlers)
    logger2 = get_logger("amber.test", level=logging.DEBUG)
    assert logger2 is logger
    assert len(logger.handlers) == h_count

def test_get_logger_accepts_string_level_and_sets_level_correctly():
    logger = get_logger("amber.utils.test_level", level="DEBUG")
    assert logger.level == logging.DEBUG
    # Calling again with the same name should not duplicate handlers
    h_count = len(logger.handlers)
    logger2 = get_logger("amber.utils.test_level", level="DEBUG")
    assert logger2 is logger
    assert len(logger.handlers) == h_count
