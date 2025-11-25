import logging
import os
from unittest.mock import Mock, patch

import pytest

from amber import utils


def test_set_seed_calls_all_rngs(monkeypatch):
    random_seed = Mock()
    np_seed = Mock()
    manual_seed = Mock()
    manual_seed_all = Mock()
    use_det = Mock()
    cudnn = Mock()
    cudnn.deterministic = False
    cudnn.benchmark = True

    monkeypatch.setattr(utils.random, "seed", random_seed)
    monkeypatch.setattr(utils.np.random, "seed", np_seed)
    monkeypatch.setattr(utils.torch, "manual_seed", manual_seed)
    monkeypatch.setattr(utils.torch.cuda, "manual_seed_all", manual_seed_all, raising=False)
    monkeypatch.setattr(utils.torch, "use_deterministic_algorithms", use_det)
    monkeypatch.setattr(utils.torch.backends, "cudnn", cudnn, raising=False)

    utils.set_seed(123)

    random_seed.assert_called_once_with(123)
    np_seed.assert_called_once_with(123)
    manual_seed.assert_called_once_with(123)
    manual_seed_all.assert_called_once_with(123)
    use_det.assert_called_once()
    assert os.environ["PYTHONHASHSEED"] == "123"
    assert cudnn.deterministic is True
    assert cudnn.benchmark is False


def test_set_seed_non_deterministic(monkeypatch):
    use_det = Mock()
    monkeypatch.setattr(utils.torch, "use_deterministic_algorithms", use_det)

    utils.set_seed(1, deterministic=False)

    use_det.assert_not_called()


def test_get_logger_idempotent_handlers():
    name = "amber-test-logger"
    logger = utils.get_logger(name, level=logging.DEBUG)
    handler_count = len(logger.handlers)

    logger2 = utils.get_logger(name)

    assert logger is logger2
    assert len(logger.handlers) == handler_count
    assert logger.propagate is True

