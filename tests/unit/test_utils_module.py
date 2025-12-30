"""Comprehensive tests for mi_crow.utils module."""

import logging
import os
from unittest.mock import Mock

import pytest

from mi_crow import utils


class TestSetSeed:
    """Comprehensive tests for set_seed function."""

    def test_set_seed_calls_all_rngs(self, monkeypatch):
        """Test that set_seed calls all RNG seed functions."""
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
        use_det.assert_called_once_with(True, warn_only=True)
        assert os.environ["PYTHONHASHSEED"] == "123"
        assert cudnn.deterministic is True
        assert cudnn.benchmark is False

    def test_set_seed_non_deterministic(self, monkeypatch):
        """Test set_seed with deterministic=False."""
        use_det = Mock()
        monkeypatch.setattr(utils.torch, "use_deterministic_algorithms", use_det)

        utils.set_seed(1, deterministic=False)

        use_det.assert_not_called()

    @pytest.mark.parametrize("seed", [0, 1, 42, 12345, -1, 2**31 - 1])
    def test_set_seed_various_seed_values(self, monkeypatch, seed):
        """Test set_seed with various seed values."""
        random_seed = Mock()
        np_seed = Mock()
        manual_seed = Mock()
        
        monkeypatch.setattr(utils.random, "seed", random_seed)
        monkeypatch.setattr(utils.np.random, "seed", np_seed)
        monkeypatch.setattr(utils.torch, "manual_seed", manual_seed)
        monkeypatch.setattr(utils.torch.cuda, "manual_seed_all", Mock(), raising=False)
        monkeypatch.setattr(utils.torch, "use_deterministic_algorithms", Mock())
        monkeypatch.setattr(utils.torch.backends, "cudnn", Mock(), raising=False)

        utils.set_seed(seed)

        random_seed.assert_called_once_with(seed)
        np_seed.assert_called_once_with(seed)
        manual_seed.assert_called_once_with(seed)
        assert os.environ["PYTHONHASHSEED"] == str(seed)

    def test_set_seed_overwrites_pythonhashseed(self, monkeypatch):
        """Test that set_seed overwrites existing PYTHONHASHSEED."""
        original_value = os.environ.get("PYTHONHASHSEED", "original")
        monkeypatch.setattr(utils.random, "seed", Mock())
        monkeypatch.setattr(utils.np.random, "seed", Mock())
        monkeypatch.setattr(utils.torch, "manual_seed", Mock())
        monkeypatch.setattr(utils.torch.cuda, "manual_seed_all", Mock(), raising=False)
        monkeypatch.setattr(utils.torch, "use_deterministic_algorithms", Mock())
        monkeypatch.setattr(utils.torch.backends, "cudnn", Mock(), raising=False)

        utils.set_seed(999)

        assert os.environ["PYTHONHASHSEED"] == "999"
        assert os.environ["PYTHONHASHSEED"] != original_value

    def test_set_seed_cudnn_settings_when_deterministic(self, monkeypatch):
        """Test that cudnn settings are set correctly when deterministic=True."""
        cudnn = Mock()
        cudnn.deterministic = False
        cudnn.benchmark = True
        
        monkeypatch.setattr(utils.random, "seed", Mock())
        monkeypatch.setattr(utils.np.random, "seed", Mock())
        monkeypatch.setattr(utils.torch, "manual_seed", Mock())
        monkeypatch.setattr(utils.torch.cuda, "manual_seed_all", Mock(), raising=False)
        monkeypatch.setattr(utils.torch, "use_deterministic_algorithms", Mock())
        monkeypatch.setattr(utils.torch.backends, "cudnn", cudnn, raising=False)

        utils.set_seed(42, deterministic=True)

        assert cudnn.deterministic is True
        assert cudnn.benchmark is False

    def test_set_seed_cudnn_not_modified_when_non_deterministic(self, monkeypatch):
        """Test that cudnn settings are not modified when deterministic=False."""
        cudnn = Mock()
        cudnn.deterministic = False
        cudnn.benchmark = True
        original_deterministic = cudnn.deterministic
        original_benchmark = cudnn.benchmark
        
        monkeypatch.setattr(utils.random, "seed", Mock())
        monkeypatch.setattr(utils.np.random, "seed", Mock())
        monkeypatch.setattr(utils.torch, "manual_seed", Mock())
        monkeypatch.setattr(utils.torch.cuda, "manual_seed_all", Mock(), raising=False)
        monkeypatch.setattr(utils.torch, "use_deterministic_algorithms", Mock())
        monkeypatch.setattr(utils.torch.backends, "cudnn", cudnn, raising=False)

        utils.set_seed(42, deterministic=False)

        assert cudnn.deterministic == original_deterministic
        assert cudnn.benchmark == original_benchmark


class TestGetLogger:
    """Comprehensive tests for get_logger function."""

    def test_get_logger_creates_new_logger(self):
        """Test that get_logger creates a new logger with correct name."""
        name = "mi_crow-test-new-logger"
        logger = utils.get_logger(name)
        
        assert logger.name == name
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO

    def test_get_logger_idempotent_handlers(self):
        """Test that get_logger is idempotent for handlers."""
        name = "mi_crow-test-logger"
        logger = utils.get_logger(name, level=logging.DEBUG)
        handler_count = len(logger.handlers)

        logger2 = utils.get_logger(name)

        assert logger is logger2
        assert len(logger.handlers) == handler_count
        assert logger.propagate is True

    @pytest.mark.parametrize("level", [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ])
    def test_get_logger_with_int_level(self, level):
        """Test get_logger with integer log levels."""
        name = f"mi_crow-test-{level}"
        logger = utils.get_logger(name, level=level)
        
        assert logger.level == level

    @pytest.mark.parametrize("level_str,expected", [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ])
    def test_get_logger_with_string_level(self, level_str, expected):
        """Test get_logger with string log levels (uppercase)."""
        name = f"mi_crow-test-{level_str}"
        logger = utils.get_logger(name, level=level_str)
        
        assert logger.level == expected

    def test_get_logger_with_lowercase_string_level_raises_error(self):
        """Test that lowercase string levels raise ValueError (current behavior)."""
        name = "mi_crow-test-lowercase"
        # getLevelName("debug") returns "Level debug" which is invalid
        with pytest.raises(ValueError, match="Unknown level"):
            utils.get_logger(name, level="debug")

    def test_get_logger_sets_propagation(self):
        """Test that get_logger sets propagate=True."""
        name = "mi_crow-test-propagate"
        logger = utils.get_logger(name)
        
        assert logger.propagate is True

    def test_get_logger_creates_handler_on_first_call(self):
        """Test that handler is created on first call."""
        name = "mi_crow-test-handler-creation"
        logger = utils.get_logger(name)
        
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_get_logger_handler_has_formatter(self):
        """Test that handler has correct formatter."""
        name = "mi_crow-test-formatter"
        logger = utils.get_logger(name)
        
        handler = logger.handlers[0]
        assert handler.formatter is not None
        assert isinstance(handler.formatter, logging.Formatter)

    def test_get_logger_formatter_format(self):
        """Test that formatter has correct format."""
        name = "mi_crow-test-format"
        logger = utils.get_logger(name)
        
        handler = logger.handlers[0]
        formatter = handler.formatter
        assert "%(asctime)s" in formatter._fmt
        assert "%(levelname)s" in formatter._fmt
        assert "%(name)s" in formatter._fmt
        assert "%(message)s" in formatter._fmt

    def test_get_logger_multiple_calls_same_name(self):
        """Test that multiple calls with same name return same logger."""
        name = "mi_crow-test-same"
        logger1 = utils.get_logger(name)
        logger2 = utils.get_logger(name)
        logger3 = utils.get_logger(name)
        
        assert logger1 is logger2
        assert logger2 is logger3

    def test_get_logger_different_names_different_loggers(self):
        """Test that different names create different loggers."""
        logger1 = utils.get_logger("mi_crow-test-1")
        logger2 = utils.get_logger("mi_crow-test-2")
        
        assert logger1 is not logger2
        assert logger1.name != logger2.name

    def test_get_logger_does_not_add_duplicate_handlers(self):
        """Test that calling get_logger multiple times doesn't add duplicate handlers."""
        name = "mi_crow-test-no-duplicates"
        logger = utils.get_logger(name)
        initial_handler_count = len(logger.handlers)
        
        # Call multiple times
        utils.get_logger(name)
        utils.get_logger(name)
        utils.get_logger(name)
        
        assert len(logger.handlers) == initial_handler_count

    def test_get_logger_preserves_existing_handlers(self):
        """Test that get_logger preserves existing handlers."""
        name = "mi_crow-test-preserve"
        logger = logging.getLogger(name)
        custom_handler = logging.StreamHandler()
        logger.addHandler(custom_handler)
        initial_handler_count = len(logger.handlers)
        
        # Call get_logger - should not add another handler
        result_logger = utils.get_logger(name)
        
        assert result_logger is logger
        assert len(logger.handlers) == initial_handler_count
        assert custom_handler in logger.handlers

    def test_get_logger_ensures_propagation_even_with_existing_handler(self):
        """Test that get_logger ensures propagation even if handler exists."""
        name = "mi_crow-test-propagation"
        logger = logging.getLogger(name)
        logger.propagate = False
        logger.addHandler(logging.StreamHandler())
        
        result_logger = utils.get_logger(name)
        
        assert result_logger.propagate is True

    def test_get_logger_with_invalid_string_level_raises_error(self):
        """Test that invalid string level raises ValueError."""
        name = "mi_crow-test-invalid-level"
        # getLevelName("INVALID_LEVEL") returns "Level INVALID_LEVEL" which is invalid
        with pytest.raises(ValueError, match="Unknown level"):
            utils.get_logger(name, level="INVALID_LEVEL")

    def test_get_logger_default_level_is_info(self):
        """Test that default level is INFO when not specified."""
        name = "mi_crow-test-default"
        logger = utils.get_logger(name)
        
        assert logger.level == logging.INFO

    def test_get_logger_returns_same_logger_different_levels(self):
        """Test that same logger is returned even with different level on second call."""
        name = "mi_crow-test-level-change"
        logger1 = utils.get_logger(name, level=logging.DEBUG)
        logger2 = utils.get_logger(name, level=logging.WARNING)
        
        assert logger1 is logger2
        # Level should be updated to WARNING from second call
        assert logger2.level == logging.WARNING

