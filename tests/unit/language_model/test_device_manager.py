"""Tests for device_manager module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from mi_crow.language_model.device_manager import (
    ensure_context_device,
    normalize_device,
    sync_model_to_context_device,
)
from tests.unit.fixtures.language_models import create_language_model_from_mock
from tests.unit.fixtures.models import create_mock_model
from tests.unit.fixtures.stores import create_temp_store
from tests.unit.fixtures.tokenizers import create_mock_tokenizer


class TestNormalizeDevice:
    """Tests for normalize_device function."""

    def test_normalize_device_none_returns_cpu(self):
        """Test that None device normalizes to 'cpu'."""
        result = normalize_device(None)
        assert result == "cpu"

    def test_normalize_device_cpu_string(self):
        """Test that 'cpu' string is returned as-is."""
        result = normalize_device("cpu")
        assert result == "cpu"

    def test_normalize_device_torch_device_cpu(self):
        """Test that torch.device('cpu') normalizes to 'cpu'."""
        result = normalize_device(torch.device("cpu"))
        assert result == "cpu"

    def test_normalize_device_cuda_normalized_to_cuda_0(self):
        """Test that 'cuda' normalizes to 'cuda:0' when CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        result = normalize_device("cuda")
        assert result == "cuda:0"

    def test_normalize_device_cuda_0_preserved(self):
        """Test that 'cuda:0' is preserved when CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        result = normalize_device("cuda:0")
        assert result == "cuda:0"

    def test_normalize_device_cuda_raises_when_not_available(self):
        """Test that 'cuda' raises ValueError when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(ValueError, match="CUDA is not available"):
                normalize_device("cuda")

    def test_normalize_device_mps_when_available(self):
        """Test that 'mps' works when MPS is available."""
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())
        if not mps_available:
            pytest.skip("MPS not available")

        result = normalize_device("mps")
        assert result == "mps"

    def test_normalize_device_mps_raises_when_not_available(self):
        """Test that 'mps' raises ValueError when MPS is not available."""
        with patch("torch.backends.mps.is_available", return_value=False):
            with pytest.raises(ValueError, match="MPS is not available"):
                normalize_device("mps")

    def test_normalize_device_torch_device_cuda(self):
        """Test that torch.device('cuda:0') normalizes correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        result = normalize_device(torch.device("cuda:0"))
        assert result == "cuda:0"


class TestEnsureContextDevice:
    """Tests for ensure_context_device function."""

    def test_ensure_context_device_valid(self, temp_store):
        """Test that valid context.device returns torch.device."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        from mi_crow.language_model.language_model import LanguageModel

        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device="cpu")
        device = ensure_context_device(lm)
        assert isinstance(device, torch.device)
        assert str(device) == "cpu"

    def test_ensure_context_device_cuda(self, temp_store):
        """Test that CUDA context.device returns correct torch.device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        from mi_crow.language_model.language_model import LanguageModel

        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device="cuda")
        device = ensure_context_device(lm)
        assert isinstance(device, torch.device)
        assert str(device) == "cuda:0"

    def test_ensure_context_device_raises_when_no_context(self):
        """Test that missing context raises ValueError."""
        mock_lm = MagicMock()
        del mock_lm.context
        with pytest.raises(ValueError, match="context.device"):
            ensure_context_device(mock_lm)

    def test_ensure_context_device_raises_when_no_device_attr(self):
        """Test that missing device attribute raises ValueError."""
        mock_lm = MagicMock()
        mock_lm.context = MagicMock()
        del mock_lm.context.device
        with pytest.raises(ValueError, match="context.device"):
            ensure_context_device(mock_lm)

    def test_ensure_context_device_raises_when_device_none(self):
        """Test that None device raises ValueError."""
        mock_lm = MagicMock()
        mock_lm.context = MagicMock()
        mock_lm.context.device = None
        with pytest.raises(ValueError, match="context.device"):
            ensure_context_device(mock_lm)


class TestSyncModelToContextDevice:
    """Tests for sync_model_to_context_device function."""

    def test_sync_model_no_op_when_already_on_device(self, temp_store):
        """Test that sync is no-op when model is already on context.device."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        from mi_crow.language_model.language_model import LanguageModel

        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device="cpu")
        sync_model_to_context_device(lm)
        from mi_crow.language_model.utils import get_device_from_model

        model_device = get_device_from_model(lm.model)
        assert model_device.type == "cpu"

    def test_sync_model_moves_from_cuda_to_cpu(self, temp_store):
        """Test that model is moved from CUDA to CPU when context.device is CPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = create_mock_model()
        model = model.cuda()
        tokenizer = create_mock_tokenizer()
        from mi_crow.language_model.language_model import LanguageModel

        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device="cpu")
        from mi_crow.language_model.utils import get_device_from_model

        model_device_before = get_device_from_model(lm.model)
        assert model_device_before.type == "cuda"
        sync_model_to_context_device(lm)
        model_device_after = get_device_from_model(lm.model)
        assert model_device_after.type == "cpu"

    def test_sync_model_moves_from_cpu_to_cuda(self, temp_store):
        """Test that model is moved from CPU to CUDA when context.device is CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        from mi_crow.language_model.language_model import LanguageModel

        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device="cuda")
        from mi_crow.language_model.utils import get_device_from_model

        model_device_after = get_device_from_model(lm.model)
        assert model_device_after.type == "cuda"
        assert str(model_device_after) == "cuda:0"

    def test_sync_model_raises_when_no_context(self):
        """Test that sync raises ValueError when context is missing."""
        mock_lm = MagicMock()
        del mock_lm.context
        with pytest.raises(ValueError, match="context.device"):
            sync_model_to_context_device(mock_lm)

    def test_sync_model_raises_when_device_none(self):
        """Test that sync raises ValueError when device is None."""
        mock_lm = MagicMock()
        mock_lm.context = MagicMock()
        mock_lm.context.device = None
        mock_lm.model = create_mock_model()
        with pytest.raises(ValueError, match="context.device"):
            sync_model_to_context_device(mock_lm)
