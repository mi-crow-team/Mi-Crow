"""Tests for TopKSae."""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

from amber.mechanistic.sae.modules.topk_sae import TopKSae
from amber.hooks.hook import HookType


class TestTopKSaeInitialization:
    """Tests for TopKSae initialization."""

    @patch('amber.mechanistic.sae.modules.topk_sae.OvercompleteTopkSAE')
    def test_init_with_parameters(self, mock_topk_sae_class):
        """Test initialization with parameters."""
        mock_engine = MagicMock()
        mock_topk_sae_class.return_value = mock_engine
        
        sae = TopKSae(n_latents=100, n_inputs=200, k=10)
        
        assert sae.k == 10
        assert sae.context.n_latents == 100
        assert sae.context.n_inputs == 200
        mock_topk_sae_class.assert_called_once()

    @patch('amber.mechanistic.sae.modules.topk_sae.OvercompleteTopkSAE')
    def test_init_creates_engine(self, mock_topk_sae_class):
        """Test that SAE engine is created."""
        mock_engine = MagicMock()
        mock_topk_sae_class.return_value = mock_engine
        
        sae = TopKSae(n_latents=100, n_inputs=200, k=10)
        assert sae.sae_engine == mock_engine


class TestTopKSaeMethods:
    """Tests for TopKSae methods."""

    @patch('amber.mechanistic.sae.modules.topk_sae.OvercompleteTopkSAE')
    def test_encode(self, mock_topk_sae_class):
        """Test encode method."""
        mock_engine = MagicMock()
        mock_engine.encode.return_value = (None, torch.randn(5, 100))
        mock_topk_sae_class.return_value = mock_engine
        
        sae = TopKSae(n_latents=100, n_inputs=200, k=10)
        x = torch.randn(5, 200)
        result = sae.encode(x)
        
        assert result is not None
        mock_engine.encode.assert_called_once_with(x)

    @patch('amber.mechanistic.sae.modules.topk_sae.OvercompleteTopkSAE')
    def test_decode(self, mock_topk_sae_class):
        """Test decode method."""
        mock_engine = MagicMock()
        mock_engine.decode.return_value = torch.randn(5, 200)
        mock_topk_sae_class.return_value = mock_engine
        
        sae = TopKSae(n_latents=100, n_inputs=200, k=10)
        x = torch.randn(5, 100)
        result = sae.decode(x)
        
        assert result is not None
        mock_engine.decode.assert_called_once_with(x)

    @patch('amber.mechanistic.sae.modules.topk_sae.OvercompleteTopkSAE')
    def test_forward(self, mock_topk_sae_class):
        """Test forward method."""
        mock_engine = MagicMock()
        mock_engine.forward.return_value = (None, None, torch.randn(5, 200))
        mock_topk_sae_class.return_value = mock_engine
        
        sae = TopKSae(n_latents=100, n_inputs=200, k=10)
        x = torch.randn(5, 200)
        result = sae.forward(x)
        
        assert result is not None
        mock_engine.forward.assert_called_once_with(x)

