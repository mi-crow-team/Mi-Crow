"""Tests for Sae abstract base class."""

import pytest
import torch
from unittest.mock import Mock, MagicMock

from amber.mechanistic.sae.sae import Sae
from amber.hooks.hook import HookType


class ConcreteSae(Sae):
    """Concrete implementation of Sae for testing."""

    def _initialize_sae_engine(self):
        mock_engine = MagicMock()
        return mock_engine

    def modify_activations(self, module, inputs, output):
        return output * 0.5

    def encode(self, x):
        return x

    def decode(self, x):
        return x

    def forward(self, x):
        return x

    def save(self, name):
        pass

    @staticmethod
    def load(path):
        return ConcreteSae(10, 20)


class TestSaeInitialization:
    """Tests for Sae initialization."""

    def test_init_with_parameters(self):
        """Test initialization with parameters."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        
        assert sae.context.n_latents == 100
        assert sae.context.n_inputs == 200
        assert sae.context.device == 'cpu'
        assert sae.hook_type == HookType.FORWARD

    def test_init_with_device(self):
        """Test initialization with device."""
        sae = ConcreteSae(n_latents=100, n_inputs=200, device='cpu')
        assert sae.context.device == 'cpu'

    def test_init_creates_context(self):
        """Test that context is created."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        assert sae.context is not None
        assert sae.context.autoencoder == sae

    def test_init_creates_trainer(self):
        """Test that trainer is created."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        assert sae.trainer is not None


class TestSaeMethods:
    """Tests for Sae methods."""

    def test_encode(self):
        """Test encode method."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        x = torch.randn(5, 200)
        result = sae.encode(x)
        assert result is not None

    def test_decode(self):
        """Test decode method."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        x = torch.randn(5, 100)
        result = sae.decode(x)
        assert result is not None

    def test_forward(self):
        """Test forward method."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        x = torch.randn(5, 200)
        result = sae.forward(x)
        assert result is not None

    def test_modify_activations(self):
        """Test modify_activations method."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        from torch import nn
        module = nn.Linear(200, 100)
        output = torch.randn(5, 100)
        
        result = sae.modify_activations(module, None, output)
        assert result is not None
        assert torch.allclose(result, output * 0.5)


class TestSaeStaticMethods:
    """Tests for Sae static methods."""

    def test_apply_activation_fn_relu(self):
        """Test _apply_activation_fn with relu."""
        tensor = torch.tensor([-1.0, 0.0, 1.0])
        result = Sae._apply_activation_fn(tensor, "relu")
        assert torch.equal(result, torch.relu(tensor))

    def test_apply_activation_fn_linear(self):
        """Test _apply_activation_fn with linear."""
        tensor = torch.tensor([-1.0, 0.0, 1.0])
        result = Sae._apply_activation_fn(tensor, "linear")
        assert torch.equal(result, tensor)

    def test_apply_activation_fn_none(self):
        """Test _apply_activation_fn with None."""
        tensor = torch.tensor([-1.0, 0.0, 1.0])
        result = Sae._apply_activation_fn(tensor, None)
        assert torch.equal(result, tensor)

    def test_apply_activation_fn_invalid_raises_error(self):
        """Test that invalid activation function raises ValueError."""
        tensor = torch.tensor([1.0])
        with pytest.raises(ValueError, match="Unknown activation function"):
            Sae._apply_activation_fn(tensor, "invalid")

