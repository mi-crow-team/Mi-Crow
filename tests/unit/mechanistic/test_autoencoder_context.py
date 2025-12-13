"""Tests for AutoencoderContext."""

import pytest
from unittest.mock import Mock

from amber.mechanistic.sae.autoencoder_context import AutoencoderContext
from tests.unit.mechanistic.test_sae_base import ConcreteSae


class TestAutoencoderContext:
    """Tests for AutoencoderContext dataclass."""

    def test_context_initialization(self):
        """Test context initialization with required parameters."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        context = AutoencoderContext(
            autoencoder=sae,
            n_latents=100,
            n_inputs=200
        )
        
        assert context.autoencoder == sae
        assert context.n_latents == 100
        assert context.n_inputs == 200

    def test_context_default_values(self):
        """Test context default values."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        context = AutoencoderContext(
            autoencoder=sae,
            n_latents=100,
            n_inputs=200
        )
        
        assert context.lm is None
        assert context.lm_layer_signature is None
        assert context.model_id is None
        assert context.device == 'cpu'
        assert context.experiment_name is None
        assert context.run_id is None
        assert context.text_tracking_enabled is False
        assert context.text_tracking_k == 5
        assert context.text_tracking_negative is False
        assert context.store is None
        assert context.tied is False
        assert context.bias_init == 0.0
        assert context.init_method == "kaiming"

    def test_context_custom_values(self):
        """Test context with custom values."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        store = Mock()
        lm = Mock()
        
        context = AutoencoderContext(
            autoencoder=sae,
            n_latents=150,
            n_inputs=250,
            lm=lm,
            lm_layer_signature=5,
            model_id="test_model",
            device="cuda",
            experiment_name="test_exp",
            run_id="test_run",
            text_tracking_enabled=True,
            text_tracking_k=10,
            text_tracking_negative=True,
            store=store,
            tied=True,
            bias_init=0.1,
            init_method="xavier"
        )
        
        assert context.n_latents == 150
        assert context.n_inputs == 250
        assert context.lm == lm
        assert context.lm_layer_signature == 5
        assert context.model_id == "test_model"
        assert context.device == "cuda"
        assert context.experiment_name == "test_exp"
        assert context.run_id == "test_run"
        assert context.text_tracking_enabled is True
        assert context.text_tracking_k == 10
        assert context.text_tracking_negative is True
        assert context.store == store
        assert context.tied is True
        assert context.bias_init == 0.1
        assert context.init_method == "xavier"

    def test_context_string_layer_signature(self):
        """Test context with string layer signature."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        context = AutoencoderContext(
            autoencoder=sae,
            n_latents=100,
            n_inputs=200,
            lm_layer_signature="layer_5"
        )
        
        assert context.lm_layer_signature == "layer_5"

    def test_context_int_layer_signature(self):
        """Test context with int layer signature."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        context = AutoencoderContext(
            autoencoder=sae,
            n_latents=100,
            n_inputs=200,
            lm_layer_signature=5
        )
        
        assert context.lm_layer_signature == 5

    def test_context_mutable_fields(self):
        """Test that context fields are mutable."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        context = AutoencoderContext(
            autoencoder=sae,
            n_latents=100,
            n_inputs=200
        )
        
        context.device = "cuda"
        context.text_tracking_enabled = True
        context.text_tracking_k = 10
        
        assert context.device == "cuda"
        assert context.text_tracking_enabled is True
        assert context.text_tracking_k == 10
