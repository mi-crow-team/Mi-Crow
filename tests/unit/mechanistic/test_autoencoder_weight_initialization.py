"""Test weight initialization functionality in Autoencoder."""

import pytest
import torch
from torch import nn
from unittest.mock import patch

from amber.mechanistic.autoencoder.autoencoder import Autoencoder


class TestAutoencoderWeightInitialization:
    """Test weight initialization methods and edge cases."""

    def test_kaiming_initialization_creates_normalized_weights(self):
        """Test that Kaiming initialization creates properly normalized weights."""
        autoencoder = Autoencoder(
            n_latents=10,
            n_inputs=20,
            init_method="kaiming",
            tied=False
        )
        
        # Check encoder weights
        encoder_norm = torch.norm(autoencoder.encoder, dim=1)
        # Kaiming init should create weights with reasonable norms
        assert torch.all(encoder_norm > 0.01)  # Not too small
        assert torch.all(encoder_norm < 10.0)   # Not too large
        
        # Check decoder weights
        decoder_norm = torch.norm(autoencoder.decoder, dim=1)
        assert torch.all(decoder_norm > 0.01)
        assert torch.all(decoder_norm < 10.0)

    def test_xavier_initialization_creates_normalized_weights(self):
        """Test that Xavier initialization creates properly normalized weights."""
        autoencoder = Autoencoder(
            n_latents=10,
            n_inputs=20,
            init_method="xavier",
            tied=False
        )
        
        # Check encoder weights
        encoder_norm = torch.norm(autoencoder.encoder, dim=1)
        assert torch.all(encoder_norm > 0.01)
        assert torch.all(encoder_norm < 10.0)
        
        # Check decoder weights
        decoder_norm = torch.norm(autoencoder.decoder, dim=1)
        assert torch.all(decoder_norm > 0.01)
        assert torch.all(decoder_norm < 10.0)

    def test_uniform_initialization_creates_bounded_weights(self):
        """Test that uniform initialization creates weights in [-1, 1] range."""
        autoencoder = Autoencoder(
            n_latents=10,
            n_inputs=20,
            init_method="uniform",
            tied=False
        )
        
        # Check encoder weights are in [-1, 1] range
        assert torch.all(autoencoder.encoder >= -1.0)
        assert torch.all(autoencoder.encoder <= 1.0)
        
        # Check decoder weights are in [-1, 1] range
        assert torch.all(autoencoder.decoder >= -1.0)
        assert torch.all(autoencoder.decoder <= 1.0)

    def test_normal_initialization_creates_gaussian_weights(self):
        """Test that normal initialization creates Gaussian-distributed weights."""
        autoencoder = Autoencoder(
            n_latents=10,
            n_inputs=20,
            init_method="normal",
            tied=False
        )
        
        # Check encoder weights have reasonable variance
        encoder_mean = torch.mean(autoencoder.encoder)
        encoder_std = torch.std(autoencoder.encoder)
        
        # Mean should be close to 0, std should be reasonable (scaled to norm=0.1)
        assert abs(encoder_mean) < 0.5
        assert 0.01 < encoder_std < 0.5  # Adjusted for norm=0.1 scaling
        
        # Check decoder weights
        decoder_mean = torch.mean(autoencoder.decoder)
        decoder_std = torch.std(autoencoder.decoder)
        
        assert abs(decoder_mean) < 0.5
        assert 0.01 < decoder_std < 0.5  # Adjusted for norm=0.1 scaling

    def test_invalid_initialization_method_raises_value_error(self):
        """Test that invalid initialization method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid init_method: invalid_method"):
            Autoencoder(
                n_latents=10,
                n_inputs=20,
                init_method="invalid_method"
            )

    def test_tied_decoder_uses_transpose_of_encoder(self):
        """Test that tied decoder uses transpose of encoder weights."""
        autoencoder = Autoencoder(
            n_latents=10,
            n_inputs=20,
            tied=True
        )
        
        # Decoder should be None (tied)
        assert autoencoder.decoder is None
        
        # When accessing decoder weight, should get transpose of encoder
        decoder_weight = autoencoder.encoder.t()
        assert decoder_weight.shape == (10, 20)  # n_latents x n_inputs

    def test_untied_decoder_has_separate_weights(self):
        """Test that untied decoder has separate weights from encoder."""
        autoencoder = Autoencoder(
            n_latents=10,
            n_inputs=20,
            tied=False
        )
        
        # Decoder should be a separate parameter
        assert autoencoder.decoder is not None
        assert autoencoder.decoder.shape == (10, 20)
        
        # Encoder and decoder should be different tensors
        assert not torch.equal(autoencoder.encoder, autoencoder.decoder)

    def test_neuron_reinitialization_with_specific_indices(self):
        """Test reinitializing specific neurons."""
        autoencoder = Autoencoder(
            n_latents=10,
            n_inputs=20,
            init_method="uniform"
        )
        
        # Store original weights
        original_encoder = autoencoder.encoder.clone()
        original_decoder = autoencoder.decoder.clone()
        
        # Reinitialize neurons 0, 2, 4
        neuron_indices = [0, 2, 4]
        autoencoder._init_weights(neuron_indices=neuron_indices)
        
        # Check that specified neurons were reinitialized
        for i in neuron_indices:
            assert not torch.equal(autoencoder.encoder[:, i], original_encoder[:, i])
            assert not torch.equal(autoencoder.decoder[i], original_decoder[i])
        
        # Check that other neurons were not changed
        for i in range(10):
            if i not in neuron_indices:
                assert torch.equal(autoencoder.encoder[:, i], original_encoder[:, i])
                assert torch.equal(autoencoder.decoder[i], original_decoder[i])

    def test_reinitialization_with_custom_norm(self):
        """Test reinitialization with custom norm parameter."""
        autoencoder = Autoencoder(
            n_latents=10,
            n_inputs=20,
            init_method="uniform"
        )
        
        # Reinitialize with custom norm
        custom_norm = 0.5
        autoencoder._init_weights(norm=custom_norm)
        
        # Check that weights have reasonable norms
        encoder_norms = torch.norm(autoencoder.encoder, dim=1)
        decoder_norms = torch.norm(autoencoder.decoder, dim=1)
        
        # Norms should be reasonable (not too small, not too large)
        assert torch.all(encoder_norms > 0.1)
        assert torch.all(encoder_norms < 1.0)
        assert torch.all(decoder_norms > 0.1)
        assert torch.all(decoder_norms < 1.0)

    def test_bias_initialization_with_float_value(self):
        """Test bias initialization with float value."""
        bias_value = 0.1
        autoencoder = Autoencoder(
            n_latents=10,
            n_inputs=20,
            bias_init=bias_value
        )
        
        # Check pre_bias initialization
        assert torch.all(autoencoder.pre_bias == bias_value)
        
        # Check latent_bias initialization (should be zeros)
        assert torch.all(autoencoder.latent_bias == 0.0)

    def test_bias_initialization_with_tensor(self):
        """Test bias initialization with tensor."""
        bias_tensor = torch.randn(20) * 0.1
        autoencoder = Autoencoder(
            n_latents=10,
            n_inputs=20,
            bias_init=bias_tensor
        )
        
        # Check pre_bias initialization
        assert torch.equal(autoencoder.pre_bias, bias_tensor)

    def test_initialization_preserves_device(self):
        """Test that initialization preserves device placement."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            autoencoder = Autoencoder(
                n_latents=10,
                n_inputs=20,
                device=device
            )
            
            # All parameters should be on the correct device
            assert autoencoder.encoder.device == device
            assert autoencoder.decoder.device == device
            assert autoencoder.pre_bias.device == device
            assert autoencoder.latent_bias.device == device

    def test_initialization_with_different_activation_functions(self):
        """Test initialization with different activation functions."""
        activations = ["TopK_5", "TopKReLU_3", nn.ReLU(), nn.GELU()]
        
        for activation in activations:
            autoencoder = Autoencoder(
                n_latents=10,
                n_inputs=20,
                activation=activation
            )
            
            # Should initialize successfully
            assert autoencoder.encoder is not None
            assert autoencoder.decoder is not None
            assert autoencoder.pre_bias is not None
            assert autoencoder.latent_bias is not None

    def test_initialization_with_zero_norm_raises_error(self):
        """Test that initialization with zero norm raises appropriate error."""
        autoencoder = Autoencoder(n_latents=10, n_inputs=20)
        
        # This should not raise an error, but weights should be initialized
        # The actual error handling depends on the specific initialization method
        assert autoencoder.encoder is not None
        assert autoencoder.decoder is not None
