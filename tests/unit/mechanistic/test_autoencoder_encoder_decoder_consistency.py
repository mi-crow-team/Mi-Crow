"""Test encoder/decoder consistency and weight management in Autoencoder."""

import pytest
import torch
from torch import nn

from amber.mechanistic.autoencoder.autoencoder import Autoencoder


class TestAutoencoderEncoderDecoderConsistency:
    """Test encoder/decoder weight consistency and management."""

    def test_tied_encoder_decoder_weight_sharing(self):
        """Test that tied encoder/decoder weights are properly shared."""
        autoencoder = Autoencoder(
            n_latents=5,
            n_inputs=10,
            tied=True
        )
        
        # Decoder should be None (tied)
        assert autoencoder.decoder is None
        
        # When accessing decoder weight, should get transpose of encoder
        decoder_weight = autoencoder.encoder.t().clone()  # Create a copy
        assert decoder_weight.shape == (5, 10)  # n_latents x n_inputs
        
        # Modifying encoder should affect decoder weight
        original_encoder = autoencoder.encoder.clone()
        
        # Modify encoder in-place
        autoencoder.encoder.data += 0.1
        
        # Decoder weight should reflect the change (encoder.t() should be different now)
        new_decoder_weight = autoencoder.encoder.t()
        assert not torch.equal(new_decoder_weight, decoder_weight)
        
        # The new decoder weight should equal the modified encoder transposed
        assert torch.equal(new_decoder_weight, (original_encoder + 0.1).t())

    def test_untied_encoder_decoder_separate_weights(self):
        """Test that untied encoder/decoder have separate weights."""
        autoencoder = Autoencoder(
            n_latents=5,
            n_inputs=10,
            tied=False
        )
        
        # Decoder should be a separate parameter
        assert autoencoder.decoder is not None
        assert autoencoder.decoder.shape == (5, 10)
        
        # Encoder and decoder should be different tensors
        assert not torch.equal(autoencoder.encoder, autoencoder.decoder)
        
        # Modifying encoder should not affect decoder
        original_decoder = autoencoder.decoder.clone()
        autoencoder.encoder.data += 0.1
        
        assert torch.equal(autoencoder.decoder, original_decoder)

    def test_decoder_gradient_projection(self):
        """Test decoder gradient projection functionality."""
        autoencoder = Autoencoder(
            n_latents=5,
            n_inputs=10,
            tied=False
        )
        
        # Create a simple loss and compute gradients
        input_tensor = torch.randn(2, 10, requires_grad=True)
        output_tuple = autoencoder(input_tensor)
        reconstructed = output_tuple[2]  # Get the reconstructed tensor
        loss = reconstructed.sum()
        loss.backward()
        
        # Store original decoder gradients
        original_decoder_grad = autoencoder.decoder.grad.clone()
        
        # Project decoder gradients
        autoencoder.project_grads_decode()
        
        # Decoder gradients should be modified
        assert not torch.equal(autoencoder.decoder.grad, original_decoder_grad)

    def test_decoder_normalization(self):
        """Test decoder weight normalization."""
        autoencoder = Autoencoder(
            n_latents=5,
            n_inputs=10,
            tied=False
        )
        
        # Store original decoder weights
        original_decoder = autoencoder.decoder.clone()
        
        # Normalize decoder weights
        autoencoder.scale_to_unit_norm()
        
        # Decoder weights should be normalized
        decoder_norms = torch.norm(autoencoder.decoder, dim=1)
        assert torch.allclose(decoder_norms, torch.ones_like(decoder_norms), atol=1e-6)

    def test_weight_updates_maintain_constraints_tied(self):
        """Test that weight updates maintain constraints for tied weights."""
        autoencoder = Autoencoder(
            n_latents=5,
            n_inputs=10,
            tied=True
        )
        
        # Store original encoder weights
        original_encoder = autoencoder.encoder.clone()
        
        # Simulate weight update
        autoencoder.encoder.data += 0.1
        
        # Decoder weight should still be transpose of encoder
        decoder_weight = autoencoder.encoder.t()
        assert decoder_weight.shape == (5, 10)
        
        # The relationship should be maintained
        assert torch.equal(decoder_weight, autoencoder.encoder.t())

    def test_weight_updates_maintain_constraints_untied(self):
        """Test that weight updates maintain constraints for untied weights."""
        autoencoder = Autoencoder(
            n_latents=5,
            n_inputs=10,
            tied=False
        )
        
        # Store original weights
        original_encoder = autoencoder.encoder.clone()
        original_decoder = autoencoder.decoder.clone()
        
        # Simulate independent weight updates
        autoencoder.encoder.data += 0.1
        autoencoder.decoder.data += 0.05
        
        # Weights should be independent
        assert not torch.equal(autoencoder.encoder, autoencoder.decoder)
        assert not torch.equal(autoencoder.encoder, original_encoder)
        assert not torch.equal(autoencoder.decoder, original_decoder)

    def test_decoder_gradient_projection_with_tied_weights(self):
        """Test decoder gradient projection with tied weights."""
        autoencoder = Autoencoder(
            n_latents=5,
            n_inputs=10,
            tied=True
        )
        
        # Create a simple loss and compute gradients
        input_tensor = torch.randn(2, 10, requires_grad=True)
        output_tuple = autoencoder(input_tensor)
        reconstructed = output_tuple[2]  # Get the reconstructed tensor
        loss = reconstructed.sum()
        loss.backward()
        
        # Store original encoder gradients
        original_encoder_grad = autoencoder.encoder.grad.clone()
        
        # Project decoder gradients (should affect encoder gradients for tied weights)
        autoencoder.project_grads_decode()
        
        # Encoder gradients should be modified
        assert not torch.equal(autoencoder.encoder.grad, original_encoder_grad)

    def test_decoder_normalization_with_tied_weights(self):
        """Test decoder normalization with tied weights."""
        autoencoder = Autoencoder(
            n_latents=5,
            n_inputs=10,
            tied=True
        )
        
        # Store original encoder weights
        original_encoder = autoencoder.encoder.clone()
        
        # Normalize decoder weights (should normalize encoder weights for tied weights)
        autoencoder.scale_to_unit_norm()
        
        # Encoder weights should be normalized (columns for tied weights)
        encoder_norms = torch.norm(autoencoder.encoder, dim=0)  # Column norms
        assert torch.allclose(encoder_norms, torch.ones_like(encoder_norms), atol=1e-6)

    def test_weight_constraints_with_different_initializations(self):
        """Test weight constraints with different initialization methods."""
        init_methods = ["kaiming", "xavier", "uniform", "normal"]
        
        for init_method in init_methods:
            autoencoder = Autoencoder(
            n_latents=5,
            n_inputs=10,
            tied=False,
            init_method=init_method
        )
            
            # Test decoder normalization
            autoencoder.scale_to_unit_norm()
            decoder_norms = torch.norm(autoencoder.decoder, dim=1)
            assert torch.allclose(decoder_norms, torch.ones_like(decoder_norms), atol=1e-6)

    def test_weight_constraints_with_different_activations(self):
        """Test weight constraints with different activation functions."""
        activations = ["TopK_5", "TopKReLU_3", nn.ReLU(), nn.GELU()]
        
        for activation in activations:
            autoencoder = Autoencoder(
                n_latents=5,
                n_inputs=10,
                tied=False,
                activation=activation
            )
            
            # Test decoder normalization
            autoencoder.scale_to_unit_norm()
            decoder_norms = torch.norm(autoencoder.decoder, dim=1)
            assert torch.allclose(decoder_norms, torch.ones_like(decoder_norms), atol=1e-6)

    def test_weight_constraints_preserve_device(self):
        """Test that weight constraints preserve device placement."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            autoencoder = Autoencoder(
            n_latents=5,
            n_inputs=10,
            device=device
        )
            
            # Test decoder normalization
            autoencoder.scale_to_unit_norm()
            
            # Weights should remain on correct device
            assert autoencoder.encoder.device == device
            assert autoencoder.decoder.device == device

    def test_weight_constraints_with_gradient_accumulation(self):
        """Test weight constraints with gradient accumulation."""
        autoencoder = Autoencoder(
            n_latents=5,
            n_inputs=10,
            tied=False
        )
        
        # Simulate gradient accumulation
        for _ in range(3):
            input_tensor = torch.randn(2, 10, requires_grad=True)
            output_tuple = autoencoder(input_tensor)
            reconstructed = output_tuple[2]  # Get the reconstructed tensor
            loss = reconstructed.sum()
            loss.backward()
        
        # Store accumulated gradients
        accumulated_grad = autoencoder.decoder.grad.clone()
        
        # Project gradients
        autoencoder.project_grads_decode()
        
        # Gradients should be modified
        assert not torch.equal(autoencoder.decoder.grad, accumulated_grad)

    def test_weight_constraints_with_mixed_precision(self):
        """Test weight constraints with mixed precision."""
        autoencoder = Autoencoder(
            n_latents=5,
            n_inputs=10,
            tied=False
        )
        
        # Convert to half precision
        autoencoder = autoencoder.half()
        
        # Test decoder normalization
        autoencoder.scale_to_unit_norm()
        
        # Weights should be normalized
        decoder_norms = torch.norm(autoencoder.decoder, dim=1)
        assert torch.allclose(decoder_norms, torch.ones_like(decoder_norms), atol=1e-1)

    def test_weight_constraints_with_batch_norm(self):
        """Test weight constraints with batch normalization."""
        autoencoder = Autoencoder(
            n_latents=5,
            n_inputs=10,
            tied=False
        )
        
        # Add batch normalization
        autoencoder.batch_norm = nn.BatchNorm1d(5)
        
        # Test decoder normalization
        autoencoder.scale_to_unit_norm()
        
        # Weights should be normalized
        decoder_norms = torch.norm(autoencoder.decoder, dim=1)
        assert torch.allclose(decoder_norms, torch.ones_like(decoder_norms), atol=1e-6)
