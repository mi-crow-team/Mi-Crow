import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch

from amber.mechanistic.autoencoder.autoencoder import Autoencoder
from amber.mechanistic.autoencoder.modules.topk import TopK


def test_save_with_normalization_metadata(tmp_path):
    """Test saving sae with dataset normalization metadata."""
    ae = Autoencoder(n_latents=8, n_inputs=16, activation="TopK_4")
    
    # Add some test data
    x = torch.randn(10, 16)
    _ = ae(x)
    
    # Test save with normalization metadata
    dataset_normalize = True
    dataset_target_norm = False
    dataset_mean = torch.randn(16)
    run_metadata = {"epoch": 10, "loss": 0.5}
    
    save_path = tmp_path / "test_model"
    ae.save(
        "test_autoencoder",
        path=save_path,
        dataset_normalize=dataset_normalize,
        dataset_target_norm=dataset_target_norm,
        dataset_mean=dataset_mean,
        run_metadata=run_metadata
    )
    
    # Verify file was created
    model_file = save_path / "test_autoencoder.pt"
    assert model_file.exists()
    
    # Load and verify metadata
    loaded_payload = torch.load(model_file, map_location="cpu")
    assert loaded_payload["dataset_normalize"] == dataset_normalize
    assert loaded_payload["dataset_target_norm"] == dataset_target_norm
    assert torch.allclose(loaded_payload["dataset_mean"], dataset_mean)
    assert loaded_payload["run_metadata"] == run_metadata


def test_save_without_normalization_metadata(tmp_path):
    """Test saving sae without normalization metadata."""
    ae = Autoencoder(n_latents=6, n_inputs=12, activation="TopK_4")
    
    # Add some test data
    x = torch.randn(5, 12)
    _ = ae(x)
    
    save_path = tmp_path / "test_model_no_norm"
    ae.save("test_autoencoder_no_norm", path=save_path)
    
    # Verify file was created
    model_file = save_path / "test_autoencoder_no_norm.pt"
    assert model_file.exists()
    
    # Load and verify no normalization metadata
    loaded_payload = torch.load(model_file, map_location="cpu")
    assert "dataset_normalize" not in loaded_payload
    assert "dataset_target_norm" not in loaded_payload
    assert "dataset_mean" not in loaded_payload
    assert "run_metadata" not in loaded_payload


def test_save_activation_serialization(tmp_path):
    """Test that activation functions are properly serialized."""
    # Test with TopK activation
    ae_topk = Autoencoder(n_latents=4, n_inputs=8, activation="TopK_3")
    save_path = tmp_path / "topk_model"
    ae_topk.save("topk_autoencoder", path=save_path)
    
    # Load and verify activation serialization
    loaded_payload = torch.load(save_path / "topk_autoencoder.pt", map_location="cpu")
    assert loaded_payload["activation"] == "TopK_3"
    
    # Test with TopKReLU activation
    ae_topk_relu = Autoencoder(n_latents=4, n_inputs=8, activation="TopKReLU_2")
    save_path_relu = tmp_path / "topk_relu_model"
    ae_topk_relu.save("topk_relu_autoencoder", path=save_path_relu)
    
    # Load and verify activation serialization
    loaded_payload_relu = torch.load(save_path_relu / "topk_relu_autoencoder.pt", map_location="cpu")
    assert loaded_payload_relu["activation"] == "TopKReLU_2"


def test_load_with_metadata(tmp_path):
    """Test loading sae with metadata."""
    # Create and save a model with metadata
    ae = Autoencoder(n_latents=6, n_inputs=10, activation="TopK_3")
    
    dataset_normalize = True
    dataset_target_norm = True
    dataset_mean = torch.randn(10)
    run_metadata = {"training_epochs": 50, "final_loss": 0.1}
    
    save_path = tmp_path / "metadata_model"
    ae.save(
        "metadata_autoencoder",
        path=save_path,
        dataset_normalize=dataset_normalize,
        dataset_target_norm=dataset_target_norm,
        dataset_mean=dataset_mean,
        run_metadata=run_metadata
    )
    
    # Load using load_model static method
    model_file = save_path / "metadata_autoencoder.pt"
    loaded_model, norm, target_norm, mean = Autoencoder.load_model(model_file)
    
    # Verify loaded model properties
    assert loaded_model.context.n_latents == 6
    assert loaded_model.context.n_inputs == 10
    assert isinstance(loaded_model.activation, TopK)
    assert norm == dataset_normalize
    assert target_norm == dataset_target_norm
    assert torch.allclose(mean, dataset_mean)
    assert loaded_model.metadata == run_metadata


def test_load_legacy_state_dict(tmp_path):
    """Test loading legacy state_dict format."""
    # Create a legacy-style payload (just state_dict)
    ae = Autoencoder(n_latents=4, n_inputs=8, activation="TopK_2")
    legacy_payload = ae.state_dict()
    
    # Save legacy format
    legacy_file = tmp_path / "legacy_model.pt"
    torch.save(legacy_payload, legacy_file)
    
    # Create new model and load legacy format
    new_ae = Autoencoder(n_latents=4, n_inputs=8, activation="TopK_2")
    new_ae.load("legacy_model", path=tmp_path)
    
    # Verify model loaded correctly
    x = torch.randn(2, 8)
    with torch.no_grad():
        output1 = ae(x)
        output2 = new_ae(x)
        assert torch.allclose(output1[0], output2[0])


def test_load_with_dict_payload(tmp_path):
    """Test loading with dict payload format."""
    # Create and save model with dict payload
    ae = Autoencoder(n_latents=5, n_inputs=10, activation="TopK_2")
    
    save_path = tmp_path / "dict_model"
    ae.save("dict_autoencoder", path=save_path)
    
    # Load using instance method
    new_ae = Autoencoder(n_latents=5, n_inputs=10, activation="TopK_2")
    new_ae.load("dict_autoencoder", path=save_path)
    
    # Verify model loaded correctly
    x = torch.randn(3, 10)
    with torch.no_grad():
        output1 = ae(x)
        output2 = new_ae(x)
        assert torch.allclose(output1[0], output2[0])


def test_save_load_with_tied_weights(tmp_path):
    """Test saving and loading with tied weights."""
    ae = Autoencoder(n_latents=6, n_inputs=12, activation="TopK_3", tied=True)
    
    save_path = tmp_path / "tied_model"
    ae.save("tied_autoencoder", path=save_path)
    
    # Load and verify tied weights
    loaded_model, _, _, _ = Autoencoder.load_model(save_path / "tied_autoencoder.pt")
    assert loaded_model.context.tied == True
    
    # Verify tied weights are actually tied
    # When tied=True, decoder should be None and encoder should be used for both
    assert loaded_model.decoder is None
    assert loaded_model.encoder is not None


def test_save_load_error_handling(tmp_path):
    """Test error handling in save/load operations."""
    ae = Autoencoder(n_latents=4, n_inputs=8, activation="TopK_2")
    
    # Test save to non-existent directory (should create it)
    non_existent_path = tmp_path / "nonexistent" / "deep" / "path"
    ae.save("error_test", path=non_existent_path)
    
    # Verify directory was created and file exists
    model_file = non_existent_path / "error_test.pt"
    assert model_file.exists()
    
    # Test loading from non-existent file
    with pytest.raises(FileNotFoundError):
        ae.load("nonexistent_model", path=tmp_path)


def test_save_load_with_different_devices(tmp_path):
    """Test saving and loading with different devices."""
    ae = Autoencoder(n_latents=4, n_inputs=8, activation="TopK_2")
    
    # Save on CPU
    save_path = tmp_path / "device_model"
    ae.save("device_autoencoder", path=save_path)
    
    # Load with specific device mapping
    model_file = save_path / "device_autoencoder.pt"
    loaded_model, _, _, _ = Autoencoder.load_model(model_file)
    
    # Verify model is on correct device
    assert loaded_model.context.device == "cpu"


def test_activation_fallback_serialization(tmp_path):
    """Test fallback activation serialization for unknown types."""
    # Create a custom activation that doesn't match known patterns
    class CustomActivation(torch.nn.Module):
        def forward(self, x):
            return torch.tanh(x)
    
    ae = Autoencoder(n_latents=4, n_inputs=8, activation=CustomActivation())
    
    save_path = tmp_path / "custom_activation_model"
    ae.save("custom_autoencoder", path=save_path)
    
    # Load and verify fallback serialization
    loaded_payload = torch.load(save_path / "custom_autoencoder.pt", map_location="cpu")
    assert loaded_payload["activation"] == "CustomActivation"


def test_save_load_with_detach_flag(tmp_path):
    """Test saving with detach flag."""
    ae = Autoencoder(n_latents=4, n_inputs=8, activation="TopK_2")
    
    # Test forward pass with detach=True
    x = torch.randn(2, 8)
    with torch.no_grad():
        recon, latents, recon_full, latents_full = ae(x, detach=True)
    
    # Verify detach worked
    assert not recon.requires_grad
    assert not latents.requires_grad
    assert not recon_full.requires_grad
    assert not latents_full.requires_grad
    
    # Save model
    save_path = tmp_path / "detach_model"
    ae.save("detach_autoencoder", path=save_path)
    
    # Load and verify
    loaded_model, _, _, _ = Autoencoder.load_model(save_path / "detach_autoencoder.pt")
    assert loaded_model.context.n_latents == 4
    assert loaded_model.context.n_inputs == 8
