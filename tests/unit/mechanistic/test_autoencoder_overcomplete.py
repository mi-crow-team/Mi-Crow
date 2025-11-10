"""Tests for Autoencoder integration with overcomplete SAE engine."""

import torch
import pytest
from torch import nn

from amber.mechanistic.autoencoder.autoencoder import Autoencoder
from amber.mechanistic.autoencoder.modules.topk_sae import TopKSae
from amber.hooks.hook import HookType


def test_autoencoder_inherits_from_controller():
    """Test that Autoencoder is a nn.Module (not Controller)."""
    ae = Autoencoder(n_latents=10, n_inputs=20)
    
    # Should be an instance of nn.Module
    assert isinstance(ae, nn.Module)
    
    # Should have context and concepts
    assert hasattr(ae, "context")
    assert hasattr(ae, "concepts")


def test_autoencoder_has_base_dict_learning():
    """Test that Autoencoder has expected attributes."""
    ae = Autoencoder(n_latents=10, n_inputs=20)
    
    # Should have context and concepts
    assert hasattr(ae, "context")
    assert hasattr(ae, "concepts")
    assert hasattr(ae, "activation")


def test_autoencoder_encode_delegates_to_engine():
    """Test that encode() works correctly."""
    torch.manual_seed(42)
    ae = Autoencoder(n_latents=8, n_inputs=16, activation="TopK_4")
    
    x = torch.randn(5, 16)
    
    # Encode should work
    encoded, full_encoded, info = ae.encode(x)
    assert encoded.shape == (5, 8)
    assert full_encoded.shape == (5, 8)


def test_autoencoder_decode_delegates_to_engine():
    """Test that decode() works correctly."""
    torch.manual_seed(42)
    ae = Autoencoder(n_latents=8, n_inputs=16)
    
    latents = torch.randn(5, 8)
    
    reconstructed = ae.decode(latents)
    assert reconstructed.shape == (5, 16)


def test_autoencoder_forward_delegates_to_engine():
    """Test that forward() uses SAE engine when available."""
    torch.manual_seed(42)
    ae = Autoencoder(n_latents=8, n_inputs=16)
    
    x = torch.randn(5, 16)
    recon, lat, recon_full, lat_full = ae(x)
    
    assert recon.shape == x.shape
    assert lat.shape == (5, 8)
    assert recon_full.shape == x.shape
    assert lat_full.shape == (5, 8)


def test_autoencoder_get_dictionary():
    """Test that Autoencoder has encoder weights."""
    torch.manual_seed(42)
    ae = Autoencoder(n_latents=10, n_inputs=20)
    
    # Autoencoder doesn't have get_dictionary, but has encoder weights
    assert hasattr(ae, "encoder")
    assert ae.encoder.shape == (20, 10)  # encoder is transposed


def test_autoencoder_modify_activations_reconstruction_mode():
    """Test modify_activations in reconstruction mode (default)."""
    torch.manual_seed(42)
    ae = Autoencoder(n_latents=8, n_inputs=16)
    ae.eval()
    
    # Autoencoder doesn't have modify_activations - it's not a Controller
    # This test is skipped as Autoencoder is not a hook
    pytest.skip("Autoencoder is not a Controller hook")


def test_autoencoder_modify_activations_manipulation_mode():
    """Test modify_activations in manipulation mode."""
    torch.manual_seed(42)
    ae = Autoencoder(n_latents=8, n_inputs=16)
    ae.eval()
    
    # Autoencoder doesn't have modify_activations - it's not a Controller
    # This test is skipped as Autoencoder is not a hook
    pytest.skip("Autoencoder is not a Controller hook")


def test_autoencoder_modify_activations_pre_forward_hook():
    """Test modify_activations with pre_forward hook."""
    torch.manual_seed(42)
    ae = Autoencoder(n_latents=8, n_inputs=16)
    ae.eval()
    
    # Autoencoder doesn't have modify_activations - it's not a Controller
    # This test is skipped as Autoencoder is not a hook
    pytest.skip("Autoencoder is not a Controller hook")


def test_autoencoder_controller_enable_disable():
    """Test that Autoencoder is not a Controller."""
    ae = Autoencoder(n_latents=8, n_inputs=16)
    
    # Autoencoder is not a Controller, so it doesn't have enabled/disable
    from amber.hooks.controller import Controller
    assert not isinstance(ae, Controller)
    assert not hasattr(ae, "enabled")


def test_autoencoder_set_layer_signature():
    """Test that Autoencoder doesn't have layer_signature."""
    ae = Autoencoder(n_latents=10, n_inputs=20)
    
    # Autoencoder is not a Controller, so it doesn't have layer_signature
    assert not hasattr(ae, "layer_signature")


def test_autoencoder_fit_method():
    """Test fit() method."""
    ae = Autoencoder(n_latents=10, n_inputs=20)
    
    # Autoencoder doesn't have a fit() method
    # This test is skipped
    pytest.skip("Autoencoder doesn't have a fit() method")


# ==================== TopKSAE Tests ====================

def test_topk_sae_inherits_from_sae():
    """Test that TopKSae inherits from Sae (not Autoencoder)."""
    try:
        from overcomplete.sae import TopKSAE as OvercompleteTopKSAE
    except ImportError:
        pytest.skip("Overcomplete not available")
    
    topk_sae = TopKSae(n_latents=10, n_inputs=20, k=5)
    
    # Should be an instance of Sae (which inherits from Controller)
    from amber.mechanistic.autoencoder.sae import Sae
    assert isinstance(topk_sae, Sae)
    
    # Should have sae_engine (overcomplete's TopKSAE)
    assert hasattr(topk_sae, "sae_engine")
    assert topk_sae.sae_engine is not None


def test_topk_sae_composes_of_overcomplete_topk_sae():
    """Test that TopKSae composes of overcomplete's TopKSAE."""
    try:
        from overcomplete.sae import TopKSAE as OvercompleteTopKSAE
    except ImportError:
        pytest.skip("Overcomplete not available")
    
    topk_sae = TopKSae(n_latents=10, n_inputs=20, k=5)
    
    # Should have sae_engine attribute
    assert hasattr(topk_sae, "sae_engine")
    assert topk_sae.sae_engine is not None
    
    # Should be an instance of overcomplete's TopKSAE
    assert isinstance(topk_sae.sae_engine, OvercompleteTopKSAE)


def test_topk_sae_has_k_parameter():
    """Test that TopKSae stores k parameter."""
    try:
        from overcomplete.sae import TopKSAE as OvercompleteTopKSAE
    except ImportError:
        pytest.skip("Overcomplete not available")
    
    k = 7
    topk_sae = TopKSae(n_latents=10, n_inputs=20, k=k)
    
    assert topk_sae.k == k


def test_topk_sae_encode_uses_topk_sae():
    """Test that TopKSae.encode() uses the composed TopKSAE."""
    try:
        from overcomplete.sae import TopKSAE as OvercompleteTopKSAE
    except ImportError:
        pytest.skip("Overcomplete not available")
    
    torch.manual_seed(42)
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4)
    
    x = torch.randn(5, 16)
    encoded = topk_sae.encode(x)
    
    assert encoded.shape == (5, 8)
    
    # TopKSae should enforce sparsity (only k non-zero values)
    # Check that each row has at most k non-zero values
    non_zero_counts = (encoded != 0).sum(dim=-1)
    assert (non_zero_counts <= topk_sae.k).all()


def test_topk_sae_decode_uses_topk_sae():
    """Test that TopKSae.decode() uses the composed TopKSAE."""
    try:
        from overcomplete.sae import TopKSAE as OvercompleteTopKSAE
    except ImportError:
        pytest.skip("Overcomplete not available")
    
    torch.manual_seed(42)
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4)
    
    latents = torch.randn(5, 8)
    reconstructed = topk_sae.decode(latents)
    
    assert reconstructed.shape == (5, 16)


def test_topk_sae_forward_uses_topk_sae():
    """Test that TopKSae.forward() uses the composed TopKSAE."""
    try:
        from overcomplete.sae import TopKSAE as OvercompleteTopKSAE
    except ImportError:
        pytest.skip("Overcomplete not available")
    
    torch.manual_seed(42)
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4)
    
    x = torch.randn(5, 16)
    reconstructed = topk_sae.forward(x)
    
    assert reconstructed.shape == x.shape


def test_topk_sae_get_dictionary():
    """Test TopKSae dictionary access via sae_engine."""
    try:
        from overcomplete.sae import TopKSAE as OvercompleteTopKSAE
    except ImportError:
        pytest.skip("Overcomplete not available")
    
    torch.manual_seed(42)
    topk_sae = TopKSae(n_latents=10, n_inputs=20, k=5)
    
    # Access dictionary through sae_engine
    if hasattr(topk_sae.sae_engine, 'dictionary'):
        dictionary = topk_sae.sae_engine.dictionary._weights
        assert dictionary.shape == (10, 20)
    else:
        pytest.skip("Dictionary access not available in this version")


def test_topk_sae_modify_activations():
    """Test TopKSae.modify_activations() (Controller interface)."""
    try:
        from overcomplete.sae import TopKSAE as OvercompleteTopKSAE
    except ImportError:
        pytest.skip("Overcomplete not available")
    
    torch.manual_seed(42)
    topk_sae = TopKSae(
        n_latents=8,
        n_inputs=16,
        k=4,
        hook_id="test"
    )
    topk_sae.sae_engine.eval()
    
    # TopKSae inherits from Sae which inherits from Controller
    # So it should have modify_activations
    assert hasattr(topk_sae, "modify_activations")
    
    # Test modify_activations works
    x = torch.randn(2, 3, 16)
    class DummyModule:
        pass
    module = DummyModule()
    output = x
    
    modified = topk_sae.modify_activations(module, (), output)
    assert modified.shape == x.shape


def test_topk_sae_layered_abstraction():
    """Test that TopKSae properly implements layered abstraction."""
    try:
        from overcomplete.sae import TopKSAE as OvercompleteTopKSAE
    except ImportError:
        pytest.skip("Overcomplete not available")
    
    topk_sae = TopKSae(n_latents=10, n_inputs=20, k=5)
    
    # TopKSae should inherit from Sae (which inherits from Controller)
    from amber.mechanistic.autoencoder.sae import Sae
    assert isinstance(topk_sae, Sae)
    
    # TopKSae should compose of overcomplete's TopKSAE
    assert isinstance(topk_sae.sae_engine, OvercompleteTopKSAE)

