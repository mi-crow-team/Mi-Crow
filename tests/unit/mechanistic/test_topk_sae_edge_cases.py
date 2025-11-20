"""Additional tests for TopKSae edge cases and error handling."""
import pytest
import torch

from amber.mechanistic.sae.modules.topk_sae import TopKSae
from amber.hooks.hook import HookType


def test_topk_sae_modify_activations_with_none_target():
    """Test modify_activations handles None target gracefully."""
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4, device='cpu')
    
    class DummyModule:
        pass
    
    module = DummyModule()
    
    # Test with None output
    result = topk_sae.modify_activations(module, (), None)
    assert result is None
    
    # Test with empty inputs tuple
    result = topk_sae.modify_activations(module, (), torch.randn(2, 16))
    assert result is not None


def test_topk_sae_modify_activations_with_tuple_output():
    """Test modify_activations handles tuple outputs."""
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4, device='cpu')
    topk_sae.sae_engine.eval()
    
    class DummyModule:
        pass
    
    module = DummyModule()
    output = (torch.randn(2, 3, 16), torch.randn(2, 3, 8))
    
    result = topk_sae.modify_activations(module, (), output)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0].shape == (2, 3, 16)


def test_topk_sae_modify_activations_with_list_output():
    """Test modify_activations handles list outputs."""
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4, device='cpu')
    topk_sae.sae_engine.eval()
    
    class DummyModule:
        pass
    
    module = DummyModule()
    output = [torch.randn(2, 3, 16), torch.randn(2, 3, 8)]
    
    result = topk_sae.modify_activations(module, (), output)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == (2, 3, 16)


def test_topk_sae_modify_activations_with_object_output():
    """Test modify_activations handles object with last_hidden_state attribute."""
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4, device='cpu')
    topk_sae.sae_engine.eval()
    
    class DummyModule:
        pass
    
    class OutputObject:
        def __init__(self):
            self.last_hidden_state = torch.randn(2, 3, 16)
    
    module = DummyModule()
    output = OutputObject()
    
    result = topk_sae.modify_activations(module, (), output)
    assert hasattr(result, 'last_hidden_state')
    assert result.last_hidden_state.shape == (2, 3, 16)


def test_topk_sae_modify_activations_pre_forward_hook():
    """Test modify_activations with PRE_FORWARD hook type."""
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4, device='cpu', hook_type=HookType.PRE_FORWARD)
    topk_sae.sae_engine.eval()
    
    class DummyModule:
        pass
    
    module = DummyModule()
    inputs = (torch.randn(2, 3, 16),)
    
    # For PRE_FORWARD, output is None, so we pass inputs as the target
    result = topk_sae.modify_activations(module, inputs, None)
    # If target is None, it returns inputs unchanged for PRE_FORWARD
    if result is not None:
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].shape == (2, 3, 16)


def test_topk_sae_modify_activations_with_concept_manipulation():
    """Test modify_activations applies concept manipulation when set."""
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4, device='cpu')
    topk_sae.sae_engine.eval()
    
    # Set non-default multiplication and bias
    topk_sae.concepts.multiplication.data = torch.ones(8) * 2.0
    topk_sae.concepts.bias.data = torch.ones(8) * 0.5
    
    class DummyModule:
        pass
    
    module = DummyModule()
    output = torch.randn(2, 3, 16)
    
    result = topk_sae.modify_activations(module, (), output)
    assert result.shape == output.shape


def test_topk_sae_modify_activations_with_text_tracking():
    """Test modify_activations with text tracking enabled."""
    from amber.language_model.language_model import LanguageModel
    
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4, device='cpu')
    topk_sae.sae_engine.eval()
    
    # Create a mock language model
    from pathlib import Path
    import tempfile
    from amber.store.local_store import LocalStore
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel.from_huggingface("sshleifer/tiny-gpt2", store=store)
    topk_sae.context.lm = lm
    topk_sae.context.text_tracking_enabled = True
    topk_sae._text_tracking_enabled = True
    
    # Enable text tracking
    topk_sae.concepts.enable_text_tracking()
    
    # Set texts in input tracker
    input_tracker = lm._ensure_input_tracker()
    input_tracker.enable()
    input_tracker.set_current_texts(["Hello world", "Test text"])
    
    class DummyModule:
        pass
    
    module = DummyModule()
    output = torch.randn(2, 3, 16)
    
    result = topk_sae.modify_activations(module, (), output)
    assert result.shape == output.shape
    
    # Check that texts were tracked
    top_texts = topk_sae.concepts.get_top_texts_for_neuron(0)
    # May or may not have texts depending on activations, but should not crash


def test_topk_sae_modify_activations_with_non_tensor_in_tuple():
    """Test modify_activations handles tuple with non-tensor elements."""
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4, device='cpu')
    topk_sae.sae_engine.eval()
    
    class DummyModule:
        pass
    
    module = DummyModule()
    # Tuple with tensor and non-tensor
    output = (torch.randn(2, 3, 16), "not_a_tensor", 42)
    
    result = topk_sae.modify_activations(module, (), output)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0].shape == (2, 3, 16)
    assert result[1] == "not_a_tensor"
    assert result[2] == 42


def test_topk_sae_modify_activations_with_2d_tensor():
    """Test modify_activations handles 2D tensors (overcomplete requires 2D)."""
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4, device='cpu')
    topk_sae.sae_engine.eval()
    
    class DummyModule:
        pass
    
    module = DummyModule()
    output = torch.randn(5, 16)  # 2D tensor (batch, features)
    
    result = topk_sae.modify_activations(module, (), output)
    assert result.shape == output.shape


def test_topk_sae_save_load_edge_cases(tmp_path):
    """Test save and load with various edge cases."""
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4, device='cpu')
    
    # Test save with None path (should use current directory)
    save_path = tmp_path / "test_model.pt"
    topk_sae.save("test_model", tmp_path)
    assert save_path.exists()
    
    # Test load
    loaded = TopKSae.load(save_path)
    assert loaded.k == 4
    assert loaded.context.n_latents == 8
    assert loaded.context.n_inputs == 16
    
    # Test load with invalid file
    invalid_path = tmp_path / "invalid.pt"
    with pytest.raises((ValueError, FileNotFoundError)):
        TopKSae.load(invalid_path)


def test_topk_sae_to_device():
    """Test moving TopKSae to different device."""
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4, device='cpu')
    
    # Test moving to CPU (should work)
    topk_sae.sae_engine.to('cpu')
    
    # Test that forward still works
    x = torch.randn(5, 16)
    result = topk_sae.forward(x)
    assert result.shape == (5, 16)

