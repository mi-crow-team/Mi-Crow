"""Test TopKSAE functionality: detector metadata saving and output modification."""

import sys
import types
import pytest
import torch
from unittest.mock import MagicMock

# Mock overcomplete module before importing TopKSae
overcomplete_module = types.ModuleType('overcomplete')
overcomplete_sae_module = types.ModuleType('overcomplete.sae')
# Mock classes
overcomplete_sae_module.SAE = MagicMock
# TopKSAE and SAE are also available at top level
overcomplete_module.TopKSAE = MagicMock
overcomplete_module.SAE = MagicMock
overcomplete_module.sae = overcomplete_sae_module
sys.modules['overcomplete'] = overcomplete_module
sys.modules['overcomplete.sae'] = overcomplete_sae_module

from mi_crow.hooks.hook import HookType
from mi_crow.mechanistic.sae.modules.topk_sae import TopKSae
from tests.unit.fixtures.stores import create_temp_store


class StubEngine:
    """Stub engine that simulates TopKSAE behavior."""
    def __init__(self):
        self.loaded = None

    def encode(self, x):
        # Return (pre_codes, codes) where codes are sparse TopK
        # For simplicity, pre_codes = x + 1, codes = x * 0.5 (sparse)
        pre_codes = x + 1
        # Simulate TopK: only keep top values, zero out others
        codes = x * 0.5
        return pre_codes, codes

    def decode(self, z):
        return z + 2

    def forward(self, x):
        # Return (pre_codes, codes, x_reconstructed)
        pre_codes, codes = self.encode(x)
        x_reconstructed = self.decode(codes)
        return pre_codes, codes, x_reconstructed

    def state_dict(self):
        return {"weight": torch.ones(1)}

    def load_state_dict(self, state):
        self.loaded = state

    def parameters(self):
        return [torch.nn.Parameter(torch.ones(1, requires_grad=True))]

    def to(self, *args, **kwargs):
        return self


@pytest.fixture(autouse=True)
def stub_engine(monkeypatch):
    """Replace TopKSAE engine with stub for testing."""
    monkeypatch.setattr(
        TopKSae,
        "_initialize_sae_engine",
        lambda self: StubEngine(),
    )


def make_topk():
    """Create a TopKSAE instance for testing."""
    return TopKSae(n_latents=4, n_inputs=4, k=2)


class TestTopKSaeDetectorMetadata:
    """Test that TopKSAE saves detector metadata when attached."""

    def test_modify_activations_saves_metadata(self):
        """Test that modify_activations saves metadata to detector."""
        sae = make_topk()
        sae.layer_signature = "test_layer"
        sae.hook_type = HookType.FORWARD
        
        # Create test output
        output = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ])
        
        # Modify activations (this now saves metadata)
        sae.modify_activations(MagicMock(), (), output)
        
        # Verify metadata is saved
        assert 'batch_items' in sae.metadata
        batch_items = sae.metadata['batch_items']
        assert len(batch_items) == 2
        
        # Verify tensor metadata is saved as 3D tensor
        assert 'neurons' in sae.tensor_metadata
        assert 'activations' in sae.tensor_metadata
        neurons_tensor = sae.tensor_metadata['neurons']
        activations_tensor = sae.tensor_metadata['activations']
        # Should be 3D: (batch, seq=1, features) for 2D input
        assert neurons_tensor.shape == (2, 1, 4)
        assert activations_tensor.shape == (2, 1, 4)
        assert torch.equal(neurons_tensor, activations_tensor)
        
        # Verify batch items contain activation data
        # StubEngine.encode returns pre_codes = x + 1
        # So for first item: [2.0, 3.0, 4.0, 5.0]
        item_0 = batch_items[0]
        assert 'nonzero_indices' in item_0
        assert 'activations' in item_0
        assert len(item_0['activations']) > 0

    def test_metadata_saved_when_hook_runs(self):
        """Test that metadata is saved when hook runs during forward pass."""
        sae = make_topk()
        sae.layer_signature = "test_layer"
        sae.hook_type = HookType.FORWARD
        
        # Create a mock module
        module = MagicMock()
        
        # Create test output
        output = torch.randn(2, 4)
        
        # Simulate hook execution (as both Controller and Detector)
        # process_activations is now empty, modify_activations does the saving
        if sae._is_both_controller_and_detector():
            sae.process_activations(module, (), output)  # Empty, just for interface
        
        # Modify activations (this saves metadata)
        sae.modify_activations(module, (), output)
        
        # Verify metadata was saved
        assert 'batch_items' in sae.metadata
        assert 'neurons' in sae.tensor_metadata
        assert 'activations' in sae.tensor_metadata
        # Verify 3D tensor shape
        neurons_tensor = sae.tensor_metadata['neurons']
        assert len(neurons_tensor.shape) == 3  # (batch, seq, features)

    def test_metadata_structure_for_store(self, temp_store):
        """Test that metadata structure is correct for saving to store."""
        sae = make_topk()
        sae.layer_signature = "test_layer"  # Use string for store compatibility
        sae.hook_type = HookType.FORWARD
        
        # Modify activations to populate metadata (saving happens here now)
        output = torch.randn(2, 4)
        sae.modify_activations(MagicMock(), (), output)
        
        # Verify metadata structure matches what store expects
        assert 'batch_items' in sae.metadata
        assert 'neurons' in sae.tensor_metadata
        assert 'activations' in sae.tensor_metadata
        
        # Verify tensor is 3D
        neurons_tensor = sae.tensor_metadata['neurons']
        assert len(neurons_tensor.shape) == 3
        assert neurons_tensor.shape[0] == 2  # batch size
        
        # Verify metadata can be saved to store format
        # The store expects: metadata dict and tensor_metadata dict keyed by layer_signature
        metadata_dict = {sae.layer_signature: sae.metadata}
        tensor_metadata_dict = {sae.layer_signature: sae.tensor_metadata}
        
        # Save to store
        path = temp_store.put_detector_metadata("test_run", 0, metadata_dict, tensor_metadata_dict)
        assert path is not None
        
        # Retrieve and verify
        retrieved_metadata, retrieved_tensors = temp_store.get_detector_metadata("test_run", 0)
        
        # Check that metadata exists for our layer
        assert "test_layer" in retrieved_metadata
        assert "test_layer" in retrieved_tensors
        
        # Check that neurons and activations tensors were saved
        assert "neurons" in retrieved_tensors["test_layer"]
        assert "activations" in retrieved_tensors["test_layer"]
        # Verify 3D shape
        assert len(retrieved_tensors["test_layer"]["neurons"].shape) == 3


class TestTopKSaeOutputModification:
    """Test that TopKSAE modifies outputs after layer attachment."""

    def test_modify_activations_returns_reconstructed_tensor(self):
        """Test that modify_activations returns reconstructed tensor."""
        sae = make_topk()
        sae.hook_type = HookType.FORWARD
        
        # Create test output
        output = torch.ones(2, 4)
        
        # Modify activations
        result = sae.modify_activations(MagicMock(), (), output)
        
        # Should return reconstructed tensor
        assert isinstance(result, torch.Tensor)
        assert result.shape == output.shape
        # StubEngine: encode returns codes = x * 0.5, decode adds 2
        # So result should be (output * 0.5) + 2 = 0.5 + 2 = 2.5
        expected = torch.full((2, 4), 2.5)
        assert torch.allclose(result, expected)

    def test_modify_activations_modifies_object_in_place(self):
        """Test that modify_activations modifies objects with last_hidden_state in place."""
        sae = make_topk()
        sae.hook_type = HookType.FORWARD
        
        # Create object with last_hidden_state
        class OutputObject:
            def __init__(self):
                self.last_hidden_state = torch.ones(2, 4)
        
        output = OutputObject()
        original_tensor = output.last_hidden_state.clone()
        
        # Modify activations
        result = sae.modify_activations(MagicMock(), (), output)
        
        # Should modify in place
        assert result is output
        assert not torch.equal(output.last_hidden_state, original_tensor)
        # Should be reconstructed
        expected = torch.full((2, 4), 2.5)
        assert torch.allclose(output.last_hidden_state, expected)

    def test_modify_activations_modifies_tuple_output(self):
        """Test that modify_activations modifies tuple outputs."""
        sae = make_topk()
        sae.hook_type = HookType.FORWARD
        
        # Create tuple output
        output = (torch.ones(2, 4), torch.zeros(2, 4))
        
        # Modify activations
        result = sae.modify_activations(MagicMock(), (), output)
        
        # Should return tuple with first tensor modified
        assert isinstance(result, tuple)
        assert len(result) == 2
        # First tensor should be reconstructed
        expected = torch.full((2, 4), 2.5)
        assert torch.allclose(result[0], expected)
        # Second tensor should be unchanged
        assert torch.equal(result[1], output[1])

    def test_output_modified_during_forward_pass(self):
        """Test that outputs are actually modified when hook runs."""
        sae = make_topk()
        sae.hook_type = HookType.FORWARD
        sae.layer_signature = "test_layer"
        
        # Create object with last_hidden_state
        class OutputObject:
            def __init__(self):
                self.last_hidden_state = torch.ones(2, 4)
        
        output = OutputObject()
        original_tensor = output.last_hidden_state.clone()
        
        # Simulate hook execution
        # For forward hooks, modify_activations is called but output is modified in-place
        # when it's an object with attributes
        sae.modify_activations(MagicMock(), (), output)
        
        # Verify output was modified
        assert not torch.equal(output.last_hidden_state, original_tensor)

    def test_pre_forward_modifies_inputs(self):
        """Test that pre_forward hooks modify inputs."""
        sae = make_topk()
        sae.hook_type = HookType.PRE_FORWARD
        
        # Create input tuple
        inputs = (torch.ones(2, 4), torch.zeros(2, 4))
        
        # Modify activations
        result = sae.modify_activations(MagicMock(), inputs, None)
        
        # Should return tuple with first tensor modified
        assert isinstance(result, tuple)
        assert len(result) == 2
        # First tensor should be reconstructed
        expected = torch.full((2, 4), 2.5)
        assert torch.allclose(result[0], expected)
        # Second tensor should be unchanged
        assert torch.equal(result[1], inputs[1])


class TestTopKSaeIntegration:
    """Integration tests for TopKSAE detector and controller functionality."""

    def test_both_detector_and_controller_functionality(self):
        """Test that TopKSAE works as both detector and controller."""
        sae = make_topk()
        sae.layer_signature = "test_layer"
        sae.hook_type = HookType.FORWARD
        
        # Create test output
        output = torch.ones(2, 4)
        original_output = output.clone()
        
        # Simulate full hook execution
        # process_activations is now empty, modify_activations does both saving and modification
        if sae._is_both_controller_and_detector():
            sae.process_activations(MagicMock(), (), output)  # Empty, just for interface
        
        # Modify activations (this saves metadata and modifies output)
        modified_output = sae.modify_activations(MagicMock(), (), output)
        
        # Verify detector saved metadata
        assert 'batch_items' in sae.metadata
        assert 'neurons' in sae.tensor_metadata
        assert 'activations' in sae.tensor_metadata
        # Verify 3D tensor
        neurons_tensor = sae.tensor_metadata['neurons']
        assert len(neurons_tensor.shape) == 3  # (batch, seq, features)
        
        # Verify controller modified output
        assert isinstance(modified_output, torch.Tensor)
        assert not torch.equal(modified_output, original_output)
        expected = torch.full((2, 4), 2.5)
        assert torch.allclose(modified_output, expected)

    def test_modify_activations_3d_input_saves_3d_tensor(self):
        """Test that TopKSae saves 3D tensor correctly for 3D input."""
        sae = make_topk()
        sae.hook_type = HookType.FORWARD
        module = MagicMock()
        
        # Create test input (3D: batch, seq, features)
        output = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
        ])
        
        sae.modify_activations(module, (), output)
        
        # Verify 3D tensor is saved
        assert 'neurons' in sae.tensor_metadata
        assert 'activations' in sae.tensor_metadata
        neurons_tensor = sae.tensor_metadata['neurons']
        activations_tensor = sae.tensor_metadata['activations']
        assert neurons_tensor.shape == (2, 2, 4)  # (batch, seq, features)
        assert activations_tensor.shape == (2, 2, 4)
        assert torch.equal(neurons_tensor, activations_tensor)
        
        # Verify batch_items (should have 4 items: 2 batch * 2 seq)
        batch_items = sae.metadata['batch_items']
        assert len(batch_items) == 4  # batch * seq

