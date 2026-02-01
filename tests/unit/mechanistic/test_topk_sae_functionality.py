"""Test TopKSAE functionality: detector metadata saving and output modification."""

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

overcomplete_module = types.ModuleType("overcomplete")
overcomplete_sae_module = types.ModuleType("overcomplete.sae")
overcomplete_sae_module.SAE = MagicMock
overcomplete_module.TopKSAE = MagicMock
overcomplete_module.SAE = MagicMock
overcomplete_module.sae = overcomplete_sae_module
sys.modules["overcomplete"] = overcomplete_module
sys.modules["overcomplete.sae"] = overcomplete_sae_module


from mi_crow.hooks.hook import HookType
from mi_crow.mechanistic.sae.modules.topk_sae import TopKSae
from tests.unit.fixtures.stores import create_temp_store


class StubEngine:
    """Stub engine that simulates TopKSAE behavior."""

    def __init__(self):
        self.loaded = None

    def encode(self, x):
        pre_codes = x + 1
        codes = x * 0.5
        return pre_codes, codes

    def decode(self, z):
        return z + 2

    def forward(self, x):
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
        output = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        )
        sae.modify_activations(MagicMock(), (), output)
        assert "batch_items" in sae.metadata
        batch_items = sae.metadata["batch_items"]
        assert len(batch_items) == 2
        assert "neurons" in sae.tensor_metadata
        assert "activations" in sae.tensor_metadata
        neurons_tensor = sae.tensor_metadata["neurons"]
        activations_tensor = sae.tensor_metadata["activations"]
        assert neurons_tensor.shape == (2, 1, 4)
        assert activations_tensor.shape == (2, 1, 4)
        assert torch.equal(neurons_tensor, activations_tensor)
        item_0 = batch_items[0]
        assert "nonzero_indices" in item_0
        assert "activations" in item_0
        assert len(item_0["activations"]) > 0

    def test_metadata_saved_when_hook_runs(self):
        """Test that metadata is saved when hook runs during forward pass."""
        sae = make_topk()
        sae.layer_signature = "test_layer"
        sae.hook_type = HookType.FORWARD
        module = MagicMock()
        output = torch.randn(2, 4)
        if sae._is_both_controller_and_detector():
            sae.process_activations(module, (), output)

        sae.modify_activations(module, (), output)
        assert "batch_items" in sae.metadata
        assert "neurons" in sae.tensor_metadata
        assert "activations" in sae.tensor_metadata
        neurons_tensor = sae.tensor_metadata["neurons"]
        assert len(neurons_tensor.shape) == 3

    def test_metadata_structure_for_store(self, temp_store):
        """Test that metadata structure is correct for saving to store."""
        sae = make_topk()
        sae.layer_signature = "test_layer"
        sae.hook_type = HookType.FORWARD
        output = torch.randn(2, 4)
        sae.modify_activations(MagicMock(), (), output)
        assert "batch_items" in sae.metadata
        assert "neurons" in sae.tensor_metadata
        assert "activations" in sae.tensor_metadata
        neurons_tensor = sae.tensor_metadata["neurons"]
        assert len(neurons_tensor.shape) == 3
        assert neurons_tensor.shape[0] == 2
        metadata_dict = {sae.layer_signature: sae.metadata}
        tensor_metadata_dict = {sae.layer_signature: sae.tensor_metadata}
        path = temp_store.put_detector_metadata("test_run", 0, metadata_dict, tensor_metadata_dict)
        assert path is not None
        retrieved_metadata, retrieved_tensors = temp_store.get_detector_metadata("test_run", 0)
        assert "test_layer" in retrieved_metadata
        assert "test_layer" in retrieved_tensors
        assert "neurons" in retrieved_tensors["test_layer"]
        assert "activations" in retrieved_tensors["test_layer"]
        assert len(retrieved_tensors["test_layer"]["neurons"].shape) == 3


class TestTopKSaeOutputModification:
    """Test that TopKSAE modifies outputs after layer attachment."""

    def test_modify_activations_returns_reconstructed_tensor(self):
        """Test that modify_activations returns reconstructed tensor."""
        sae = make_topk()
        sae.hook_type = HookType.FORWARD
        output = torch.ones(2, 4)
        result = sae.modify_activations(MagicMock(), (), output)
        assert isinstance(result, torch.Tensor)
        assert result.shape == output.shape
        expected = torch.full((2, 4), 2.5)
        assert torch.allclose(result, expected)

    def test_modify_activations_modifies_object_in_place(self):
        """Test that modify_activations modifies objects with last_hidden_state in place."""
        sae = make_topk()
        sae.hook_type = HookType.FORWARD

        class OutputObject:
            def __init__(self):
                self.last_hidden_state = torch.ones(2, 4)

        output = OutputObject()
        original_tensor = output.last_hidden_state.clone()
        result = sae.modify_activations(MagicMock(), (), output)
        assert result is output
        assert not torch.equal(output.last_hidden_state, original_tensor)
        expected = torch.full((2, 4), 2.5)
        assert torch.allclose(output.last_hidden_state, expected)

    def test_modify_activations_modifies_tuple_output(self):
        """Test that modify_activations modifies tuple outputs."""
        sae = make_topk()
        sae.hook_type = HookType.FORWARD
        output = (torch.ones(2, 4), torch.zeros(2, 4))
        result = sae.modify_activations(MagicMock(), (), output)
        assert isinstance(result, tuple)
        assert len(result) == 2
        expected = torch.full((2, 4), 2.5)
        assert torch.allclose(result[0], expected)
        assert torch.equal(result[1], output[1])

    def test_output_modified_during_forward_pass(self):
        """Test that outputs are actually modified when hook runs."""
        sae = make_topk()
        sae.hook_type = HookType.FORWARD
        sae.layer_signature = "test_layer"

        class OutputObject:
            def __init__(self):
                self.last_hidden_state = torch.ones(2, 4)

        output = OutputObject()
        original_tensor = output.last_hidden_state.clone()
        sae.modify_activations(MagicMock(), (), output)
        assert not torch.equal(output.last_hidden_state, original_tensor)

    def test_pre_forward_modifies_inputs(self):
        """Test that pre_forward hooks modify inputs."""
        sae = make_topk()
        sae.hook_type = HookType.PRE_FORWARD
        inputs = (torch.ones(2, 4), torch.zeros(2, 4))
        result = sae.modify_activations(MagicMock(), inputs, None)
        assert isinstance(result, tuple)
        assert len(result) == 2
        expected = torch.full((2, 4), 2.5)
        assert torch.allclose(result[0], expected)
        assert torch.equal(result[1], inputs[1])


class TestTopKSaeIntegration:
    """Integration tests for TopKSAE detector and controller functionality."""

    def test_both_detector_and_controller_functionality(self):
        """Test that TopKSAE works as both detector and controller."""
        sae = make_topk()
        sae.layer_signature = "test_layer"
        sae.hook_type = HookType.FORWARD
        output = torch.ones(2, 4)
        original_output = output.clone()
        if sae._is_both_controller_and_detector():
            sae.process_activations(MagicMock(), (), output)

        modified_output = sae.modify_activations(MagicMock(), (), output)
        assert "batch_items" in sae.metadata
        assert "neurons" in sae.tensor_metadata
        assert "activations" in sae.tensor_metadata
        neurons_tensor = sae.tensor_metadata["neurons"]
        assert len(neurons_tensor.shape) == 3
        assert isinstance(modified_output, torch.Tensor)
        assert not torch.equal(modified_output, original_output)
        expected = torch.full((2, 4), 2.5)
        assert torch.allclose(modified_output, expected)

    def test_modify_activations_3d_input_saves_3d_tensor(self):
        """Test that TopKSae saves 3D tensor correctly for 3D input."""
        sae = make_topk()
        sae.hook_type = HookType.FORWARD
        module = MagicMock()
        output = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
            ]
        )
        sae.modify_activations(module, (), output)
        assert "neurons" in sae.tensor_metadata
        assert "activations" in sae.tensor_metadata
        neurons_tensor = sae.tensor_metadata["neurons"]
        activations_tensor = sae.tensor_metadata["activations"]
        assert neurons_tensor.shape == (2, 2, 4)
        assert activations_tensor.shape == (2, 2, 4)
        assert torch.equal(neurons_tensor, activations_tensor)
        batch_items = sae.metadata["batch_items"]
        assert len(batch_items) == 4
