"""Comprehensive unit tests for SAE utility functions."""

import pytest
import torch
from unittest.mock import Mock, MagicMock

from amber.hooks.hook import HookType
from amber.mechanistic.sae.utils import (
    extract_activation_tensor,
    reshape_for_sae,
    reshape_from_sae,
    reconstruct_hook_output,
    _reconstruct_forward_output,
    _reconstruct_pre_forward_inputs,
)


class TestExtractActivationTensor:
    """Test extract_activation_tensor function."""

    def test_extract_from_forward_tensor_output(self):
        """Test extracting tensor from FORWARD hook with tensor output."""
        output = torch.randn(2, 3, 4)
        inputs = (torch.randn(1, 2),)
        
        tensor, original = extract_activation_tensor(HookType.FORWARD, inputs, output)
        
        assert tensor is not None
        assert torch.equal(tensor, output)
        assert original is output

    def test_extract_from_forward_tuple_output(self):
        """Test extracting tensor from FORWARD hook with tuple output."""
        tensor1 = torch.randn(2, 3)
        tensor2 = torch.randn(2, 5)
        output = (tensor1, tensor2, "other")
        inputs = (torch.randn(1, 2),)
        
        tensor, original = extract_activation_tensor(HookType.FORWARD, inputs, output)
        
        assert tensor is not None
        assert torch.equal(tensor, tensor1)
        assert original is output

    def test_extract_from_forward_list_output(self):
        """Test extracting tensor from FORWARD hook with list output."""
        tensor1 = torch.randn(2, 3)
        tensor2 = torch.randn(2, 5)
        output = [tensor1, tensor2, "other"]
        inputs = (torch.randn(1, 2),)
        
        tensor, original = extract_activation_tensor(HookType.FORWARD, inputs, output)
        
        assert tensor is not None
        assert torch.equal(tensor, tensor1)
        assert original is output

    def test_extract_from_forward_object_output(self):
        """Test extracting tensor from FORWARD hook with object output."""
        tensor = torch.randn(2, 3)
        output = Mock()
        output.last_hidden_state = tensor
        inputs = (torch.randn(1, 2),)
        
        tensor_result, original = extract_activation_tensor(HookType.FORWARD, inputs, output)
        
        assert tensor_result is not None
        assert torch.equal(tensor_result, tensor)
        assert original is output

    def test_extract_from_forward_none_output(self):
        """Test extracting from FORWARD hook with None output."""
        inputs = (torch.randn(1, 2),)
        
        tensor, original = extract_activation_tensor(HookType.FORWARD, inputs, None)
        
        assert tensor is None
        assert original is None

    def test_extract_from_pre_forward_tensor_input(self):
        """Test extracting tensor from PRE_FORWARD hook with tensor input."""
        inputs = (torch.randn(2, 3, 4),)
        output = None
        
        tensor, original = extract_activation_tensor(HookType.PRE_FORWARD, inputs, output)
        
        assert tensor is not None
        assert torch.equal(tensor, inputs[0])
        assert original is inputs

    def test_extract_from_pre_forward_tuple_input(self):
        """Test extracting tensor from PRE_FORWARD hook with tuple in first position."""
        tensor = torch.randn(2, 3)
        inputs = ((tensor, torch.randn(1, 2)), "other")
        output = None
        
        tensor_result, original = extract_activation_tensor(HookType.PRE_FORWARD, inputs, output)
        
        assert tensor_result is not None
        assert torch.equal(tensor_result, tensor)
        assert original is inputs

    def test_extract_from_pre_forward_empty_inputs(self):
        """Test extracting from PRE_FORWARD hook with empty inputs."""
        inputs = ()
        output = None
        
        tensor, original = extract_activation_tensor(HookType.PRE_FORWARD, inputs, output)
        
        assert tensor is None
        assert original is inputs


class TestReshapeForSae:
    """Test reshape_for_sae function."""

    def test_reshape_2d_tensor_no_change(self):
        """Test that 2D tensor is not reshaped."""
        tensor = torch.randn(10, 16)
        
        reshaped, original_shape, needs_reshape = reshape_for_sae(tensor)
        
        assert torch.equal(reshaped, tensor)
        assert original_shape == (10, 16)
        assert needs_reshape is False

    def test_reshape_3d_tensor_flattens(self):
        """Test that 3D tensor is flattened to 2D."""
        tensor = torch.randn(2, 5, 16)
        
        reshaped, original_shape, needs_reshape = reshape_for_sae(tensor)
        
        assert reshaped.shape == (10, 16)
        assert original_shape == (2, 5, 16)
        assert needs_reshape is True
        assert torch.equal(reshaped, tensor.reshape(-1, 16))

    def test_reshape_4d_tensor_flattens(self):
        """Test that 4D tensor is flattened to 2D."""
        tensor = torch.randn(2, 3, 4, 8)
        
        reshaped, original_shape, needs_reshape = reshape_for_sae(tensor)
        
        assert reshaped.shape == (24, 8)
        assert original_shape == (2, 3, 4, 8)
        assert needs_reshape is True

    def test_reshape_1d_tensor_raises(self):
        """Test that 1D tensor raises error (should be caught by validation)."""
        tensor = torch.randn(10)
        
        # This should work but will fail later in validation
        reshaped, original_shape, needs_reshape = reshape_for_sae(tensor)
        
        assert reshaped.shape == (10, 1)  # Reshapes to (10, 1)
        assert needs_reshape is False  # len(shape) == 1, so no reshape


class TestReshapeFromSae:
    """Test reshape_from_sae function."""

    def test_reshape_back_2d_no_change(self):
        """Test reshaping 2D tensor back when no reshape was needed."""
        tensor = torch.randn(10, 16)
        original_shape = (10, 16)
        
        result = reshape_from_sae(tensor, original_shape, needs_reshape=False)
        
        assert torch.equal(result, tensor)
        assert result.shape == (10, 16)

    def test_reshape_back_3d_from_flattened(self):
        """Test reshaping back to 3D from flattened 2D."""
        tensor = torch.randn(10, 16)
        original_shape = (2, 5, 16)
        
        result = reshape_from_sae(tensor, original_shape, needs_reshape=True)
        
        assert result.shape == (2, 5, 16)
        assert torch.equal(result, tensor.reshape(2, 5, 16))

    def test_reshape_back_4d_from_flattened(self):
        """Test reshaping back to 4D from flattened 2D."""
        tensor = torch.randn(24, 8)
        original_shape = (2, 3, 4, 8)
        
        result = reshape_from_sae(tensor, original_shape, needs_reshape=True)
        
        assert result.shape == (2, 3, 4, 8)
        assert torch.equal(result, tensor.reshape(2, 3, 4, 8))


class TestReconstructHookOutput:
    """Test reconstruct_hook_output function."""

    def test_reconstruct_forward_tensor(self):
        """Test reconstructing FORWARD hook with tensor output."""
        reconstructed = torch.randn(2, 3, 4)
        original = torch.randn(2, 3, 4)
        
        result = reconstruct_hook_output(HookType.FORWARD, reconstructed, original)
        
        assert torch.equal(result, reconstructed)
        assert result is not original

    def test_reconstruct_forward_tuple(self):
        """Test reconstructing FORWARD hook with tuple output."""
        reconstructed = torch.randn(2, 3)
        original = (torch.randn(2, 3), torch.randn(2, 5), "other")
        
        result = reconstruct_hook_output(HookType.FORWARD, reconstructed, original)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert torch.equal(result[0], reconstructed)
        assert torch.equal(result[1], original[1])
        assert result[2] == "other"

    def test_reconstruct_forward_list(self):
        """Test reconstructing FORWARD hook with list output."""
        reconstructed = torch.randn(2, 3)
        original = [torch.randn(2, 3), torch.randn(2, 5), "other"]
        
        result = reconstruct_hook_output(HookType.FORWARD, reconstructed, original)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert torch.equal(result[0], reconstructed)
        assert torch.equal(result[1], original[1])
        assert result[2] == "other"

    def test_reconstruct_forward_object(self):
        """Test reconstructing FORWARD hook with object output."""
        reconstructed = torch.randn(2, 3)
        original = Mock()
        original.last_hidden_state = torch.randn(2, 3)
        
        result = reconstruct_hook_output(HookType.FORWARD, reconstructed, original)
        
        assert result is original
        assert torch.equal(original.last_hidden_state, reconstructed)

    def test_reconstruct_forward_object_no_hidden_state(self):
        """Test reconstructing FORWARD hook with object without last_hidden_state."""
        reconstructed = torch.randn(2, 3)
        original = Mock()
        del original.last_hidden_state
        
        result = reconstruct_hook_output(HookType.FORWARD, reconstructed, original)
        
        assert result is original

    def test_reconstruct_pre_forward_inputs(self):
        """Test reconstructing PRE_FORWARD hook inputs."""
        reconstructed = torch.randn(2, 3)
        original = (torch.randn(2, 3), "other", torch.randn(1, 2))
        
        result = reconstruct_hook_output(HookType.PRE_FORWARD, reconstructed, original)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert torch.equal(result[0], reconstructed)
        assert result[1] == "other"
        assert torch.equal(result[2], original[2])

    def test_reconstruct_pre_forward_empty_inputs(self):
        """Test reconstructing PRE_FORWARD hook with empty inputs."""
        reconstructed = torch.randn(2, 3)
        original = ()
        
        result = reconstruct_hook_output(HookType.PRE_FORWARD, reconstructed, original)
        
        assert isinstance(result, tuple)
        assert len(result) == 0


class TestReconstructForwardOutput:
    """Test _reconstruct_forward_output helper function."""

    def test_reconstruct_tensor(self):
        """Test reconstructing tensor output."""
        reconstructed = torch.randn(2, 3)
        original = torch.randn(2, 3)
        
        result = _reconstruct_forward_output(reconstructed, original)
        
        assert torch.equal(result, reconstructed)
        assert result is not original

    def test_reconstruct_tuple_with_tensor_first(self):
        """Test reconstructing tuple with tensor in first position."""
        reconstructed = torch.randn(2, 3)
        original = (torch.randn(2, 3), "other")
        
        result = _reconstruct_forward_output(reconstructed, original)
        
        assert isinstance(result, tuple)
        assert torch.equal(result[0], reconstructed)
        assert result[1] == "other"

    def test_reconstruct_tuple_with_tensor_second(self):
        """Test reconstructing tuple with tensor in second position."""
        reconstructed = torch.randn(2, 3)
        original = ("other", torch.randn(2, 3))
        
        result = _reconstruct_forward_output(reconstructed, original)
        
        assert isinstance(result, tuple)
        assert result[0] == "other"
        assert torch.equal(result[1], reconstructed)

    def test_reconstruct_list(self):
        """Test reconstructing list output."""
        reconstructed = torch.randn(2, 3)
        original = [torch.randn(2, 3), "other"]
        
        result = _reconstruct_forward_output(reconstructed, original)
        
        assert isinstance(result, list)
        assert torch.equal(result[0], reconstructed)
        assert result[1] == "other"

    def test_reconstruct_object_with_hidden_state(self):
        """Test reconstructing object with last_hidden_state."""
        reconstructed = torch.randn(2, 3)
        original = Mock()
        original.last_hidden_state = torch.randn(2, 3)
        
        result = _reconstruct_forward_output(reconstructed, original)
        
        assert result is original
        assert torch.equal(original.last_hidden_state, reconstructed)

    def test_reconstruct_object_without_hidden_state(self):
        """Test reconstructing object without last_hidden_state."""
        reconstructed = torch.randn(2, 3)
        original = Mock()
        
        result = _reconstruct_forward_output(reconstructed, original)
        
        assert result is original


class TestReconstructPreForwardInputs:
    """Test _reconstruct_pre_forward_inputs helper function."""

    def test_reconstruct_single_input(self):
        """Test reconstructing single input."""
        reconstructed = torch.randn(2, 3)
        original = (torch.randn(2, 3),)
        
        result = _reconstruct_pre_forward_inputs(reconstructed, original)
        
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert torch.equal(result[0], reconstructed)

    def test_reconstruct_multiple_inputs(self):
        """Test reconstructing multiple inputs."""
        reconstructed = torch.randn(2, 3)
        original = (torch.randn(2, 3), "other", torch.randn(1, 2))
        
        result = _reconstruct_pre_forward_inputs(reconstructed, original)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert torch.equal(result[0], reconstructed)
        assert result[1] == "other"
        assert torch.equal(result[2], original[2])

    def test_reconstruct_empty_inputs(self):
        """Test reconstructing empty inputs."""
        reconstructed = torch.randn(2, 3)
        original = ()
        
        result = _reconstruct_pre_forward_inputs(reconstructed, original)
        
        assert isinstance(result, tuple)
        assert len(result) == 0


class TestSaeUtilsIntegration:
    """Integration tests for SAE utils functions."""

    def test_full_pipeline_3d_tensor_forward(self):
        """Test full pipeline: extract, reshape, process, reshape back, reconstruct."""
        # Simulate FORWARD hook with 3D tensor output
        output = torch.randn(2, 5, 16)
        inputs = (torch.randn(1, 2),)
        
        # Extract
        tensor, original = extract_activation_tensor(HookType.FORWARD, inputs, output)
        assert tensor is not None
        
        # Reshape for SAE
        tensor_2d, original_shape, needs_reshape = reshape_for_sae(tensor)
        assert tensor_2d.shape == (10, 16)
        assert needs_reshape is True
        
        # Simulate SAE processing (just pass through for test)
        processed = tensor_2d * 2.0
        
        # Reshape back
        reconstructed_3d = reshape_from_sae(processed, original_shape, needs_reshape)
        assert reconstructed_3d.shape == (2, 5, 16)
        
        # Reconstruct output
        result = reconstruct_hook_output(HookType.FORWARD, reconstructed_3d, original)
        assert torch.equal(result, reconstructed_3d)

    def test_full_pipeline_2d_tensor_pre_forward(self):
        """Test full pipeline for PRE_FORWARD hook with 2D tensor."""
        # Simulate PRE_FORWARD hook with 2D tensor input
        inputs = (torch.randn(10, 16), "other")
        output = None
        
        # Extract
        tensor, original = extract_activation_tensor(HookType.PRE_FORWARD, inputs, output)
        assert tensor is not None
        
        # Reshape for SAE (no change needed)
        tensor_2d, original_shape, needs_reshape = reshape_for_sae(tensor)
        assert tensor_2d.shape == (10, 16)
        assert needs_reshape is False
        
        # Simulate SAE processing
        processed = tensor_2d * 2.0
        
        # Reshape back
        reconstructed = reshape_from_sae(processed, original_shape, needs_reshape)
        assert reconstructed.shape == (10, 16)
        
        # Reconstruct inputs
        result = reconstruct_hook_output(HookType.PRE_FORWARD, reconstructed, original)
        assert isinstance(result, tuple)
        assert torch.equal(result[0], reconstructed)
        assert result[1] == "other"

