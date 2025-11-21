"""Tests for language model utility functions."""

import pytest
import torch
from unittest.mock import Mock, MagicMock

from amber.language_model.utils import (
    extract_model_id,
    get_device_from_model,
    move_tensors_to_device,
    extract_logits_from_output,
)


class TestExtractModelId:
    """Test suite for extract_model_id."""

    def test_with_provided_model_id(self):
        """Test with provided model_id."""
        model = Mock()
        result = extract_model_id(model, "custom_model_id")
        assert result == "custom_model_id"

    def test_with_config_name_or_path(self):
        """Test extracting from model.config.name_or_path."""
        model = Mock()
        model.config = Mock()
        model.config.name_or_path = "org/model-name"
        result = extract_model_id(model, None)
        assert result == "org_model-name"

    def test_with_class_name(self):
        """Test using model class name."""
        class MyModel:
            pass
        model = MyModel()
        result = extract_model_id(model, None)
        assert result == "MyModel"

    def test_with_config_no_name_or_path(self):
        """Test with config but no name_or_path."""
        model = Mock()
        model.config = Mock()
        del model.config.name_or_path
        model.__class__.__name__ = "MyModel"
        result = extract_model_id(model, None)
        assert result == "MyModel"


class TestGetDeviceFromModel:
    """Test suite for get_device_from_model."""

    def test_with_parameters(self):
        """Test getting device from model parameters."""
        model = torch.nn.Linear(10, 5)
        device = get_device_from_model(model)
        assert isinstance(device, torch.device)

    def test_with_cuda_model(self):
        """Test getting device from CUDA model."""
        if torch.cuda.is_available():
            model = torch.nn.Linear(10, 5).cuda()
            device = get_device_from_model(model)
            assert device.type == "cuda"
        else:
            pytest.skip("CUDA not available")

    def test_with_no_parameters(self):
        """Test getting device when model has no parameters."""
        model = Mock()
        model.parameters = Mock(return_value=iter([]))
        device = get_device_from_model(model)
        assert device.type == "cpu"


class TestMoveTensorsToDevice:
    """Test suite for move_tensors_to_device."""

    def test_move_to_cpu(self):
        """Test moving tensors to CPU."""
        tensors = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        device = torch.device("cpu")
        result = move_tensors_to_device(tensors, device)
        
        assert result["input_ids"].device.type == "cpu"
        assert result["attention_mask"].device.type == "cpu"

    def test_move_to_cuda(self):
        """Test moving tensors to CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        tensors = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        device = torch.device("cuda")
        result = move_tensors_to_device(tensors, device)
        
        assert result["input_ids"].device.type == "cuda"
        assert result["attention_mask"].device.type == "cuda"

    def test_move_with_non_blocking(self):
        """Test moving tensors with non_blocking for CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        tensors = {
            "input_ids": torch.tensor([[1, 2, 3]]),
        }
        device = torch.device("cuda")
        result = move_tensors_to_device(tensors, device)
        
        assert result["input_ids"].device.type == "cuda"

    def test_empty_dict(self):
        """Test moving empty dictionary."""
        tensors = {}
        device = torch.device("cpu")
        result = move_tensors_to_device(tensors, device)
        assert result == {}


class TestExtractLogitsFromOutput:
    """Test suite for extract_logits_from_output."""

    def test_from_object_with_logits(self):
        """Test extracting logits from object with logits attribute."""
        mock_output = Mock()
        mock_logits = torch.tensor([[1.0, 2.0, 3.0]])
        mock_output.logits = mock_logits
        
        result = extract_logits_from_output(mock_output)
        assert torch.equal(result, mock_logits)

    def test_from_tuple(self):
        """Test extracting logits from tuple."""
        mock_logits = torch.tensor([[1.0, 2.0, 3.0]])
        output = (mock_logits, "other")
        
        result = extract_logits_from_output(output)
        assert torch.equal(result, mock_logits)

    def test_from_tensor(self):
        """Test extracting logits from tensor directly."""
        mock_logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        result = extract_logits_from_output(mock_logits)
        assert torch.equal(result, mock_logits)

    def test_from_empty_tuple(self):
        """Test extracting logits from empty tuple."""
        output = ()
        
        with pytest.raises(ValueError, match="Unable to extract logits"):
            extract_logits_from_output(output)

    def test_from_invalid_type(self):
        """Test extracting logits from invalid type."""
        with pytest.raises(ValueError, match="Unable to extract logits"):
            extract_logits_from_output("invalid")

    def test_from_none(self):
        """Test extracting logits from None."""
        with pytest.raises(ValueError, match="Unable to extract logits"):
            extract_logits_from_output(None)

    def test_from_list(self):
        """Test extracting logits from list."""
        with pytest.raises(ValueError, match="Unable to extract logits"):
            extract_logits_from_output([1, 2, 3])

