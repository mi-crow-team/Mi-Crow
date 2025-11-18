"""Tests for static methods in Sae base class."""
import pytest
import torch

from amber.mechanistic.sae.sae import Sae


class TestSaeApplyActivationFn:
    """Test _apply_activation_fn static method."""
    
    def test_apply_relu_activation(self):
        """Test applying ReLU activation function."""
        tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = Sae._apply_activation_fn(tensor, "relu")
        
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
        assert torch.allclose(result, expected)
        assert result is not tensor  # Should return new tensor
    
    def test_apply_linear_activation(self):
        """Test applying linear (identity) activation function."""
        tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = Sae._apply_activation_fn(tensor, "linear")
        
        assert torch.allclose(result, tensor)
        assert result is tensor  # Should return same tensor for linear
    
    def test_apply_none_activation(self):
        """Test applying None activation (treated as linear)."""
        tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = Sae._apply_activation_fn(tensor, None)
        
        assert torch.allclose(result, tensor)
        assert result is tensor  # Should return same tensor
    
    def test_apply_unknown_activation_raises_error(self):
        """Test that unknown activation function raises ValueError."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Unknown activation function"):
            Sae._apply_activation_fn(tensor, "sigmoid")
    
    def test_apply_activation_preserves_shape(self):
        """Test that activation function preserves tensor shape."""
        shapes = [(10,), (5, 10), (2, 5, 10), (1, 2, 3, 4)]
        
        for shape in shapes:
            tensor = torch.randn(shape)
            result = Sae._apply_activation_fn(tensor, "relu")
            assert result.shape == shape
            
            result = Sae._apply_activation_fn(tensor, "linear")
            assert result.shape == shape
    
    def test_apply_relu_on_negative_values(self):
        """Test ReLU correctly zeros out negative values."""
        tensor = torch.tensor([-10.0, -5.0, -0.1, 0.0, 0.1, 5.0, 10.0])
        result = Sae._apply_activation_fn(tensor, "relu")
        
        expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.1, 5.0, 10.0])
        assert torch.allclose(result, expected)
    
    def test_apply_activation_on_empty_tensor(self):
        """Test activation function on empty tensor."""
        tensor = torch.tensor([])
        result = Sae._apply_activation_fn(tensor, "relu")
        assert result.shape == (0,)
        
        result = Sae._apply_activation_fn(tensor, "linear")
        assert result.shape == (0,)

