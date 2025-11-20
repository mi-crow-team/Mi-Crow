import pytest
import torch
from torch import nn
from dataclasses import dataclass, field
from typing import Dict, List, Any

from amber.language_model.language_model_layers import LanguageModelLayers


@dataclass
class MockContext:
    """Mock context for testing."""
    language_model: Any
    model: nn.Module | None = None
    _hook_registry: Dict[str | int, Dict[str, List[tuple[Any, Any]]]] = field(default_factory=dict)
    _hook_id_map: Dict[str, tuple[str | int, str, Any]] = field(default_factory=dict)


class ComplexOutputLayer(nn.Module):
    """Layer that returns complex nested outputs for testing edge cases."""
    
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)
    
    def forward(self, x):
        y = self.lin(x)
        # Return a tuple with multiple tensors to test selection logic
        return (y, torch.randn_like(y), {"meta": "data"})
    
    def extra_repr(self):
        return f"ComplexOutputLayer(dim={self.lin.in_features})"


class NoTensorOutputLayer(nn.Module):
    """Layer that returns no tensors to test error handling."""
    
    def forward(self, x):
        return {"no": "tensors", "here": True}
    
    def extra_repr(self):
        return "NoTensorOutputLayer()"


class SingleTensorLayer(nn.Module):
    """Simple layer that returns a single tensor."""
    
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)
    
    def forward(self, x):
        return self.lin(x)


class NestedModel(nn.Module):
    """Model with nested structure to test complex layer scenarios."""
    
    def __init__(self, d: int):
        super().__init__()
        self.emb = nn.Embedding(100, d)
        self.block1 = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            ComplexOutputLayer(d)
        )
        self.block2 = nn.Sequential(
            nn.Linear(d, d),
            NoTensorOutputLayer()
        )
        self.final = SingleTensorLayer(d)
    
    def forward(self, x):
        x = self.emb(x)
        x = self.block1(x)
        # Skip block2 as it returns no tensors
        # Handle the case where block1 might return a tuple
        if isinstance(x, tuple):
            x = x[0]  # Take the first element (tensor)
        return self.final(x)




def test_complex_nested_layer_scenarios():
    """Test complex nested layer scenarios with multiple levels of nesting."""
    d = 6
    model = NestedModel(d)
    context = MockContext(language_model=object(), model=model)
    layers = LanguageModelLayers(context=context)
    
    # Test that we can find layers at different nesting levels
    layer_names = layers.get_layer_names()
    
    # Should find layers at different nesting levels
    assert len(layer_names) > 0
    
    # Test accessing layers by different signatures
    for name in layer_names:
        layer = layers._get_layer_by_name(name)
        assert isinstance(layer, nn.Module)
    
        # Test that nested structure is properly flattened
        for idx, layer in layers.idx_to_layer.items():
            assert isinstance(layer, nn.Module)
            # Check that the layer exists in the name mapping
            layer_found = any(layer is l for l in layers.name_to_layer.values())
            assert layer_found


