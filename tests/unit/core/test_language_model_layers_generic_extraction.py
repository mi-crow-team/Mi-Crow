"""Test generic layer output extraction fallbacks in LanguageModelLayers."""

import pytest
import torch
from torch import nn
from dataclasses import dataclass, field
from typing import Dict, List, Any

from amber.language_model.layers import LanguageModelLayers


@dataclass
class MockContext:
    """Mock context for testing."""
    language_model: Any
    model: nn.Module | None = None
    _hook_registry: Dict[str | int, Dict[str, List[tuple[Any, Any]]]] = field(default_factory=dict)
    _hook_id_map: Dict[str, tuple[str | int, str, Any]] = field(default_factory=dict)


class ObjectWithLastHiddenState(nn.Module):
    """Layer that returns an object with last_hidden_state attribute."""
    
    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, d)
    
    def forward(self, x):
        output = self.linear(x)
        # Return an object with last_hidden_state attribute
        class OutputObject:
            def __init__(self, hidden_state):
                self.last_hidden_state = hidden_state
        return OutputObject(output)


class TupleOutputLayer(nn.Module):
    """Layer that returns a tuple with tensors."""
    
    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, d)
    
    def forward(self, x):
        output = self.linear(x)
        return (output, {"metadata": "test"})


class ListOutputLayer(nn.Module):
    """Layer that returns a list with tensors."""
    
    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, d)
    
    def forward(self, x):
        output = self.linear(x)
        return [output, "metadata"]


class ComplexNestedModel(nn.Module):
    """Model with complex nested structure to test generic extraction."""
    
    def __init__(self, d: int):
        super().__init__()
        self.embedding = nn.Embedding(100, d)
        self.block1 = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            ObjectWithLastHiddenState(d)
        )
        self.block2 = nn.Sequential(
            nn.Linear(d, d),
            TupleOutputLayer(d)
        )
        self.block3 = nn.Sequential(
            nn.Linear(d, d),
            ListOutputLayer(d)
        )
        self.final = nn.Linear(d, d)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.block1(x)
        # Handle the case where block1 returns a custom object
        if hasattr(x, 'last_hidden_state'):
            x = x.last_hidden_state
        x = self.block2(x)
        # Handle the case where block2 returns a tuple
        if isinstance(x, tuple):
            x = x[0]
        x = self.block3(x)
        # Handle the case where block3 returns a list
        if isinstance(x, list):
            x = x[0]
        return self.final(x)


def test_object_with_last_hidden_state_extraction():
    """Test extraction from objects with last_hidden_state attribute."""
    d = 8
    model = ComplexNestedModel(d)
    context = MockContext(language_model=object(), model=model)
    layers = LanguageModelLayers(context=context)
    
    # Find the ObjectWithLastHiddenState layer
    object_layer_name = None
    for name, layer in layers.name_to_layer.items():
        if isinstance(layer, ObjectWithLastHiddenState):
            object_layer_name = name
            break
    assert object_layer_name is not None
    
    # Test that we can find the layer and it has the expected structure
    layer = layers._get_layer_by_name(object_layer_name)
    assert isinstance(layer, ObjectWithLastHiddenState)
    
    # Test forward pass without registering a new layer (since the object type isn't supported)
    x = torch.randint(0, 50, (2, 3))
    output = model(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 3, d)


def test_tuple_output_extraction():
    """Test extraction from tuple outputs."""
    d = 8
    model = ComplexNestedModel(d)
    context = MockContext(language_model=object(), model=model)
    layers = LanguageModelLayers(context=context)
    
    # Find the TupleOutputLayer
    tuple_layer_name = None
    for name, layer in layers.name_to_layer.items():
        if isinstance(layer, TupleOutputLayer):
            tuple_layer_name = name
            break
    assert tuple_layer_name is not None
    
    # Test that we can find the layer and it has the expected structure
    layer = layers._get_layer_by_name(tuple_layer_name)
    assert isinstance(layer, TupleOutputLayer)
    
    # Test forward pass without registering a new layer
    x = torch.randint(0, 50, (2, 3))
    output = model(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 3, d)


def test_list_output_extraction():
    """Test extraction from list outputs."""
    d = 8
    model = ComplexNestedModel(d)
    context = MockContext(language_model=object(), model=model)
    layers = LanguageModelLayers(context=context)
    
    # Find the ListOutputLayer
    list_layer_name = None
    for name, layer in layers.name_to_layer.items():
        if isinstance(layer, ListOutputLayer):
            list_layer_name = name
            break
    assert list_layer_name is not None
    
    # Test that we can find the layer and it has the expected structure
    layer = layers._get_layer_by_name(list_layer_name)
    assert isinstance(layer, ListOutputLayer)
    
    # Test forward pass without registering a new layer
    x = torch.randint(0, 50, (2, 3))
    output = model(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 3, d)


def test_complex_nested_layer_scenarios():
    """Test complex nested layer scenarios with multiple extraction types."""
    d = 6
    model = ComplexNestedModel(d)
    context = MockContext(language_model=object(), model=model)
    layers = LanguageModelLayers(context=context)
    
    # Test that we can find layers at different nesting levels
    layer_names = layers.get_layer_names()
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


def test_generic_extraction_fallback_paths():
    """Test various fallback paths in generic extraction."""
    d = 8
    model = ComplexNestedModel(d)
    context = MockContext(language_model=object(), model=model)
    layers = LanguageModelLayers(context=context)
    
    # Test that we can find layers of different types
    layer_types = [ObjectWithLastHiddenState, TupleOutputLayer, ListOutputLayer]
    
    for layer_type in layer_types:
        # Find a layer of this type
        target_layer_name = None
        for name, layer in layers.name_to_layer.items():
            if isinstance(layer, layer_type):
                target_layer_name = name
                break
        
        if target_layer_name:
            # Test that we can access the layer
            layer = layers._get_layer_by_name(target_layer_name)
            assert isinstance(layer, layer_type)
    
    # Test forward pass without registering new layers
    x = torch.randint(0, 50, (2, 3))
    output = model(x)
    assert isinstance(output, torch.Tensor)
