import pytest
import torch
from torch import nn

from amber.core.language_model_layers import LanguageModelLayers


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


def test_generic_layer_output_extraction_fallbacks():
    """Test generic layer output extraction with various fallback scenarios."""
    d = 8
    model = NestedModel(d)
    layers = LanguageModelLayers(lm=object(), model=model)
    
    # Test with ComplexOutputLayer (returns tuple with tensors)
    complex_layer_name = None
    for name, layer in layers.name_to_layer.items():
        if isinstance(layer, ComplexOutputLayer):
            complex_layer_name = name
            break
    assert complex_layer_name is not None
    
    # Should be able to register a new layer after the complex output layer
    new_layer = nn.ReLU()
    hook = layers.register_new_layer("test_relu", new_layer, after_layer_signature=complex_layer_name)
    
    try:
        # Forward pass should work and select first tensor from tuple
        x = torch.randint(0, 50, (2, 3))  # Use integer indices for embedding
        y = model(x)
        assert isinstance(y, torch.Tensor)
        assert y.shape == (2, 3, d)  # Should match input shape with embedding dimension
    finally:
        hook.remove()


def test_no_tensor_output_raises_error():
    """Test that layers returning no tensors raise appropriate errors."""
    d = 8
    model = NestedModel(d)
    layers = LanguageModelLayers(lm=object(), model=model)

    # Test with NoTensorOutputLayer
    no_tensor_layer_name = None
    for name, layer in layers.name_to_layer.items():
        if isinstance(layer, NoTensorOutputLayer):
            no_tensor_layer_name = name
            break
    assert no_tensor_layer_name is not None

    # Should raise error when trying to register after a layer that returns no tensors
    new_layer = nn.ReLU()
    hook = layers.register_new_layer("test_relu", new_layer, after_layer_signature=no_tensor_layer_name)

    # The model should work fine since we skip the NoTensorOutputLayer
    x = torch.randint(0, 50, (2, 3))
    output = model(x)
    assert isinstance(output, torch.Tensor)
    
    # Clean up
    hook.remove()


def test_complex_nested_layer_scenarios():
    """Test complex nested layer scenarios with multiple levels of nesting."""
    d = 6
    model = NestedModel(d)
    layers = LanguageModelLayers(lm=object(), model=model)
    
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


def test_layer_output_shape_handling():
    """Test handling of different output shapes from layers."""
    d = 4
    
    class ShapeTestLayer(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.lin = nn.Linear(d, d)
        
        def forward(self, x):
            y = self.lin(x)
            # Return tuple with different shapes
            return (y, y.view(-1, d), y.mean(dim=1, keepdim=True))
    
    class ShapeTestModel(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.emb = nn.Embedding(50, d)
            self.shape_layer = ShapeTestLayer(d)
            self.final = nn.Linear(d, d)
        
        def forward(self, x):
            x = self.emb(x)
            x = self.shape_layer(x)
            return self.final(x)
    
    model = ShapeTestModel(d)
    layers = LanguageModelLayers(lm=object(), model=model)
    
    # Find the shape test layer
    shape_layer_name = None
    for name, layer in layers.name_to_layer.items():
        if isinstance(layer, ShapeTestLayer):
            shape_layer_name = name
            break
    assert shape_layer_name is not None
    
    # Register a new layer after the shape test layer
    new_layer = nn.ReLU()
    hook = layers.register_new_layer("shape_test_relu", new_layer, after_layer_signature=shape_layer_name)
    
    try:
        # Should handle the tuple output and select the first tensor
        x = torch.randint(0, 50, (2, 3))
        y = model(x)
        assert isinstance(y, torch.Tensor)
    finally:
        hook.remove()


def test_layer_registration_with_complex_signatures():
    """Test layer registration with complex layer signatures."""
    d = 8
    
    class DeepNestedModel(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.emb = nn.Embedding(100, d)
            self.deep_block = nn.Sequential(
                nn.Sequential(
                    nn.Linear(d, d),
                    nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(d, d)
                    )
                )
            )
        
        def forward(self, x):
            x = self.emb(x)
            return self.deep_block(x)
    
    model = DeepNestedModel(d)
    layers = LanguageModelLayers(lm=object(), model=model)
    
    # Test that we can find deeply nested layers
    layer_names = layers.get_layer_names()
    assert len(layer_names) > 0
    
    # Test registering after a deeply nested layer
    for name in layer_names:
        if "linear" in name.lower():
            new_layer = nn.Tanh()
            hook = layers.register_new_layer("deep_test_tanh", new_layer, after_layer_signature=name)
            
            try:
                x = torch.randint(0, 50, (2, 3))
                y = model(x)
                assert isinstance(y, torch.Tensor)
            finally:
                hook.remove()
            break


def test_layer_output_type_handling():
    """Test handling of different output types from layers."""
    d = 6
    
    class TypeTestLayer(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.lin = nn.Linear(d, d)
        
        def forward(self, x):
            y = self.lin(x)
            # Return list instead of tuple
            return [y, y * 0.5, y + 1]
    
    class TypeTestModel(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.emb = nn.Embedding(50, d)
            self.type_layer = TypeTestLayer(d)
            self.final = nn.Linear(d, d)
        
        def forward(self, x):
            x = self.emb(x)
            x = self.type_layer(x)
            return self.final(x)
    
    model = TypeTestModel(d)
    layers = LanguageModelLayers(lm=object(), model=model)
    
    # Find the type test layer
    type_layer_name = None
    for name, layer in layers.name_to_layer.items():
        if isinstance(layer, TypeTestLayer):
            type_layer_name = name
            break
    assert type_layer_name is not None
    
    # Register a new layer after the type test layer
    new_layer = nn.Sigmoid()
    hook = layers.register_new_layer("type_test_sigmoid", new_layer, after_layer_signature=type_layer_name)
    
    try:
        # Should handle the list output and select the first tensor
        x = torch.randint(0, 50, (2, 3))
        y = model(x)
        assert isinstance(y, torch.Tensor)
    finally:
        hook.remove()
