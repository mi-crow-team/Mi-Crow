"""Test the print_layer_names functionality in LanguageModelLayers."""

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


class MockModel(nn.Module):
    """Test model with various layer types."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 16)
        self.linear = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.batch_norm = nn.BatchNorm1d(32)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.batch_norm(x)
        return x


def test_print_layer_names_output(capsys):
    """Test that print_layer_names produces expected output."""
    model = MockModel()
    context = MockContext(language_model=object(), model=model)
    layers = LanguageModelLayers(context=context)
    
    # Capture the print output
    layers.print_layer_names()
    captured = capsys.readouterr()
    output = captured.out
    
    # Check that layer names are printed
    assert "embedding" in output.lower()
    assert "linear" in output.lower()
    assert "relu" in output.lower()
    assert "dropout" in output.lower()
    assert "batch_norm" in output.lower()
    
    # Check that weight shapes are shown for layers with weights
    assert "weight" in output.lower()
    assert "torch.size" in output.lower()


def test_print_layer_names_with_no_weight_layers(capsys):
    """Test print_layer_names with layers that don't have weights."""
    class NoWeightModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
        
        def forward(self, x):
            return self.dropout(self.relu(x))
    
    model = NoWeightModel()
    context = MockContext(language_model=object(), model=model)
    layers = LanguageModelLayers(context=context)
    
    # Capture the print output
    layers.print_layer_names()
    captured = capsys.readouterr()
    output = captured.out
    
    # Should show "No weight" for layers without weights
    assert "no weight" in output.lower()


def test_print_layer_names_with_mixed_layers(capsys):
    """Test print_layer_names with a mix of layers with and without weights."""
    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(20, 5)
        
        def forward(self, x):
            x = self.linear(x)
            x = self.relu(x)
            return self.linear2(x)
    
    model = MixedModel()
    context = MockContext(language_model=object(), model=model)
    layers = LanguageModelLayers(context=context)
    
    # Capture the print output
    layers.print_layer_names()
    captured = capsys.readouterr()
    output = captured.out
    
    # Should show both weight shapes and "No weight"
    assert "weight" in output.lower()
    assert "no weight" in output.lower()


def test_print_layer_names_empty_model(capsys):
    """Test print_layer_names with an empty model."""
    class EmptyModel(nn.Module):
        def forward(self, x):
            return x
    
    model = EmptyModel()
    context = MockContext(language_model=object(), model=model)
    layers = LanguageModelLayers(context=context)
    
    # Capture the print output
    layers.print_layer_names()
    captured = capsys.readouterr()
    output = captured.out
    
    # Should be empty or minimal output
    assert len(output.strip()) == 0 or "no weight" in output.lower()
