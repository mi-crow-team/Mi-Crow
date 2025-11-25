"""Tests for LanguageModelLayers."""

import pytest
from torch import nn

from amber.language_model.layers import LanguageModelLayers
from amber.language_model.context import LanguageModelContext
from amber.language_model.language_model import LanguageModel
from tests.unit.fixtures.language_models import create_language_model_from_mock
from tests.unit.fixtures.stores import create_temp_store
from tests.unit.fixtures.hooks import create_mock_detector


class TestLanguageModelLayers:
    """Tests for LanguageModelLayers."""

    def test_layers_initialization(self, temp_store):
        """Test layers initialization."""
        lm = create_language_model_from_mock(temp_store)
        layers = lm.layers
        
        assert layers.context == lm.context
        assert len(layers.name_to_layer) > 0
        assert len(layers.idx_to_layer) > 0

    def test_get_layer_names(self, temp_store):
        """Test getting layer names."""
        lm = create_language_model_from_mock(temp_store)
        names = lm.layers.get_layer_names()
        
        assert isinstance(names, list)
        assert len(names) > 0

    def test_get_layer_by_name(self, temp_store):
        """Test getting layer by name."""
        lm = create_language_model_from_mock(temp_store)
        names = lm.layers.get_layer_names()
        
        if names:
            layer = lm.layers._get_layer_by_name(names[0])
            assert isinstance(layer, nn.Module)

    def test_get_layer_by_name_not_found_raises_error(self, temp_store):
        """Test that getting non-existent layer raises ValueError."""
        lm = create_language_model_from_mock(temp_store)
        
        with pytest.raises(ValueError, match="Layer name 'nonexistent' not found"):
            lm.layers._get_layer_by_name("nonexistent")

    def test_get_layer_by_index(self, temp_store):
        """Test getting layer by index."""
        lm = create_language_model_from_mock(temp_store)
        layer = lm.layers._get_layer_by_index(0)
        assert isinstance(layer, nn.Module)

    def test_get_layer_by_index_not_found_raises_error(self, temp_store):
        """Test that getting non-existent index raises ValueError."""
        lm = create_language_model_from_mock(temp_store)
        
        with pytest.raises(ValueError, match="Layer index '999' not found"):
            lm.layers._get_layer_by_index(999)

    def test_register_detector(self, temp_store):
        """Test registering a detector."""
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=0)
        
        lm.layers.register_hook(0, detector)
        detectors = lm.layers.get_detectors()
        assert len(detectors) > 0

    def test_register_controller(self, temp_store):
        """Test registering a controller."""
        lm = create_language_model_from_mock(temp_store)
        from tests.unit.fixtures.hooks import create_mock_controller
        
        controller = create_mock_controller(layer_signature=0)
        lm.layers.register_hook(0, controller)
        controllers = lm.layers.get_controllers()
        assert len(controllers) > 0

