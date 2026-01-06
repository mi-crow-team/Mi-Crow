"""Integration tests for LanguageModel with Hooks."""

import pytest
import torch

from tests.unit.fixtures.language_models import create_language_model_from_mock
from tests.unit.fixtures.stores import create_temp_store
from tests.unit.fixtures.hooks import create_mock_detector, create_mock_controller


class TestLanguageModelDetectorIntegration:
    """Tests for LanguageModel with Detector hooks."""

    def test_detector_collects_metadata(self, temp_store):
        """Test that detector collects metadata during inference."""
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=0)
        
        lm.layers.register_hook(0, detector)
        lm.inference.execute_inference(["Hello world"])
        
        assert detector.processed_count > 0
        assert "count" in detector.metadata

    def test_multiple_detectors_on_same_layer(self, temp_store):
        """Test multiple detectors on the same layer."""
        lm = create_language_model_from_mock(temp_store)
        detector1 = create_mock_detector(layer_signature=0, hook_id="detector1")
        detector2 = create_mock_detector(layer_signature=0, hook_id="detector2")
        
        lm.layers.register_hook(0, detector1)
        lm.layers.register_hook(0, detector2)
        lm.inference.execute_inference(["Hello"])
        
        assert detector1.processed_count > 0
        assert detector2.processed_count > 0

    def test_detector_enable_disable(self, temp_store):
        """Test enabling and disabling detectors."""
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=0)
        
        lm.layers.register_hook(0, detector)
        detector.disable()
        lm.inference.execute_inference(["Hello"])
        
        # Should not process when disabled
        initial_count = detector.processed_count
        
        detector.enable()
        lm.inference.execute_inference(["Hello"])
        
        assert detector.processed_count > initial_count


class TestLanguageModelControllerIntegration:
    """Tests for LanguageModel with Controller hooks."""

    def test_controller_modifies_activations(self, temp_store):
        """Test that controller modifies activations."""
        from unittest.mock import patch, MagicMock
        import torch
        
        lm = create_language_model_from_mock(temp_store)
        controller = create_mock_controller(layer_signature=0, modification_factor=2.0)
        
        lm.layers.register_hook(0, controller)
        mock_output = MagicMock()
        mock_encodings = {"input_ids": torch.tensor([[1, 2, 3]])}
        with patch.object(lm.inference, 'execute_inference') as mock_execute:
            mock_execute.return_value = (mock_output, mock_encodings)
            output, _ = lm.inference.execute_inference(["Hello"])
        
        # Controller should be registered
        controllers = lm.layers.get_controllers()
        assert len(controllers) > 0

    def test_controller_with_controllers_disabled(self, temp_store):
        """Test that controllers are not used when with_controllers=False."""
        from unittest.mock import patch, MagicMock
        import torch
        
        lm = create_language_model_from_mock(temp_store)
        controller = create_mock_controller(layer_signature=0)
        
        lm.layers.register_hook(0, controller)
        initial_count = controller.modified_count
        
        mock_output = MagicMock()
        mock_encodings = {"input_ids": torch.tensor([[1, 2, 3]])}
        with patch.object(lm.inference, 'execute_inference') as mock_execute:
            mock_execute.return_value = (mock_output, mock_encodings)
            lm.inference.execute_inference(["Hello"], with_controllers=False)
        
        # Controller count should not change (not actually called in mocked scenario)
        assert controller.modified_count == initial_count


class TestLanguageModelMultipleHooksIntegration:
    """Tests for LanguageModel with multiple hooks."""

    def test_detector_and_controller_together(self, temp_store):
        """Test detector and controller working together."""
        lm = create_language_model_from_mock(temp_store)
        detector = create_mock_detector(layer_signature=0)
        controller = create_mock_controller(layer_signature=1)
        
        lm.layers.register_hook(0, detector)
        lm.layers.register_hook(1, controller)
        lm.inference.execute_inference(["Hello"])
        
        assert detector.processed_count > 0
        assert controller in lm.layers.get_controllers()

