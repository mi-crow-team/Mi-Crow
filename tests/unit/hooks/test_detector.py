"""Tests for Detector base class."""

from unittest.mock import MagicMock, Mock

import pytest
import torch
from torch import nn

from mi_crow.hooks.detector import Detector
from mi_crow.hooks.hook import HookType
from tests.unit.fixtures.hooks import MockDetector
from tests.unit.fixtures.stores import create_mock_store


class ConcreteDetector(Detector):
    """Concrete implementation of Detector for testing."""

    def process_activations(self, module, input, output):
        """Implementation of abstract method."""
        self.metadata["test"] = "value"


class TestDetectorInitialization:
    """Tests for Detector initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        detector = ConcreteDetector()
        assert detector.hook_type == HookType.FORWARD
        assert detector.store is None
        assert detector.metadata == {}
        assert detector.tensor_metadata == {}

    def test_init_with_store(self):
        """Test initialization with store."""
        store = create_mock_store()
        detector = ConcreteDetector(store=store)
        assert detector.store is store

    def test_init_with_hook_type(self):
        """Test initialization with hook type."""
        detector = ConcreteDetector(hook_type=HookType.PRE_FORWARD)
        assert detector.hook_type == HookType.PRE_FORWARD

    def test_init_with_layer_signature(self):
        """Test initialization with layer signature."""
        detector = ConcreteDetector(layer_signature="layer_0")
        assert detector.layer_signature == "layer_0"


class TestDetectorProcessActivations:
    """Tests for process_activations method."""

    def test_process_activations_called(self):
        """Test that process_activations is called."""
        detector = ConcreteDetector()
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5)
        detector._hook_fn(module, (input_tensor,), output_tensor)
        assert detector.metadata["test"] == "value"

    def test_process_activations_when_disabled(self):
        """Test that process_activations is not called when disabled."""
        detector = ConcreteDetector()
        detector.disable()
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5)
        detector._hook_fn(module, (input_tensor,), output_tensor)
        assert "test" not in detector.metadata

    def test_process_activations_exception_handling(self):
        """Test that exceptions in process_activations are handled."""

        class FailingDetector(Detector):
            def process_activations(self, module, input, output):
                raise ValueError("Test error")

        detector = FailingDetector()
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5)
        with pytest.raises(RuntimeError, match="Error in detector"):
            detector._hook_fn(module, (input_tensor,), output_tensor)


class TestDetectorMetadata:
    """Tests for detector metadata handling."""

    def test_metadata_storage(self):
        """Test that metadata can be stored."""
        detector = ConcreteDetector()
        detector.metadata["key"] = "value"
        assert detector.metadata["key"] == "value"

    def test_tensor_metadata_storage(self):
        """Test that tensor metadata can be stored."""
        detector = ConcreteDetector()
        tensor = torch.randn(2, 10)
        detector.tensor_metadata["activations"] = tensor
        assert "activations" in detector.tensor_metadata
        assert torch.equal(detector.tensor_metadata["activations"], tensor)


class TestMockDetector:
    """Tests for MockDetector utility."""

    def test_mock_detector_processes_activations(self):
        """Test that MockDetector processes activations."""
        detector = MockDetector()
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5)
        detector._hook_fn(module, (input_tensor,), output_tensor)
        assert detector.processed_count == 1
        assert detector.metadata["count"] == 1

    def test_mock_detector_multiple_calls(self):
        """Test that MockDetector tracks multiple calls."""
        detector = MockDetector()
        module = nn.Linear(10, 5)
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 5)
        detector._hook_fn(module, (input_tensor,), output_tensor)
        detector._hook_fn(module, (input_tensor,), output_tensor)
        detector._hook_fn(module, (input_tensor,), output_tensor)
        assert detector.processed_count == 3
        assert detector.metadata["count"] == 3
