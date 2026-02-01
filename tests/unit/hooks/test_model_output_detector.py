"""Tests for ModelOutputDetector."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from mi_crow.hooks.hook import HookType
from mi_crow.hooks.implementations.model_output_detector import ModelOutputDetector


class TestModelOutputDetectorInitialization:
    """Tests for ModelOutputDetector initialization."""

    @pytest.mark.parametrize("layer_signature", ["layer_0", 0, None, "model_outputs"])
    def test_init_with_various_layer_signatures(self, layer_signature):
        """Test initialization with various layer signature types."""
        detector = ModelOutputDetector(layer_signature=layer_signature)
        assert detector.layer_signature == layer_signature
        assert detector.hook_type == HookType.FORWARD
        assert detector.save_output_logits is True
        assert detector.save_output_hidden_state is False

    def test_init_with_custom_hook_id(self):
        """Test initialization with custom hook ID."""
        detector = ModelOutputDetector(layer_signature="layer_0", hook_id="custom_id")
        assert detector.id == "custom_id"

    @pytest.mark.parametrize(
        "save_output_logits,save_output_hidden_state",
        [
            (True, False),
            (False, True),
            (True, True),
            (False, False),
        ],
    )
    def test_init_with_save_flags(self, save_output_logits, save_output_hidden_state):
        """Test initialization with different save flag combinations."""
        detector = ModelOutputDetector(
            layer_signature="layer_0",
            save_output_logits=save_output_logits,
            save_output_hidden_state=save_output_hidden_state,
        )
        assert detector.save_output_logits == save_output_logits
        assert detector.save_output_hidden_state == save_output_hidden_state


class TestModelOutputDetectorExtractOutputTensor:
    """Tests for _extract_output_tensor method."""

    def test_extract_output_tensor_from_huggingface_output(self):
        """Test extracting logits and hidden_state from HuggingFace output object."""
        detector = ModelOutputDetector(layer_signature="layer_0")
        logits = torch.randn(2, 10, 50257)
        hidden_state = torch.randn(2, 10, 768)

        class HuggingFaceOutput:
            def __init__(self):
                self.logits = logits
                self.last_hidden_state = hidden_state

        output = HuggingFaceOutput()
        extracted_logits, extracted_hidden = detector._extract_output_tensor(output)
        assert extracted_logits is not None
        assert torch.equal(extracted_logits, logits)
        assert extracted_hidden is not None
        assert torch.equal(extracted_hidden, hidden_state)

    def test_extract_output_tensor_from_huggingface_output_logits_only(self):
        """Test extracting only logits when hidden_state is missing."""
        detector = ModelOutputDetector(layer_signature="layer_0")
        logits = torch.randn(2, 10, 50257)

        class HuggingFaceOutput:
            def __init__(self):
                self.logits = logits

        output = HuggingFaceOutput()
        extracted_logits, extracted_hidden = detector._extract_output_tensor(output)
        assert extracted_logits is not None
        assert torch.equal(extracted_logits, logits)
        assert extracted_hidden is None

    def test_extract_output_tensor_from_tuple(self):
        """Test extracting logits from tuple output."""
        detector = ModelOutputDetector(layer_signature="layer_0")
        logits = torch.randn(2, 10, 50257)
        output = (logits, torch.randn(2, 10, 768))
        extracted_logits, extracted_hidden = detector._extract_output_tensor(output)
        assert extracted_logits is not None
        assert torch.equal(extracted_logits, logits)
        assert extracted_hidden is None

    def test_extract_output_tensor_from_list(self):
        """Test extracting logits from list output."""
        detector = ModelOutputDetector(layer_signature="layer_0")
        logits = torch.randn(2, 10, 50257)
        output = [logits, torch.randn(2, 10, 768)]
        extracted_logits, extracted_hidden = detector._extract_output_tensor(output)
        assert extracted_logits is not None
        assert torch.equal(extracted_logits, logits)

    def test_extract_output_tensor_from_direct_tensor(self):
        """Test extracting logits from direct tensor output."""
        detector = ModelOutputDetector(layer_signature="layer_0")
        logits = torch.randn(2, 10, 50257)
        extracted_logits, extracted_hidden = detector._extract_output_tensor(logits)
        assert extracted_logits is not None
        assert torch.equal(extracted_logits, logits)
        assert extracted_hidden is None

    def test_extract_output_tensor_none_output(self):
        """Test extracting from None output."""
        detector = ModelOutputDetector(layer_signature="layer_0")
        extracted_logits, extracted_hidden = detector._extract_output_tensor(None)
        assert extracted_logits is None
        assert extracted_hidden is None

    def test_extract_output_tensor_empty_tuple(self):
        """Test extracting from empty tuple."""
        detector = ModelOutputDetector(layer_signature="layer_0")
        extracted_logits, extracted_hidden = detector._extract_output_tensor(())
        assert extracted_logits is None
        assert extracted_hidden is None

    def test_extract_output_tensor_tuple_with_non_tensor(self):
        """Test extracting from tuple with non-tensor first element."""
        detector = ModelOutputDetector(layer_signature="layer_0")
        output = ("string", torch.randn(2, 10, 50257))
        extracted_logits, extracted_hidden = detector._extract_output_tensor(output)
        assert extracted_logits is None
        assert extracted_hidden is None


class TestModelOutputDetectorProcessActivations:
    """Tests for process_activations method."""

    def test_process_activations_with_huggingface_output(self):
        """Test processing activations with HuggingFace output object."""
        detector = ModelOutputDetector(layer_signature="layer_0", save_output_logits=True)
        module = nn.Linear(10, 5)
        logits = torch.randn(2, 10, 50257)

        class HuggingFaceOutput:
            def __init__(self):
                self.logits = logits

        output = HuggingFaceOutput()
        detector.process_activations(module, (torch.randn(2, 10),), output)
        captured = detector.get_captured_output_logits()
        assert captured is not None
        assert torch.equal(captured, logits.cpu())
        assert captured.device.type == "cpu"
        assert detector.metadata["output_logits_shape"] == tuple(logits.shape)

    def test_process_activations_with_hidden_state(self):
        """Test processing activations with hidden_state."""
        detector = ModelOutputDetector(layer_signature="layer_0", save_output_hidden_state=True)
        module = nn.Linear(10, 5)
        hidden_state = torch.randn(2, 10, 768)

        class HuggingFaceOutput:
            def __init__(self):
                self.last_hidden_state = hidden_state

        output = HuggingFaceOutput()
        detector.process_activations(module, (torch.randn(2, 10),), output)
        captured = detector.get_captured_output_hidden_state()
        assert captured is not None
        assert torch.equal(captured, hidden_state.cpu())
        assert detector.metadata["output_hidden_state_shape"] == tuple(hidden_state.shape)

    def test_process_activations_with_both_logits_and_hidden_state(self):
        """Test processing activations with both logits and hidden_state."""
        detector = ModelOutputDetector(
            layer_signature="layer_0", save_output_logits=True, save_output_hidden_state=True
        )
        module = nn.Linear(10, 5)
        logits = torch.randn(2, 10, 50257)
        hidden_state = torch.randn(2, 10, 768)

        class HuggingFaceOutput:
            def __init__(self):
                self.logits = logits
                self.last_hidden_state = hidden_state

        output = HuggingFaceOutput()
        detector.process_activations(module, (torch.randn(2, 10),), output)
        assert detector.get_captured_output_logits() is not None
        assert detector.get_captured_output_hidden_state() is not None

    def test_process_activations_with_tuple_output(self):
        """Test processing activations with tuple output."""
        detector = ModelOutputDetector(layer_signature="layer_0", save_output_logits=True)
        module = nn.Linear(10, 5)
        logits = torch.randn(2, 10, 50257)
        output = (logits, torch.randn(2, 10, 768))
        detector.process_activations(module, (torch.randn(2, 10),), output)
        captured = detector.get_captured_output_logits()
        assert captured is not None
        assert torch.equal(captured, logits.cpu())

    def test_process_activations_with_direct_tensor(self):
        """Test processing activations with direct tensor output."""
        detector = ModelOutputDetector(layer_signature="layer_0", save_output_logits=True)
        module = nn.Linear(10, 5)
        logits = torch.randn(2, 10, 50257)
        detector.process_activations(module, (torch.randn(2, 10),), logits)
        captured = detector.get_captured_output_logits()
        assert captured is not None
        assert torch.equal(captured, logits.cpu())

    def test_process_activations_none_output(self):
        """Test processing activations when output is None."""
        detector = ModelOutputDetector(layer_signature="layer_0")
        module = nn.Linear(10, 5)
        detector.process_activations(module, (torch.randn(2, 10),), None)
        assert detector.get_captured_output_logits() is None
        assert detector.get_captured_output_hidden_state() is None

    def test_process_activations_save_output_logits_false(self):
        """Test that logits are not saved when save_output_logits is False."""
        detector = ModelOutputDetector(layer_signature="layer_0", save_output_logits=False)
        module = nn.Linear(10, 5)
        logits = torch.randn(2, 10, 50257)

        class HuggingFaceOutput:
            def __init__(self):
                self.logits = logits

        detector.process_activations(module, (torch.randn(2, 10),), HuggingFaceOutput())
        assert detector.get_captured_output_logits() is None

    def test_process_activations_save_output_hidden_state_false(self):
        """Test that hidden_state is not saved when save_output_hidden_state is False."""
        detector = ModelOutputDetector(layer_signature="layer_0", save_output_hidden_state=False)
        module = nn.Linear(10, 5)
        hidden_state = torch.randn(2, 10, 768)

        class HuggingFaceOutput:
            def __init__(self):
                self.last_hidden_state = hidden_state

        detector.process_activations(module, (torch.randn(2, 10),), HuggingFaceOutput())
        assert detector.get_captured_output_hidden_state() is None

    def test_process_activations_overwrites_previous(self):
        """Test that processing new activations overwrites previous."""
        detector = ModelOutputDetector(layer_signature="layer_0", save_output_logits=True)
        module = nn.Linear(10, 5)
        logits1 = torch.randn(2, 10, 50257)
        logits2 = torch.randn(3, 10, 50257)

        class Output1:
            def __init__(self):
                self.logits = logits1

        class Output2:
            def __init__(self):
                self.logits = logits2

        detector.process_activations(module, (torch.randn(2, 10),), Output1())
        captured1 = detector.get_captured_output_logits()
        detector.process_activations(module, (torch.randn(3, 10),), Output2())
        captured2 = detector.get_captured_output_logits()
        assert not torch.equal(captured1, captured2)
        assert torch.equal(captured2, logits2.cpu())

    def test_process_activations_exception_handling(self):
        """Test exception handling in process_activations."""
        detector = ModelOutputDetector(layer_signature="layer_0", save_output_logits=True)
        module = nn.Linear(10, 5)
        logits = torch.randn(2, 10, 50257)

        class HuggingFaceOutput:
            def __init__(self):
                self.logits = logits

        with patch.object(torch.Tensor, "detach", side_effect=RuntimeError("Test error")):
            with pytest.raises(RuntimeError, match="Error extracting outputs"):
                detector.process_activations(module, (torch.randn(2, 10),), HuggingFaceOutput())


class TestModelOutputDetectorGetCaptured:
    """Tests for get_captured methods."""

    def test_get_captured_output_logits_returns_tensor(self):
        """Test that get_captured_output_logits returns captured tensor."""
        detector = ModelOutputDetector(layer_signature="layer_0", save_output_logits=True)
        module = nn.Linear(10, 5)
        logits = torch.randn(2, 10, 50257)

        class HuggingFaceOutput:
            def __init__(self):
                self.logits = logits

        detector.process_activations(module, (torch.randn(2, 10),), HuggingFaceOutput())
        captured = detector.get_captured_output_logits()
        assert captured is not None
        assert isinstance(captured, torch.Tensor)
        assert captured.device.type == "cpu"

    def test_get_captured_output_logits_returns_none_when_nothing_captured(self):
        """Test that get_captured_output_logits returns None when nothing captured."""
        detector = ModelOutputDetector(layer_signature="layer_0")
        captured = detector.get_captured_output_logits()
        assert captured is None

    def test_get_captured_output_hidden_state_returns_tensor(self):
        """Test that get_captured_output_hidden_state returns captured tensor."""
        detector = ModelOutputDetector(layer_signature="layer_0", save_output_hidden_state=True)
        module = nn.Linear(10, 5)
        hidden_state = torch.randn(2, 10, 768)

        class HuggingFaceOutput:
            def __init__(self):
                self.last_hidden_state = hidden_state

        detector.process_activations(module, (torch.randn(2, 10),), HuggingFaceOutput())
        captured = detector.get_captured_output_hidden_state()
        assert captured is not None
        assert isinstance(captured, torch.Tensor)

    def test_get_captured_output_hidden_state_returns_none_when_nothing_captured(self):
        """Test that get_captured_output_hidden_state returns None when nothing captured."""
        detector = ModelOutputDetector(layer_signature="layer_0")
        captured = detector.get_captured_output_hidden_state()
        assert captured is None


class TestModelOutputDetectorClearCaptured:
    """Tests for clear_captured method."""

    def test_clear_captured_removes_all_tensors(self):
        """Test that clear_captured removes all captured tensors."""
        detector = ModelOutputDetector(
            layer_signature="layer_0", save_output_logits=True, save_output_hidden_state=True
        )
        module = nn.Linear(10, 5)
        logits = torch.randn(2, 10, 50257)
        hidden_state = torch.randn(2, 10, 768)

        class HuggingFaceOutput:
            def __init__(self):
                self.logits = logits
                self.last_hidden_state = hidden_state

        detector.process_activations(module, (torch.randn(2, 10),), HuggingFaceOutput())
        assert detector.get_captured_output_logits() is not None
        assert detector.get_captured_output_hidden_state() is not None
        detector.clear_captured()
        assert detector.get_captured_output_logits() is None
        assert detector.get_captured_output_hidden_state() is None
        assert "output_logits_shape" not in detector.metadata
        assert "output_hidden_state_shape" not in detector.metadata

    def test_clear_captured_when_nothing_captured(self):
        """Test that clear_captured works when nothing captured."""
        detector = ModelOutputDetector(layer_signature="layer_0")
        detector.clear_captured()
        assert detector.get_captured_output_logits() is None

    def test_clear_captured_partial_data(self):
        """Test that clear_captured handles partial data correctly."""
        detector = ModelOutputDetector(layer_signature="layer_0", save_output_logits=True)
        module = nn.Linear(10, 5)
        logits = torch.randn(2, 10, 50257)

        class HuggingFaceOutput:
            def __init__(self):
                self.logits = logits

        detector.process_activations(module, (torch.randn(2, 10),), HuggingFaceOutput())
        detector.clear_captured()
        assert detector.get_captured_output_logits() is None
        assert "output_logits_shape" not in detector.metadata
