"""Tests for InferenceEngine."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from amber.language_model.inference import InferenceEngine
from amber.language_model.language_model import LanguageModel
from tests.unit.fixtures import (
    create_language_model,
    create_mock_tokenizer,
)


class TestInferenceEngine:
    """Test suite for InferenceEngine."""

    def test_init(self, mock_language_model):
        """Test initialization."""
        engine = InferenceEngine(mock_language_model)
        assert engine.lm == mock_language_model

    def test_prepare_tokenizer_kwargs_none(self, mock_language_model):
        """Test preparing tokenizer kwargs with None."""
        engine = InferenceEngine(mock_language_model)
        kwargs = engine.prepare_tokenizer_kwargs(None)
        
        assert kwargs["padding"] is True
        assert kwargs["truncation"] is True
        assert kwargs["return_tensors"] == "pt"

    def test_prepare_tokenizer_kwargs_with_kwargs(self, mock_language_model):
        """Test preparing tokenizer kwargs with existing kwargs."""
        engine = InferenceEngine(mock_language_model)
        kwargs = engine.prepare_tokenizer_kwargs({"max_length": 128, "padding": False})
        
        assert kwargs["max_length"] == 128
        assert kwargs["padding"] is False  # User override
        assert kwargs["truncation"] is True
        assert kwargs["return_tensors"] == "pt"

    def test_setup_trackers_with_tracker(self, mock_language_model):
        """Test setting up trackers when tracker exists."""
        engine = InferenceEngine(mock_language_model)
        mock_tracker = Mock()
        mock_tracker.enabled = True
        mock_language_model._input_tracker = mock_tracker
        
        engine.setup_trackers(["text1", "text2"])
        
        mock_tracker.set_current_texts.assert_called_once_with(["text1", "text2"])

    def test_setup_trackers_no_tracker(self, mock_language_model):
        """Test setting up trackers when tracker is None."""
        engine = InferenceEngine(mock_language_model)
        mock_language_model._input_tracker = None
        
        # Should not raise
        engine.setup_trackers(["text1", "text2"])

    def test_setup_trackers_disabled(self, mock_language_model):
        """Test setting up trackers when tracker is disabled."""
        engine = InferenceEngine(mock_language_model)
        mock_tracker = Mock()
        mock_tracker.enabled = False
        mock_language_model._input_tracker = mock_tracker
        
        # Should not call set_current_texts
        engine.setup_trackers(["text1", "text2"])
        mock_tracker.set_current_texts.assert_not_called()

    def test_prepare_controllers_with_controllers_true(self, mock_language_model):
        """Test preparing controllers when with_controllers is True."""
        engine = InferenceEngine(mock_language_model)
        mock_language_model.layers.get_controllers = Mock(return_value=[])
        
        result = engine.prepare_controllers(with_controllers=True)
        
        assert result == []
        mock_language_model.layers.get_controllers.assert_not_called()

    def test_prepare_controllers_with_controllers_false(self, mock_language_model):
        """Test preparing controllers when with_controllers is False."""
        engine = InferenceEngine(mock_language_model)
        controller1 = Mock()
        controller1.enabled = True
        controller2 = Mock()
        controller2.enabled = False
        controller3 = Mock()
        controller3.enabled = True
        
        mock_language_model.layers.get_controllers = Mock(return_value=[controller1, controller2, controller3])
        
        result = engine.prepare_controllers(with_controllers=False)
        
        assert result == [controller1, controller3]
        controller1.disable.assert_called_once()
        controller2.disable.assert_not_called()
        controller3.disable.assert_called_once()

    def test_restore_controllers(self, mock_language_model):
        """Test restoring controllers."""
        engine = InferenceEngine(mock_language_model)
        controller1 = Mock()
        controller2 = Mock()
        
        engine.restore_controllers([controller1, controller2])
        
        controller1.enable.assert_called_once()
        controller2.enable.assert_called_once()

    def test_run_model_forward_no_autocast(self, mock_language_model):
        """Test running model forward without autocast."""
        engine = InferenceEngine(mock_language_model)
        enc = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_output = Mock()
        mock_language_model.context.model = Mock(return_value=mock_output)
        
        with patch("torch.inference_mode"):
            output = engine.run_model_forward(enc, autocast=False, device_type="cpu", autocast_dtype=None)
        
        assert output == mock_output
        mock_language_model.context.model.assert_called_once_with(**enc)

    def test_run_model_forward_with_autocast_cuda(self, mock_language_model):
        """Test running model forward with autocast on CUDA."""
        engine = InferenceEngine(mock_language_model)
        enc = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_output = Mock()
        mock_language_model.context.model = Mock(return_value=mock_output)
        
        with patch("torch.inference_mode"):
            with patch("torch.autocast") as mock_autocast:
                mock_autocast.return_value.__enter__ = Mock()
                mock_autocast.return_value.__exit__ = Mock(return_value=None)
                output = engine.run_model_forward(enc, autocast=True, device_type="cuda", autocast_dtype=None)
        
        assert output == mock_output
        mock_autocast.assert_called_once_with("cuda", dtype=torch.float16)

    def test_run_model_forward_with_autocast_custom_dtype(self, mock_language_model):
        """Test running model forward with autocast and custom dtype."""
        engine = InferenceEngine(mock_language_model)
        enc = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_output = Mock()
        mock_language_model.context.model = Mock(return_value=mock_output)
        
        with patch("torch.inference_mode"):
            with patch("torch.autocast") as mock_autocast:
                mock_autocast.return_value.__enter__ = Mock()
                mock_autocast.return_value.__exit__ = Mock(return_value=None)
                output = engine.run_model_forward(enc, autocast=True, device_type="cuda", autocast_dtype=torch.bfloat16)
        
        assert output == mock_output
        mock_autocast.assert_called_once_with("cuda", dtype=torch.bfloat16)

    def test_run_model_forward_autocast_cpu(self, mock_language_model):
        """Test running model forward with autocast on CPU (should not use autocast)."""
        engine = InferenceEngine(mock_language_model)
        enc = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_output = Mock()
        mock_language_model.context.model = Mock(return_value=mock_output)
        
        with patch("torch.inference_mode"):
            with patch("torch.autocast") as mock_autocast:
                output = engine.run_model_forward(enc, autocast=True, device_type="cpu", autocast_dtype=None)
        
        assert output == mock_output
        mock_autocast.assert_not_called()

    def test_execute_inference_success(self, mock_language_model):
        """Test executing inference successfully."""
        engine = InferenceEngine(mock_language_model)
        texts = ["text1", "text2"]
        
        mock_enc = {"input_ids": torch.tensor([[1, 2], [3, 4]])}
        mock_output = Mock()
        
        mock_language_model.tokenize = Mock(return_value=mock_enc)
        mock_language_model.context.model = Mock(return_value=mock_output)
        mock_language_model.context.model.eval = Mock()
        mock_language_model.layers.get_controllers = Mock(return_value=[])
        
        with patch("amber.language_model.inference.get_device_from_model", return_value=torch.device("cpu")):
            with patch("amber.language_model.inference.move_tensors_to_device", return_value=mock_enc):
                with patch("torch.inference_mode"):
                    output, enc = engine.execute_inference(texts)
        
        assert output == mock_output
        assert enc == mock_enc
        mock_language_model.tokenize.assert_called_once()
        mock_language_model.context.model.eval.assert_called_once()

    def test_execute_inference_empty_texts(self, mock_language_model):
        """Test executing inference with empty texts."""
        engine = InferenceEngine(mock_language_model)
        
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            engine.execute_inference([])

    def test_execute_inference_no_tokenizer(self, mock_language_model):
        """Test executing inference when tokenizer is None."""
        engine = InferenceEngine(mock_language_model)
        mock_language_model.context.tokenizer = None
        
        with pytest.raises(ValueError, match="Tokenizer must be initialized"):
            engine.execute_inference(["text1"])

    def test_execute_inference_with_controllers_disabled(self, mock_language_model):
        """Test executing inference with controllers disabled."""
        engine = InferenceEngine(mock_language_model)
        texts = ["text1"]
        
        controller = Mock()
        controller.enabled = True
        mock_language_model.layers.get_controllers = Mock(return_value=[controller])
        mock_language_model.tokenize = Mock(return_value={"input_ids": torch.tensor([[1]])})
        mock_language_model.context.model = Mock(return_value=Mock())
        mock_language_model.context.model.eval = Mock()
        
        with patch("amber.language_model.inference.get_device_from_model", return_value=torch.device("cpu")):
            with patch("amber.language_model.inference.move_tensors_to_device", return_value={"input_ids": torch.tensor([[1]])}):
                with patch("torch.inference_mode"):
                    engine.execute_inference(texts, with_controllers=False)
        
        controller.disable.assert_called_once()
        controller.enable.assert_called_once()

    def test_extract_logits_from_output_object(self, mock_language_model):
        """Test extracting logits from output object with logits attribute."""
        engine = InferenceEngine(mock_language_model)
        mock_output = Mock()
        mock_logits = torch.tensor([[1.0, 2.0, 3.0]])
        mock_output.logits = mock_logits
        
        result = engine.extract_logits(mock_output)
        
        assert torch.equal(result, mock_logits)

    def test_extract_logits_from_tuple(self, mock_language_model):
        """Test extracting logits from tuple output."""
        engine = InferenceEngine(mock_language_model)
        mock_logits = torch.tensor([[1.0, 2.0, 3.0]])
        output = (mock_logits,)
        
        result = engine.extract_logits(output)
        
        assert torch.equal(result, mock_logits)

    def test_extract_logits_from_tensor(self, mock_language_model):
        """Test extracting logits from tensor output."""
        engine = InferenceEngine(mock_language_model)
        mock_logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        result = engine.extract_logits(mock_logits)
        
        assert torch.equal(result, mock_logits)

    def test_extract_logits_invalid_output(self, mock_language_model):
        """Test extracting logits from invalid output."""
        engine = InferenceEngine(mock_language_model)
        
        with pytest.raises(ValueError, match="Unable to extract logits"):
            engine.extract_logits("invalid")

