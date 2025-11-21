"""Tests for LanguageModel core class."""

import pytest

from amber.language_model.language_model import LanguageModel
from tests.unit.fixtures.language_models import create_language_model_from_mock
from tests.unit.fixtures.stores import create_temp_store
from tests.unit.fixtures.hooks import create_mock_detector, create_mock_controller


class TestLanguageModelInitialization:
    """Tests for LanguageModel initialization."""

    def test_init_with_model_and_tokenizer(self, temp_store):
        """Test initialization with model and tokenizer."""
        lm = create_language_model_from_mock(temp_store)
        
        assert lm.model is not None
        assert lm.tokenizer is not None
        assert lm.store == temp_store
        assert lm.model_id is not None

    def test_properties(self, temp_store):
        """Test LanguageModel properties."""
        lm = create_language_model_from_mock(temp_store)
        
        assert lm.model == lm.context.model
        assert lm.tokenizer == lm.context.tokenizer
        assert lm.model_id == lm.context.model_id
        assert lm.store == lm.context.store


class TestLanguageModelTokenize:
    """Tests for tokenize method."""

    def test_tokenize_single_text(self, temp_store):
        """Test tokenizing single text."""
        from unittest.mock import patch
        lm = create_language_model_from_mock(temp_store)
        
        # Mock the tokenizer.tokenize method to avoid complex transformers interactions
        with patch.object(lm.lm_tokenizer, 'tokenize') as mock_tokenize:
            mock_tokenize.return_value = {"input_ids": [[1, 2, 3]]}
            result = lm.tokenize(["Hello"])
            
            assert result is not None
            mock_tokenize.assert_called_once()

    def test_tokenize_multiple_texts(self, temp_store):
        """Test tokenizing multiple texts."""
        from unittest.mock import patch
        lm = create_language_model_from_mock(temp_store)
        
        # Mock the tokenizer.tokenize method
        with patch.object(lm.lm_tokenizer, 'tokenize') as mock_tokenize:
            mock_tokenize.return_value = {"input_ids": [[1, 2], [3, 4]]}
            result = lm.tokenize(["Hello", "World"])
            
            assert result is not None
            mock_tokenize.assert_called_once()


class TestLanguageModelForwards:
    """Tests for forwards method."""

    def test_forwards_single_text(self, temp_store):
        """Test forward pass with single text."""
        from unittest.mock import patch, MagicMock
        import torch
        
        lm = create_language_model_from_mock(temp_store)
        
        # Mock the inference engine
        mock_output = MagicMock()
        mock_encodings = {"input_ids": torch.tensor([[1, 2, 3]])}
        with patch.object(lm._inference_engine, 'execute_inference') as mock_execute:
            mock_execute.return_value = (mock_output, mock_encodings)
            output, encodings = lm.forwards(["Hello"])
            
            assert output is not None
            assert encodings is not None
            mock_execute.assert_called_once()

    def test_forwards_multiple_texts(self, temp_store):
        """Test forward pass with multiple texts."""
        from unittest.mock import patch, MagicMock
        import torch
        
        lm = create_language_model_from_mock(temp_store)
        
        # Mock the inference engine
        mock_output = MagicMock()
        mock_encodings = {"input_ids": torch.tensor([[1, 2], [3, 4]])}
        with patch.object(lm._inference_engine, 'execute_inference') as mock_execute:
            mock_execute.return_value = (mock_output, mock_encodings)
            output, encodings = lm.forwards(["Hello", "World"])
            
            assert output is not None
            assert encodings is not None
            mock_execute.assert_called_once()


class TestLanguageModelGenerate:
    """Tests for generate method."""

    def test_generate_single_text(self, temp_store):
        """Test generation with single text."""
        from unittest.mock import patch, MagicMock
        import torch
        
        lm = create_language_model_from_mock(temp_store)
        
        # Mock the inference engine and tokenizer
        mock_output = MagicMock()
        mock_encodings = {"input_ids": torch.tensor([[1, 2, 3]])}
        with patch.object(lm._inference_engine, 'execute_inference') as mock_execute, \
             patch.object(lm._inference_engine, 'extract_logits') as mock_extract, \
             patch.object(lm.tokenizer, 'decode') as mock_decode:
            
            mock_execute.return_value = (mock_output, mock_encodings)
            mock_extract.return_value = torch.randn(1, 3, 1000)  # [batch, seq, vocab]
            mock_decode.return_value = "Generated text"
            
            result = lm.generate(["Hello"])
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] == "Generated text"

    def test_generate_empty_texts_raises_error(self, temp_store):
        """Test that empty texts list raises ValueError."""
        lm = create_language_model_from_mock(temp_store)
        
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            lm.generate([])


class TestLanguageModelHooks:
    """Tests for hook management."""

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
        controller = create_mock_controller(layer_signature=0)
        
        lm.layers.register_hook(0, controller)
        controllers = lm.layers.get_controllers()
        assert len(controllers) > 0

