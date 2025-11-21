"""Tests for LanguageModel initialization functions."""

import pytest
from unittest.mock import patch, MagicMock
from torch import nn

from amber.language_model.initialization import initialize_model_id, create_from_huggingface, create_from_local_torch
from amber.language_model.language_model import LanguageModel
from tests.unit.fixtures.stores import create_temp_store


class TestInitializeModelId:
    """Tests for initialize_model_id function."""

    def test_initialize_model_id_with_provided_id(self, temp_store):
        """Test initialization with provided model ID."""
        model = nn.Linear(10, 5)
        model_id = initialize_model_id(model, "custom_id")
        assert model_id == "custom_id"

    def test_initialize_model_id_auto_generated(self, temp_store):
        """Test auto-generation of model ID."""
        model = nn.Linear(10, 5)
        model_id = initialize_model_id(model, None)
        assert model_id is not None
        assert isinstance(model_id, str)


class TestCreateFromHuggingface:
    """Tests for create_from_huggingface function."""

    def test_create_from_huggingface_success(self, temp_store):
        """Test creating LanguageModel from HuggingFace."""
        with patch("amber.language_model.initialization.AutoTokenizer") as mock_tokenizer_class, \
             patch("amber.language_model.initialization.AutoModelForCausalLM") as mock_model_class:
            
            mock_tokenizer = MagicMock()
            mock_model = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model
            
            lm = create_from_huggingface(
                LanguageModel,
                "test/model",
                temp_store
            )
            
            assert lm.model == mock_model
            assert lm.tokenizer == mock_tokenizer
            # Check that from_pretrained was called with model name and tokenizer_params
            mock_tokenizer_class.from_pretrained.assert_called_once()
            assert mock_tokenizer_class.from_pretrained.call_args[0][0] == "test/model"
            mock_model_class.from_pretrained.assert_called_once()
            assert mock_model_class.from_pretrained.call_args[0][0] == "test/model"

    def test_create_from_huggingface_with_params(self, temp_store):
        """Test creating with tokenizer and model parameters."""
        with patch("amber.language_model.initialization.AutoTokenizer") as mock_tokenizer_class, \
             patch("amber.language_model.initialization.AutoModelForCausalLM") as mock_model_class:
            
            mock_tokenizer = MagicMock()
            mock_model = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model
            
            tokenizer_params = {"padding": True}
            model_params = {"torch_dtype": "float16"}
            
            lm = create_from_huggingface(
                LanguageModel,
                "test/model",
                temp_store,
                tokenizer_params=tokenizer_params,
                model_params=model_params
            )
            
            # Check that from_pretrained was called with correct arguments
            mock_tokenizer_class.from_pretrained.assert_called_once()
            assert mock_tokenizer_class.from_pretrained.call_args[0][0] == "test/model"
            assert mock_tokenizer_class.from_pretrained.call_args[1] == tokenizer_params
            
            mock_model_class.from_pretrained.assert_called_once()
            assert mock_model_class.from_pretrained.call_args[0][0] == "test/model"
            assert mock_model_class.from_pretrained.call_args[1] == model_params

    def test_create_from_huggingface_empty_model_name_raises_error(self, temp_store):
        """Test that empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            create_from_huggingface(LanguageModel, "", temp_store)

    def test_create_from_huggingface_none_store_raises_error(self):
        """Test that None store raises ValueError."""
        with pytest.raises(ValueError, match="store cannot be None"):
            create_from_huggingface(LanguageModel, "test/model", None)

    def test_create_from_huggingface_load_failure_raises_error(self, temp_store):
        """Test that load failure raises RuntimeError."""
        with patch("amber.language_model.initialization.AutoTokenizer") as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.side_effect = Exception("Network error")
            
            with pytest.raises(RuntimeError, match="Failed to load model"):
                create_from_huggingface(LanguageModel, "test/model", temp_store)


class TestCreateFromLocalTorch:
    """Tests for create_from_local_torch function."""

    def test_create_from_local_torch_success(self, temp_store, tmp_path):
        """Test creating LanguageModel from local torch files."""
        # Create temporary paths that exist
        model_path = tmp_path / "model"
        tokenizer_path = tmp_path / "tokenizer"
        model_path.mkdir()
        tokenizer_path.mkdir()
        
        with patch("amber.language_model.initialization.AutoTokenizer") as mock_tokenizer_class, \
             patch("amber.language_model.initialization.AutoModelForCausalLM") as mock_model_class:
            
            mock_tokenizer = MagicMock()
            mock_model = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model
            
            lm = create_from_local_torch(
                LanguageModel,
                str(model_path),
                str(tokenizer_path),
                temp_store
            )
            
            assert lm.model == mock_model
            assert lm.tokenizer == mock_tokenizer
            mock_tokenizer_class.from_pretrained.assert_called_once()
            assert mock_tokenizer_class.from_pretrained.call_args[0][0] == str(tokenizer_path)
            mock_model_class.from_pretrained.assert_called_once()
            assert mock_model_class.from_pretrained.call_args[0][0] == str(model_path)

