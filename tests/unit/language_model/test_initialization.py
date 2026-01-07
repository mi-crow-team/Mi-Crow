"""Tests for LanguageModel initialization functions."""

import pytest
from unittest.mock import patch, MagicMock
from torch import nn

from mi_crow.language_model.initialization import initialize_model_id, create_from_huggingface, create_from_local_torch
from mi_crow.language_model.language_model import LanguageModel
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
        with patch("mi_crow.language_model.initialization.AutoTokenizer") as mock_tokenizer_class, \
             patch("mi_crow.language_model.initialization.AutoModelForCausalLM") as mock_model_class:
            
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
        with patch("mi_crow.language_model.initialization.AutoTokenizer") as mock_tokenizer_class, \
             patch("mi_crow.language_model.initialization.AutoModelForCausalLM") as mock_model_class:
            
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
        with patch("mi_crow.language_model.initialization.AutoTokenizer") as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.side_effect = Exception("Network error")
            
            with pytest.raises(RuntimeError, match="Failed to load model"):
                create_from_huggingface(LanguageModel, "test/model", temp_store)

    def test_create_from_huggingface_cuda_not_available_raises(self, temp_store, monkeypatch):
        """Requesting CUDA when it is not available should raise a clear ValueError."""
        model = nn.Linear(10, 5)
        
        with patch("mi_crow.language_model.initialization.AutoTokenizer") as mock_tokenizer_class, \
             patch("mi_crow.language_model.initialization.AutoModelForCausalLM") as mock_model_class, \
             patch("mi_crow.language_model.initialization.torch.cuda.is_available", return_value=False):
            
            mock_tokenizer = MagicMock()
            mock_model = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model
            
            msg = "Requested device 'cuda' but CUDA is not available"
            with pytest.raises(ValueError, match=msg):
                # We bypass create_from_huggingface's own cuda logic by directly
                # constructing a LanguageModel with device='cuda'.
                LanguageModel(model, mock_tokenizer, temp_store, device="cuda")


class TestCreateFromLocalTorch:
    """Tests for create_from_local_torch function."""

    def test_create_from_local_torch_success(self, temp_store, tmp_path):
        """Test creating LanguageModel from local torch files."""
        # Create temporary paths that exist
        model_path = tmp_path / "model"
        tokenizer_path = tmp_path / "tokenizer"
        model_path.mkdir()
        tokenizer_path.mkdir()
        
        with patch("mi_crow.language_model.initialization.AutoTokenizer") as mock_tokenizer_class, \
             patch("mi_crow.language_model.initialization.AutoModelForCausalLM") as mock_model_class:
            
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

    def test_create_from_local_torch_none_store_raises(self, tmp_path):
        model_path = tmp_path / "model"
        tokenizer_path = tmp_path / "tokenizer"
        model_path.mkdir()
        tokenizer_path.mkdir()
        with pytest.raises(ValueError):
            create_from_local_torch(LanguageModel, str(model_path), str(tokenizer_path), None)

    def test_create_from_local_torch_missing_paths_raise(self, temp_store, tmp_path):
        model_path = tmp_path / "model"
        tokenizer_path = tmp_path / "tokenizer"
        model_path.mkdir()
        with pytest.raises(FileNotFoundError):
            create_from_local_torch(LanguageModel, str(model_path), str(tokenizer_path), temp_store)

    def test_create_from_local_torch_load_failure(self, temp_store, tmp_path):
        model_path = tmp_path / "model"
        tokenizer_path = tmp_path / "tokenizer"
        model_path.mkdir()
        tokenizer_path.mkdir()
        with patch("mi_crow.language_model.initialization.AutoTokenizer") as mock_tokenizer_class:
            mock_tokenizer_class.from_pretrained.side_effect = Exception("boom")
            with pytest.raises(RuntimeError):
                create_from_local_torch(LanguageModel, str(model_path), str(tokenizer_path), temp_store)

    def test_create_from_local_torch_sets_device_map_for_cuda(self, temp_store, tmp_path):
        """create_from_local_torch should set device_map='auto' when device='cuda' and CUDA is available."""
        model_path = tmp_path / "model"
        tokenizer_path = tmp_path / "tokenizer"
        model_path.mkdir()
        tokenizer_path.mkdir()
        
        with patch("mi_crow.language_model.initialization.AutoTokenizer") as mock_tokenizer_class, \
             patch("mi_crow.language_model.initialization.AutoModelForCausalLM") as mock_model_class, \
             patch("mi_crow.language_model.initialization.torch.cuda.is_available", return_value=True):
            
            mock_tokenizer = MagicMock()
            mock_model = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model
            
            create_from_local_torch(
                LanguageModel,
                str(model_path),
                str(tokenizer_path),
                temp_store,
                device="cuda",
            )
            
            mock_model_class.from_pretrained.assert_called_once()
            _, kwargs = mock_model_class.from_pretrained.call_args
            assert kwargs.get("device_map") == "auto"

