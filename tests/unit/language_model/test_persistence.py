"""Tests for model persistence functions."""

import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from amber.language_model.persistence import save_model, load_model_from_saved_file
from amber.language_model.contracts import ModelMetadata
from tests.unit.fixtures import create_language_model, create_temp_store


class TestSaveModel:
    """Test suite for save_model."""

    def test_save_model_default_path(self, mock_language_model, temp_store):
        """Test saving model with default path."""
        mock_language_model.context.store = temp_store
        mock_language_model.context.model_id = "test_model"
        mock_language_model.context.model.state_dict = Mock(return_value={"layer.weight": torch.tensor([1.0])})
        mock_language_model.context._hook_registry = {}
        
        with patch("amber.language_model.persistence.collect_hooks_metadata", return_value={}):
            save_path = save_model(mock_language_model)
        
        assert save_path.exists()
        assert "test_model" in str(save_path)
        assert save_path.name == "model.pt"

    def test_save_model_custom_path_absolute(self, mock_language_model, temp_store, tmpdir):
        """Test saving model with custom absolute path."""
        mock_language_model.context.store = temp_store
        mock_language_model.context.model.state_dict = Mock(return_value={"layer.weight": torch.tensor([1.0])})
        mock_language_model.context._hook_registry = {}
        
        custom_path = Path(tmpdir) / "custom_model.pt"
        
        with patch("amber.language_model.persistence.collect_hooks_metadata", return_value={}):
            save_path = save_model(mock_language_model, path=custom_path)
        
        assert save_path == custom_path
        assert save_path.exists()

    def test_save_model_custom_path_relative(self, mock_language_model, temp_store):
        """Test saving model with custom relative path."""
        mock_language_model.context.store = temp_store
        mock_language_model.context.model.state_dict = Mock(return_value={"layer.weight": torch.tensor([1.0])})
        mock_language_model.context._hook_registry = {}
        
        with patch("amber.language_model.persistence.collect_hooks_metadata", return_value={}):
            save_path = save_model(mock_language_model, path="custom/model.pt")
        
        assert save_path.exists()
        assert str(save_path).endswith("custom/model.pt")

    def test_save_model_with_hooks(self, mock_language_model, temp_store):
        """Test saving model with hooks metadata."""
        mock_language_model.context.store = temp_store
        mock_language_model.context.model_id = "test_model"
        mock_language_model.context.model.state_dict = Mock(return_value={"layer.weight": torch.tensor([1.0])})
        mock_language_model.context._hook_registry = {"layer_0": {}}
        
        hooks_info = {"layer_0": [{"hook_id": "hook1"}]}
        
        with patch("amber.language_model.persistence.collect_hooks_metadata", return_value=hooks_info):
            save_path = save_model(mock_language_model)
        
        # Verify saved data contains hooks
        payload = torch.load(save_path, map_location="cpu")
        assert "metadata" in payload
        assert payload["metadata"]["hooks"] == hooks_info

    def test_save_model_store_none_raises_error(self, mock_language_model):
        """Test saving model when store is None raises error."""
        mock_language_model.context.store = None
        
        with pytest.raises(ValueError, match="Store must be provided"):
            save_model(mock_language_model)

    def test_save_model_oserror_handling(self, mock_language_model, temp_store):
        """Test saving model with OSError handling."""
        mock_language_model.context.store = temp_store
        mock_language_model.context.model.state_dict = Mock(return_value={"layer.weight": torch.tensor([1.0])})
        mock_language_model.context._hook_registry = {}
        
        with patch("amber.language_model.persistence.collect_hooks_metadata", return_value={}):
            with patch("torch.save", side_effect=OSError("disk full")):
                with pytest.raises(OSError, match="Failed to save model"):
                    save_model(mock_language_model)

    def test_save_model_creates_parent_directories(self, mock_language_model, temp_store):
        """Test that save_model creates parent directories."""
        mock_language_model.context.store = temp_store
        mock_language_model.context.model.state_dict = Mock(return_value={"layer.weight": torch.tensor([1.0])})
        mock_language_model.context._hook_registry = {}
        
        with patch("amber.language_model.persistence.collect_hooks_metadata", return_value={}):
            save_path = save_model(mock_language_model, path="deep/nested/path/model.pt")
        
        assert save_path.exists()
        assert save_path.parent.exists()


class TestLoadModelFromSavedFile:
    """Test suite for load_model_from_saved_file."""

    def test_load_model_success(self, mock_language_model, temp_store, tmpdir):
        """Test loading model successfully."""
        # Create a saved model file
        saved_path = Path(tmpdir) / "saved_model.pt"
        model_state_dict = {"layer.weight": torch.tensor([1.0])}
        metadata = {
            "model_id": "test_model",
            "hooks": {},
            "model_path": str(saved_path)
        }
        payload = {
            "model_state_dict": model_state_dict,
            "metadata": metadata
        }
        torch.save(payload, saved_path)
        
        from tests.unit.fixtures.models import SimpleLM
        
        with patch("amber.language_model.persistence.AutoTokenizer.from_pretrained") as mock_tokenizer:
            with patch("amber.language_model.persistence.AutoModelForCausalLM.from_pretrained") as mock_model:
                mock_tokenizer.return_value = Mock()
                mock_model_instance = SimpleLM()
                mock_model_instance.load_state_dict = Mock(return_value=None)
                mock_model.return_value = mock_model_instance
                
                lm = load_model_from_saved_file(
                    type(mock_language_model),
                    saved_path,
                    temp_store,
                    model_id=None
                )
        
        assert lm is not None
        assert lm.model_id == "test_model"
        mock_model_instance.load_state_dict.assert_called_once()

    def test_load_model_with_provided_model_id(self, mock_language_model, temp_store, tmpdir):
        """Test loading model with provided model_id."""
        saved_path = Path(tmpdir) / "saved_model.pt"
        payload = {
            "model_state_dict": {"layer.weight": torch.tensor([1.0])},
            "metadata": {"model_id": "saved_model", "hooks": {}, "model_path": str(saved_path)}
        }
        torch.save(payload, saved_path)
        
        from tests.unit.fixtures.models import SimpleLM
        
        with patch("amber.language_model.persistence.AutoTokenizer.from_pretrained") as mock_tokenizer:
            with patch("amber.language_model.persistence.AutoModelForCausalLM.from_pretrained") as mock_model:
                mock_tokenizer.return_value = Mock()
                mock_model_instance = SimpleLM()
                mock_model_instance.load_state_dict = Mock(return_value=None)
                mock_model.return_value = mock_model_instance
                
                lm = load_model_from_saved_file(
                    type(mock_language_model),
                    saved_path,
                    temp_store,
                    model_id="custom_model_id"
                )
        
        assert lm.model_id == "custom_model_id"

    def test_load_model_file_not_found(self, mock_language_model, temp_store):
        """Test loading model when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Saved model file not found"):
            load_model_from_saved_file(
                type(mock_language_model),
                Path("/nonexistent/path/model.pt"),
                temp_store
            )

    def test_load_model_store_none_raises_error(self, mock_language_model, tmpdir):
        """Test loading model when store is None."""
        saved_path = Path(tmpdir) / "saved_model.pt"
        payload = {
            "model_state_dict": {"layer.weight": torch.tensor([1.0])},
            "metadata": {"model_id": "test", "hooks": {}, "model_path": str(saved_path)}
        }
        torch.save(payload, saved_path)
        
        with pytest.raises(ValueError, match="store cannot be None"):
            load_model_from_saved_file(
                type(mock_language_model),
                saved_path,
                None
            )

    def test_load_model_invalid_format_missing_state_dict(self, mock_language_model, temp_store, tmpdir):
        """Test loading model with invalid format (missing state_dict)."""
        saved_path = Path(tmpdir) / "invalid_model.pt"
        payload = {"metadata": {"model_id": "test", "hooks": {}}}
        torch.save(payload, saved_path)
        
        with pytest.raises(ValueError, match="missing 'model_state_dict' key"):
            load_model_from_saved_file(
                type(mock_language_model),
                saved_path,
                temp_store
            )

    def test_load_model_invalid_format_missing_metadata(self, mock_language_model, temp_store, tmpdir):
        """Test loading model with invalid format (missing metadata)."""
        saved_path = Path(tmpdir) / "invalid_model.pt"
        payload = {"model_state_dict": {"layer.weight": torch.tensor([1.0])}}
        torch.save(payload, saved_path)
        
        with pytest.raises(ValueError, match="missing 'metadata' key"):
            load_model_from_saved_file(
                type(mock_language_model),
                saved_path,
                temp_store
            )

    def test_load_model_no_model_id_in_metadata(self, mock_language_model, temp_store, tmpdir):
        """Test loading model when model_id is not in metadata and not provided."""
        saved_path = Path(tmpdir) / "saved_model.pt"
        payload = {
            "model_state_dict": {"layer.weight": torch.tensor([1.0])},
            "metadata": {"hooks": {}, "model_path": str(saved_path)}
        }
        torch.save(payload, saved_path)
        
        with pytest.raises(ValueError, match="model_id not found in saved metadata"):
            load_model_from_saved_file(
                type(mock_language_model),
                saved_path,
                temp_store,
                model_id=None
            )

    def test_load_model_huggingface_load_failure(self, mock_language_model, temp_store, tmpdir):
        """Test loading model when HuggingFace load fails."""
        saved_path = Path(tmpdir) / "saved_model.pt"
        payload = {
            "model_state_dict": {"layer.weight": torch.tensor([1.0])},
            "metadata": {"model_id": "nonexistent/model", "hooks": {}, "model_path": str(saved_path)}
        }
        torch.save(payload, saved_path)
        
        with patch("amber.language_model.persistence.AutoTokenizer.from_pretrained", side_effect=Exception("not found")):
            with pytest.raises(ValueError, match="Failed to load model"):
                load_model_from_saved_file(
                    type(mock_language_model),
                    saved_path,
                    temp_store
                )

    def test_load_model_state_dict_load_failure(self, mock_language_model, temp_store, tmpdir):
        """Test loading model when state dict load fails."""
        saved_path = Path(tmpdir) / "saved_model.pt"
        payload = {
            "model_state_dict": {"layer.weight": torch.tensor([1.0])},
            "metadata": {"model_id": "test_model", "hooks": {}, "model_path": str(saved_path)}
        }
        torch.save(payload, saved_path)
        
        with patch("amber.language_model.persistence.AutoTokenizer.from_pretrained") as mock_tokenizer:
            with patch("amber.language_model.persistence.AutoModelForCausalLM.from_pretrained") as mock_model:
                mock_tokenizer.return_value = Mock()
                mock_model_instance = Mock()
                mock_model_instance.load_state_dict = Mock(side_effect=RuntimeError("size mismatch"))
                mock_model.return_value = mock_model_instance
                
                with pytest.raises(RuntimeError, match="Failed to load state dict"):
                    load_model_from_saved_file(
                        type(mock_language_model),
                        saved_path,
                        temp_store
                    )

    def test_load_model_torch_load_failure(self, mock_language_model, temp_store, tmpdir):
        """Test loading model when torch.load fails."""
        saved_path = Path(tmpdir) / "corrupted_model.pt"
        # Create a file that can't be loaded
        saved_path.write_text("not a valid pytorch file")
        
        with pytest.raises(RuntimeError, match="Failed to load model file"):
            load_model_from_saved_file(
                type(mock_language_model),
                saved_path,
                temp_store
            )

