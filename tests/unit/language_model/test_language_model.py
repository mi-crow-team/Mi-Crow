"""Tests for LanguageModel core class."""

import pytest
import torch

from mi_crow.language_model.language_model import LanguageModel
from tests.unit.fixtures.language_models import create_language_model_from_mock
from tests.unit.fixtures.stores import create_temp_store
from tests.unit.fixtures.hooks import create_mock_detector, create_mock_controller
from tests.unit.fixtures.models import create_mock_model
from tests.unit.fixtures.tokenizers import create_mock_tokenizer
from unittest.mock import MagicMock


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
        
        with patch.object(lm.lm_tokenizer, 'tokenize') as mock_tokenize:
            mock_tokenize.return_value = {"input_ids": [[1, 2, 3]]}
            result = lm.tokenize(["Hello"])
            
            assert result is not None
            mock_tokenize.assert_called_once()

    def test_tokenize_multiple_texts(self, temp_store):
        """Test tokenizing multiple texts."""
        from unittest.mock import patch
        lm = create_language_model_from_mock(temp_store)
        
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
        
        mock_output = MagicMock()
        mock_encodings = {"input_ids": torch.tensor([[1, 2, 3]])}
        with patch.object(lm.inference, 'execute_inference') as mock_execute:
            mock_execute.return_value = (mock_output, mock_encodings)
            output, encodings = lm.inference.execute_inference(["Hello"])
            
            assert output is not None
            assert encodings is not None
            mock_execute.assert_called_once()

    def test_inference_multiple_texts(self, temp_store):
        """Test inference with multiple texts."""
        from unittest.mock import patch, MagicMock
        import torch
        
        lm = create_language_model_from_mock(temp_store)
        
        mock_output = MagicMock()
        mock_encodings = {"input_ids": torch.tensor([[1, 2], [3, 4]])}
        with patch.object(lm.inference, 'execute_inference') as mock_execute:
            mock_execute.return_value = (mock_output, mock_encodings)
            output, encodings = lm.inference.execute_inference(["Hello", "World"])
            
            assert output is not None
            assert encodings is not None
            mock_execute.assert_called_once()


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


class TestLanguageModelMetadata:
    """Tests for metadata helpers."""

    def test_save_detector_metadata_requires_store(self, temp_store):
        lm = create_language_model_from_mock(temp_store)
        lm.context.store = None

        with pytest.raises(ValueError, match="Store must be provided"):
            lm.save_detector_metadata("run", 0)

    def test_save_detector_metadata_success(self, temp_store):
        lm = create_language_model_from_mock(temp_store)
        store = MagicMock()
        lm.context.store = store

        detector = create_mock_detector(layer_signature=1)
        detector.metadata = {"key": "value"}
        detector.tensor_metadata = {"activations": torch.ones(1)}
        lm.layers.register_hook(1, detector)

        lm.save_detector_metadata("run", 0)
        store.put_detector_metadata.assert_called_once()

    def test_save_detector_metadata_unified(self, temp_store):
        """Test save_detector_metadata with unified=True."""
        lm = create_language_model_from_mock(temp_store)
        store = MagicMock()
        lm.context.store = store

        detector = create_mock_detector(layer_signature=1)
        detector.metadata = {"key": "value"}
        detector.tensor_metadata = {"activations": torch.ones(1)}
        lm.layers.register_hook(1, detector)

        result = lm.save_detector_metadata("run", None, unified=True)
        store.put_run_detector_metadata.assert_called_once()
        store.put_detector_metadata.assert_not_called()

    def test_save_detector_metadata_batch_idx_required_when_not_unified(self, temp_store):
        """Test save_detector_metadata requires batch_idx when unified=False."""
        lm = create_language_model_from_mock(temp_store)
        store = MagicMock()
        lm.context.store = store

        with pytest.raises(ValueError, match="batch_idx must be provided when unified is False"):
            lm.save_detector_metadata("run", None, unified=False)

    def test_clear_detectors(self, temp_store):
        """Test clear_detectors clears all detector metadata."""
        from mi_crow.hooks.implementations.layer_activation_detector import LayerActivationDetector
        
        lm = create_language_model_from_mock(temp_store)
        detector = LayerActivationDetector(layer_signature=0)
        detector.metadata = {"key": "value"}
        detector.tensor_metadata = {"activations": torch.ones(2, 3)}
        lm.layers.register_hook(0, detector)

        lm.clear_detectors()

        assert len(detector.metadata) == 0
        assert len(detector.tensor_metadata) == 0

    def test_clear_detectors_calls_clear_captured(self, temp_store):
        """Test clear_detectors calls clear_captured if available."""
        from mi_crow.hooks.implementations.layer_activation_detector import LayerActivationDetector
        
        lm = create_language_model_from_mock(temp_store)
        detector = LayerActivationDetector(layer_signature=0)
        detector.tensor_metadata = {"activations": torch.ones(2, 3)}
        lm.layers.register_hook(0, detector)

        detector.process_activations(None, None, torch.ones(2, 3))
        assert detector.get_captured() is not None

        lm.clear_detectors()

        assert detector.get_captured() is None

    def test_save_model(self, temp_store):
        """Test save_model method."""
        from unittest.mock import patch
        lm = create_language_model_from_mock(temp_store)
        
        with patch('mi_crow.language_model.language_model.save_model') as mock_save:
            mock_save.return_value = temp_store.base_path / "model.pt"
            result = lm.save_model()
            
            mock_save.assert_called_once_with(lm, None)
            assert result == temp_store.base_path / "model.pt"

    def test_save_model_with_path(self, temp_store):
        """Test save_model with custom path."""
        from unittest.mock import patch
        lm = create_language_model_from_mock(temp_store)
        custom_path = temp_store.base_path / "custom" / "model.pt"
        
        with patch('mi_crow.language_model.language_model.save_model') as mock_save:
            mock_save.return_value = custom_path
            result = lm.save_model(custom_path)
            
            mock_save.assert_called_once_with(lm, custom_path)
            assert result == custom_path

    def test_from_local(self, temp_store):
        """Test from_local class method."""
        from unittest.mock import patch
        from pathlib import Path
        
        with patch('mi_crow.language_model.language_model.load_model_from_saved_file') as mock_load:
            mock_lm = create_language_model_from_mock(temp_store)
            mock_load.return_value = mock_lm
            
            result = LanguageModel.from_local("path/to/model.pt", temp_store)
            
            mock_load.assert_called_once_with(LanguageModel, "path/to/model.pt", temp_store, None)
            assert result == mock_lm


class TestLanguageModelModelId:
    """Tests for model ID extraction and initialization."""

    def test_model_id_from_config_name_or_path(self, temp_store):
        """Test model_id extracted from config.name_or_path with slash replacement."""
        from tests.unit.fixtures.models import create_mock_model
        from unittest.mock import MagicMock
        
        model = create_mock_model()
        if not hasattr(model, 'config'):
            model.config = MagicMock()
        model.config.name_or_path = "test/model"
        tokenizer = create_mock_tokenizer()
        
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store)
        
        assert lm.model_id == "test_model"

    def test_model_id_fallback_to_class_name(self, temp_store):
        """Test model_id falls back to class name when config.name_or_path missing."""
        from tests.unit.fixtures.models import create_mock_model
        
        model = create_mock_model()
        if hasattr(model, 'config') and hasattr(model.config, 'name_or_path'):
            delattr(model.config, 'name_or_path')
        tokenizer = create_mock_tokenizer()
        
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store)
        
        assert lm.model_id == model.__class__.__name__

    def test_model_id_provided_explicitly(self, temp_store):
        """Test model_id uses provided value when explicitly set."""
        from tests.unit.fixtures.models import create_mock_model
        
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, model_id="custom_id")
        
        assert lm.model_id == "custom_id"

    def test_from_huggingface_model_id_extraction(self, temp_store):
        """Test from_huggingface extracts model_id correctly."""
        from unittest.mock import patch, MagicMock
        
        with patch('mi_crow.language_model.initialization.AutoTokenizer') as mock_tokenizer, \
             patch('mi_crow.language_model.initialization.AutoModelForCausalLM') as mock_model:
            
            mock_tokenizer_instance = MagicMock()
            mock_model_instance = MagicMock()
            mock_model_instance.config.name_or_path = "huggingface/gpt2"
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            mock_model.from_pretrained.return_value = mock_model_instance
            
            lm = LanguageModel.from_huggingface("huggingface/gpt2", temp_store)
            
            assert lm.model_id == "huggingface_gpt2"


class TestLanguageModelInputTracker:
    """Tests for InputTracker singleton management."""

    def test_ensure_input_tracker_creates_singleton(self, temp_store):
        """Test _ensure_input_tracker creates singleton."""
        lm = create_language_model_from_mock(temp_store)
        
        tracker1 = lm._ensure_input_tracker()
        tracker2 = lm._ensure_input_tracker()
        
        assert tracker1 is tracker2
        assert tracker1 is lm._input_tracker

    def test_get_input_tracker_returns_none_when_not_created(self, temp_store):
        """Test get_input_tracker returns None when tracker not created."""
        lm = create_language_model_from_mock(temp_store)
        
        assert lm.get_input_tracker() is None

    def test_get_input_tracker_returns_tracker_after_creation(self, temp_store):
        """Test get_input_tracker returns tracker after creation."""
        lm = create_language_model_from_mock(temp_store)
        
        tracker = lm._ensure_input_tracker()
        retrieved = lm.get_input_tracker()
        
        assert retrieved is tracker


class TestLanguageModelInferenceControllerRestoration:
    """Tests for controller restoration during inference."""

    def test_controllers_restored_after_exception(self, temp_store):
        """Test controllers are restored even if exception occurs during inference."""
        from unittest.mock import patch, MagicMock
        
        lm = create_language_model_from_mock(temp_store)
        controller = create_mock_controller(layer_signature=0)
        controller.enable()
        lm.layers.register_hook(0, controller)
        
        with patch.object(lm.inference, '_run_model_forward', side_effect=RuntimeError("Test error")):
            with pytest.raises(RuntimeError):
                lm.inference.execute_inference(["test"])
        
        assert controller.enabled is True

    def test_controllers_disabled_when_with_controllers_false(self, temp_store):
        """Test controllers are temporarily disabled when with_controllers=False."""
        from unittest.mock import patch, MagicMock
        
        lm = create_language_model_from_mock(temp_store)
        controller = create_mock_controller(layer_signature=0)
        controller.enable()
        lm.layers.register_hook(0, controller)
        
        mock_output = MagicMock()
        mock_encodings = {"input_ids": torch.tensor([[1, 2, 3]])}
        
        with patch.object(lm.inference, 'execute_inference') as mock_execute:
            mock_execute.return_value = (mock_output, mock_encodings)
            
            lm.inference.execute_inference(["test"], with_controllers=False)
            
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            assert call_args[0][0] == ["test"]
            assert call_args[1]["with_controllers"] is False


class TestLanguageModelSpecialTokenExtraction:
    """Tests for special token ID extraction."""

    def test_extract_special_token_ids_from_tokenizer(self, temp_store):
        """Test special token IDs are extracted from tokenizer."""
        from tests.unit.fixtures.tokenizers import create_mock_tokenizer
        
        tokenizer = create_mock_tokenizer()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 3
        tokenizer.unk_token_id = 1
        
        model = create_mock_model()
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store)
        
        assert 0 in lm.context.special_token_ids
        assert 1 in lm.context.special_token_ids
        assert 2 in lm.context.special_token_ids
        assert 3 in lm.context.special_token_ids

    def test_extract_special_token_ids_handles_list_values(self, temp_store):
        """Test special token ID extraction handles list values."""
        from tests.unit.fixtures.tokenizers import create_mock_tokenizer
        
        tokenizer = create_mock_tokenizer()
        tokenizer.all_special_ids = None
        tokenizer.eos_token_id = [4, 2]
        
        model = create_mock_model()
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store)
        
        assert 4 in lm.context.special_token_ids
        assert 2 in lm.context.special_token_ids

    def test_extract_special_token_ids_from_all_special_ids(self, temp_store):
        """Test special token IDs extracted from all_special_ids attribute."""
        from tests.unit.fixtures.tokenizers import create_mock_tokenizer
        
        tokenizer = create_mock_tokenizer()
        tokenizer.all_special_ids = [10, 20, 30]
        
        model = create_mock_model()
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store)
        
        assert 10 in lm.context.special_token_ids
        assert 20 in lm.context.special_token_ids
        assert 30 in lm.context.special_token_ids

