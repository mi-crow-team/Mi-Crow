"""Integration tests for HuggingFace."""

import pytest
from unittest.mock import patch, MagicMock

from mi_crow.language_model.initialization import create_from_huggingface
from mi_crow.language_model.language_model import LanguageModel
from tests.unit.fixtures.stores import create_temp_store


class TestHuggingFaceIntegration:
    """Tests for HuggingFace integration."""

    @patch('mi_crow.language_model.initialization.AutoTokenizer')
    @patch('mi_crow.language_model.initialization.AutoModelForCausalLM')
    def test_model_loading_from_hub(self, mock_model_class, mock_tokenizer_class, temp_store):
        """Test loading model from HuggingFace Hub."""
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
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()

    @patch('mi_crow.language_model.initialization.AutoTokenizer')
    @patch('mi_crow.language_model.initialization.AutoModelForCausalLM')
    def test_dataset_loading_from_hub(self, mock_model_class, mock_tokenizer_class, temp_store):
        """Test loading dataset from HuggingFace Hub."""
        from mi_crow.datasets.text_dataset import TextDataset
        from unittest.mock import patch as mock_patch
        
        with mock_patch("mi_crow.datasets.text_dataset.load_dataset") as mock_load:
            from datasets import Dataset
            mock_ds = Dataset.from_dict({"text": ["a", "b"]})
            mock_load.return_value = mock_ds
            
            dataset = TextDataset.from_huggingface(
                "test/dataset",
                temp_store
            )
            
            assert len(dataset) == 2
            mock_load.assert_called_once()

    @patch('mi_crow.language_model.initialization.AutoTokenizer')
    def test_network_failure_handling(self, mock_tokenizer_class, temp_store):
        """Test error handling for network failures."""
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Network error")
        
        with pytest.raises(RuntimeError, match="Failed to load model"):
            create_from_huggingface(
                LanguageModel,
                "test/model",
                temp_store
            )

    @patch('mi_crow.language_model.initialization.AutoTokenizer')
    @patch('mi_crow.language_model.initialization.AutoModelForCausalLM')
    def test_invalid_model_id_handling(self, mock_model_class, mock_tokenizer_class, temp_store):
        """Test error handling for invalid model IDs."""
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Model not found")
        
        with pytest.raises(RuntimeError, match="Failed to load model"):
            create_from_huggingface(
                LanguageModel,
                "invalid/model",
                temp_store
            )

