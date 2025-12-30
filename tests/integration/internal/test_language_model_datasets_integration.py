"""Integration tests for LanguageModel with Datasets."""

import pytest

from tests.unit.fixtures.language_models import create_language_model_from_mock
from tests.unit.fixtures.stores import create_temp_store
from tests.unit.fixtures.datasets import create_text_dataset


class TestLanguageModelDatasetsIntegration:
    """Tests for LanguageModel with Datasets."""

    def test_language_model_with_text_dataset(self, temp_store):
        """Test LanguageModel with TextDataset."""
        lm = create_language_model_from_mock(temp_store)
        dataset = create_text_dataset(temp_store, texts=["Hello", "World"])
        
        # Process texts from dataset
        texts = list(dataset.iter_items())
        output, encodings = lm.forwards(texts)
        
        assert output is not None
        assert encodings is not None

    def test_batch_processing_from_dataset(self, temp_store):
        """Test batch processing from dataset."""
        lm = create_language_model_from_mock(temp_store)
        dataset = create_text_dataset(
            temp_store,
            texts=["Text 1", "Text 2", "Text 3", "Text 4"]
        )
        
        # Process in batches
        for batch in dataset.iter_batches(batch_size=2):
            output, encodings = lm.forwards(batch)
            assert output is not None

    def test_streaming_dataset_with_inference(self, temp_store):
        """Test streaming dataset with inference."""
        from amber.datasets.loading_strategy import LoadingStrategy
        
        lm = create_language_model_from_mock(temp_store)
        dataset = create_text_dataset(
            temp_store,
            texts=["Text 1", "Text 2"],
            loading_strategy=LoadingStrategy.STREAMING
        )
        
        # Process streaming dataset
        for text in dataset.iter_items():
            output, encodings = lm.forwards([text])
            assert output is not None

