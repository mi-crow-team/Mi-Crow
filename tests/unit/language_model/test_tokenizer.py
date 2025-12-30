"""Tests for LanguageModelTokenizer."""

import pytest

from mi_crow.language_model.tokenizer import LanguageModelTokenizer
from tests.unit.fixtures.language_models import create_language_model_from_mock
from tests.unit.fixtures.stores import create_temp_store


class TestLanguageModelTokenizer:
    """Tests for LanguageModelTokenizer."""

    def test_tokenizer_initialization(self, temp_store):
        """Test tokenizer initialization."""
        lm = create_language_model_from_mock(temp_store)
        tokenizer = lm.lm_tokenizer
        
        assert tokenizer.context == lm.context

    def test_tokenize_single_text(self, temp_store):
        """Test tokenizing single text."""
        from unittest.mock import patch
        lm = create_language_model_from_mock(temp_store)
        
        with patch.object(lm.lm_tokenizer, 'tokenize') as mock_tokenize:
            mock_tokenize.return_value = {"input_ids": [[1, 2, 3]]}
            result = lm.tokenize(["Hello world"])
            
            assert result is not None
            mock_tokenize.assert_called_once()

    def test_tokenize_single_text(self, temp_store):
        """Test tokenizing single text."""
        from unittest.mock import patch
        lm = create_language_model_from_mock(temp_store)
        
        with patch.object(lm.lm_tokenizer, 'tokenize') as mock_tokenize:
            mock_tokenize.return_value = {"input_ids": [[1, 2, 3]]}
            result = lm.tokenize(["Hello world"])
            
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

    def test_split_to_tokens_single(self, temp_store):
        """Test splitting single text to tokens."""
        lm = create_language_model_from_mock(temp_store)
        tokens = lm.lm_tokenizer.split_to_tokens("Hello world")
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_split_to_tokens_multiple(self, temp_store):
        """Test splitting multiple texts to tokens."""
        lm = create_language_model_from_mock(temp_store)
        tokens = lm.lm_tokenizer.split_to_tokens(["Hello", "World"])
        
        assert isinstance(tokens, list)
        assert len(tokens) == 2
        assert isinstance(tokens[0], list)

