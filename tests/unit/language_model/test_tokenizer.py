"""Tests for LanguageModelTokenizer."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from mi_crow.language_model.context import LanguageModelContext
from mi_crow.language_model.tokenizer import LanguageModelTokenizer
from tests.unit.fixtures import create_mock_tokenizer
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
        with patch.object(lm.lm_tokenizer, "tokenize") as mock_tokenize:
            mock_tokenize.return_value = {"input_ids": [[1, 2, 3]]}
            result = lm.tokenize(["Hello world"])
            assert result is not None
            mock_tokenize.assert_called_once()

    def test_tokenize_multiple_texts(self, temp_store):
        """Test tokenizing multiple texts."""
        from unittest.mock import patch

        lm = create_language_model_from_mock(temp_store)
        with patch.object(lm.lm_tokenizer, "tokenize") as mock_tokenize:
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

    def test_setup_pad_token_with_eos_token(self, mock_language_model):
        """Test setting up pad token when eos_token exists."""
        tokenizer = create_mock_tokenizer()
        tokenizer.eos_token = "<eos>"
        tokenizer.eos_token_id = 2
        tokenizer.pad_token = None
        model = Mock()
        model.config = Mock()
        model.config.pad_token_id = None
        mock_language_model.context.tokenizer = tokenizer
        mock_language_model.context.model = model
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        lm_tokenizer._setup_pad_token(tokenizer, model)
        assert tokenizer.pad_token == "<eos>"
        assert model.config.pad_token_id == 2

    def test_setup_pad_token_without_eos_token(self, mock_language_model):
        """Test setting up pad token when eos_token doesn't exist."""
        tokenizer = create_mock_tokenizer()
        tokenizer.eos_token = None
        tokenizer.add_special_tokens = Mock()
        tokenizer.pad_token_id = 3
        tokenizer.__len__ = Mock(return_value=1000)
        model = Mock()
        model.config = Mock()
        model.config.pad_token_id = None
        model.resize_token_embeddings = Mock(return_value=None)
        mock_language_model.context.tokenizer = tokenizer
        mock_language_model.context.model = model
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        lm_tokenizer._setup_pad_token(tokenizer, model)
        tokenizer.add_special_tokens.assert_called_once_with({"pad_token": "[PAD]"})
        model.resize_token_embeddings.assert_called_once_with(len(tokenizer))
        assert model.config.pad_token_id == 3

    def test_setup_pad_token_no_model_config(self, mock_language_model):
        """Test setting up pad token when model has no config."""
        tokenizer = create_mock_tokenizer()
        tokenizer.eos_token = "<eos>"
        tokenizer.eos_token_id = 2
        model = Mock()
        del model.config
        mock_language_model.context.tokenizer = tokenizer
        mock_language_model.context.model = model
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        lm_tokenizer._setup_pad_token(tokenizer, model)

    def test_try_tokenize_with_method_tokenize(self, mock_language_model):
        """Test tokenizing with tokenize method."""
        tokenizer = create_mock_tokenizer()
        tokenizer.tokenize = Mock(return_value=["hello", "world"])
        mock_language_model.context.tokenizer = tokenizer
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        result = lm_tokenizer._try_tokenize_with_method(
            tokenizer, "hello world", add_special_tokens=False, method_name="tokenize"
        )
        assert result == ["hello", "world"]
        tokenizer.tokenize.assert_called_once_with("hello world", add_special_tokens=False)

    def test_try_tokenize_with_method_encode(self, mock_language_model):
        """Test tokenizing with encode method."""
        tokenizer = create_mock_tokenizer()
        tokenizer.encode = Mock(return_value=[1, 2, 3])
        tokenizer.convert_ids_to_tokens = Mock(return_value=["hello", "world", "!"])
        mock_language_model.context.tokenizer = tokenizer
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        result = lm_tokenizer._try_tokenize_with_method(
            tokenizer,
            "hello world!",
            add_special_tokens=False,
            method_name="encode",
            fallback_method="convert_ids_to_tokens",
        )
        assert result == ["hello", "world", "!"]
        tokenizer.encode.assert_called_once_with("hello world!", add_special_tokens=False)
        tokenizer.convert_ids_to_tokens.assert_called_once_with([1, 2, 3])

    def test_try_tokenize_with_method_encode_plus(self, mock_language_model):
        """Test tokenizing with encode_plus method."""
        tokenizer = create_mock_tokenizer()
        tokenizer.encode_plus = Mock(return_value={"input_ids": [1, 2, 3]})
        tokenizer.convert_ids_to_tokens = Mock(return_value=["hello", "world", "!"])
        mock_language_model.context.tokenizer = tokenizer
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        result = lm_tokenizer._try_tokenize_with_method(
            tokenizer,
            "hello world!",
            add_special_tokens=False,
            method_name="encode_plus",
            fallback_method="convert_ids_to_tokens",
        )
        assert result == ["hello", "world", "!"]
        tokenizer.encode_plus.assert_called_once_with("hello world!", add_special_tokens=False)

    def test_try_tokenize_with_method_no_method(self, mock_language_model):
        """Test tokenizing when method doesn't exist."""
        tokenizer = object()
        mock_language_model.context.tokenizer = create_mock_tokenizer()
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        result = lm_tokenizer._try_tokenize_with_method(
            tokenizer, "hello world", add_special_tokens=False, method_name="tokenize"
        )
        assert result is None

    def test_try_tokenize_with_method_exception(self, mock_language_model):
        """Test tokenizing when method raises exception."""
        tokenizer = create_mock_tokenizer()
        tokenizer.tokenize = Mock(side_effect=ValueError("error"))
        mock_language_model.context.tokenizer = tokenizer
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        result = lm_tokenizer._try_tokenize_with_method(
            tokenizer, "hello world", add_special_tokens=False, method_name="tokenize"
        )
        assert result is None

    def test_split_single_text_to_tokens_no_tokenizer(self, mock_language_model):
        """Test splitting text when tokenizer is None."""
        mock_language_model.context.tokenizer = None
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        result = lm_tokenizer._split_single_text_to_tokens("hello world", add_special_tokens=False)
        assert result == ["hello", "world"]

    def test_split_single_text_to_tokens_non_string(self, mock_language_model):
        """Test splitting non-string text."""
        tokenizer = create_mock_tokenizer()
        mock_language_model.context.tokenizer = tokenizer
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        with pytest.raises(TypeError, match="Expected str"):
            lm_tokenizer._split_single_text_to_tokens(123, add_special_tokens=False)

    def test_split_single_text_to_tokens_fallback(self, mock_language_model):
        """Test splitting text with fallback to split."""
        tokenizer = create_mock_tokenizer()
        tokenizer.tokenize = None
        tokenizer.encode = None
        tokenizer.encode_plus = None
        mock_language_model.context.tokenizer = tokenizer
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        result = lm_tokenizer._split_single_text_to_tokens("hello world", add_special_tokens=False)
        assert result == ["hello", "world"]

    def test_tokenize_callable_tokenizer(self, mock_language_model):
        """Test tokenizing with callable tokenizer."""

        class CallableTokenizer:
            def __init__(self):
                self.pad_token = "<pad>"
                self.calls = []

            def __call__(self, texts, **kwargs):
                self.calls.append((texts, kwargs))
                return {"input_ids": [[1, 2, 3]]}

        tokenizer = CallableTokenizer()
        mock_language_model.context.tokenizer = tokenizer
        mock_language_model.context.model = Mock()
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        result = lm_tokenizer.tokenize(["hello", "world"], padding=True)
        assert "input_ids" in result
        assert len(tokenizer.calls) == 1

    def test_tokenize_callable_tokenizer_typeerror(self, mock_language_model):
        """Test tokenizing with callable tokenizer that raises TypeError."""
        tokenizer = create_mock_tokenizer()
        tokenizer.__call__ = Mock(side_effect=TypeError("error"))
        tokenizer.batch_encode_plus = Mock(return_value={"input_ids": [[1, 2, 3]]})
        mock_language_model.context.tokenizer = tokenizer
        mock_language_model.context.model = Mock()
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        result = lm_tokenizer.tokenize(["hello", "world"], padding=True)
        assert "input_ids" in result
        tokenizer.batch_encode_plus.assert_called_once()

    def test_tokenize_batch_encode_plus(self, mock_language_model):
        """Test tokenizing with batch_encode_plus."""
        tokenizer = create_mock_tokenizer()
        tokenizer.__call__ = None
        tokenizer.batch_encode_plus = Mock(return_value={"input_ids": [[1, 2, 3]]})
        mock_language_model.context.tokenizer = tokenizer
        mock_language_model.context.model = Mock()
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        result = lm_tokenizer.tokenize(["hello", "world"], padding=True)
        assert "input_ids" in result
        tokenizer.batch_encode_plus.assert_called_once()

    def test_tokenize_encode_plus_fallback(self, mock_language_model):
        """Test tokenizing with encode_plus fallback."""

        class EncodePlusTokenizer:
            def __init__(self):
                self.pad_token = "<pad>"
                self.encode_plus = Mock(return_value={"input_ids": [1, 2, 3]})
                self.pad = Mock(return_value={"input_ids": [[1, 2, 3]]})

            def __call__(self, *args, **kwargs):
                raise TypeError("force fallback")

        tokenizer = EncodePlusTokenizer()
        mock_language_model.context.tokenizer = tokenizer
        mock_language_model.context.model = Mock()
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        result = lm_tokenizer.tokenize(["hello", "world"], padding=True, return_tensors="pt")
        assert "input_ids" in result
        assert tokenizer.encode_plus.call_count == 2
        tokenizer.pad.assert_called_once()

    def test_tokenize_encode_plus_no_pad(self, mock_language_model):
        """Test tokenizing with encode_plus but no pad method."""

        class EncodePlusNoPadTokenizer:
            def __init__(self):
                self.pad_token = "<pad>"
                self.encode_plus = Mock(return_value={"input_ids": [1, 2, 3]})

            def __call__(self, *args, **kwargs):
                raise TypeError("force fallback")

        tokenizer = EncodePlusNoPadTokenizer()
        mock_language_model.context.tokenizer = tokenizer
        mock_language_model.context.model = Mock()
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        result = lm_tokenizer.tokenize(["hello", "world"], padding=True)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_tokenize_no_tokenizer_raises_error(self, mock_language_model):
        """Test tokenizing when tokenizer is None raises error."""
        mock_language_model.context.tokenizer = None
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        with pytest.raises(ValueError, match="Tokenizer must be initialized"):
            lm_tokenizer.tokenize(["hello", "world"])

    def test_tokenize_no_usable_method_raises_error(self, mock_language_model):
        """Test tokenizing when no usable method exists."""

        class NoMethodTokenizer:
            def __init__(self):
                self.pad_token = "<pad>"

            def __call__(self, *args, **kwargs):
                raise TypeError("force fallback")

        tokenizer = NoMethodTokenizer()
        mock_language_model.context.tokenizer = tokenizer
        mock_language_model.context.model = Mock()
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        with pytest.raises(TypeError, match="not usable for batch tokenization"):
            lm_tokenizer.tokenize(["hello", "world"])

    def test_tokenize_setup_pad_token_when_needed(self, mock_language_model):
        """Test that pad token is set up when needed."""
        tokenizer = create_mock_tokenizer()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<eos>"
        tokenizer.eos_token_id = 2
        tokenizer.__call__ = Mock(return_value={"input_ids": [[1, 2, 3]]})
        model = Mock()
        model.config = Mock()
        model.config.pad_token_id = None
        mock_language_model.context.tokenizer = tokenizer
        mock_language_model.context.model = model
        lm_tokenizer = LanguageModelTokenizer(mock_language_model.context)
        lm_tokenizer.tokenize(["hello"], padding=True)
        assert tokenizer.pad_token == "<eos>"
