"""Tests for LanguageModelContext."""

import pytest

from mi_crow.language_model.context import LanguageModelContext
from tests.unit.fixtures.language_models import create_language_model_from_mock
from tests.unit.fixtures.stores import create_temp_store


class TestLanguageModelContext:
    """Tests for LanguageModelContext."""

    def test_context_initialization(self, temp_store):
        """Test context initialization."""
        lm = create_language_model_from_mock(temp_store)
        context = lm.context
        assert context.language_model == lm
        assert context.model is not None
        assert context.tokenizer is not None
        assert context.store == temp_store
        assert context.model_id is not None

    def test_context_default_values(self, temp_store):
        """Test context default values."""
        lm = create_language_model_from_mock(temp_store)
        context = lm.context
        assert context.device == "cpu"
        assert context.dtype is None
        assert context.tokenizer_params is None
        assert context.model_params is None
        assert context._hook_registry == {}
        assert context._hook_id_map == {}

    def test_context_hook_registry(self, temp_store):
        """Test hook registry in context."""
        lm = create_language_model_from_mock(temp_store)
        context = lm.context
        assert context._hook_registry == {}
        assert context._hook_id_map == {}
