"""Tests for LoadingStrategy enum."""

import pytest

from amber.datasets.loading_strategy import LoadingStrategy


def test_loading_strategy_enum_values():
    """Test that LoadingStrategy has correct enum values."""
    assert LoadingStrategy.MEMORY.value == "memory"
    assert LoadingStrategy.DYNAMIC_LOAD.value == "dynamic_load"
    assert LoadingStrategy.ITERABLE_ONLY.value == "iterable_only"


def test_loading_strategy_enum_membership():
    """Test LoadingStrategy enum membership."""
    assert LoadingStrategy.MEMORY in LoadingStrategy
    assert LoadingStrategy.DYNAMIC_LOAD in LoadingStrategy
    assert LoadingStrategy.ITERABLE_ONLY in LoadingStrategy


def test_loading_strategy_from_string():
    """Test creating LoadingStrategy from string value."""
    assert LoadingStrategy("memory") == LoadingStrategy.MEMORY
    assert LoadingStrategy("dynamic_load") == LoadingStrategy.DYNAMIC_LOAD
    assert LoadingStrategy("iterable_only") == LoadingStrategy.ITERABLE_ONLY


def test_loading_strategy_invalid_string():
    """Test that invalid string raises ValueError."""
    with pytest.raises(ValueError):
        LoadingStrategy("invalid")


def test_loading_strategy_list_all():
    """Test listing all loading strategies."""
    strategies = list(LoadingStrategy)
    assert len(strategies) == 3
    assert LoadingStrategy.MEMORY in strategies
    assert LoadingStrategy.DYNAMIC_LOAD in strategies
    assert LoadingStrategy.ITERABLE_ONLY in strategies

