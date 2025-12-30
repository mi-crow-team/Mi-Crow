"""Tests for LoadingStrategy enum."""

import pytest

from amber.datasets.loading_strategy import LoadingStrategy


def test_loading_strategy_enum_values():
    """Test that LoadingStrategy has correct enum values."""
    assert LoadingStrategy.MEMORY.value == "memory"
    assert LoadingStrategy.DISK.value == "dynamic_load"
    assert LoadingStrategy.STREAMING.value == "iterable_only"


def test_loading_strategy_enum_membership():
    """Test LoadingStrategy enum membership."""
    assert LoadingStrategy.MEMORY in LoadingStrategy
    assert LoadingStrategy.DISK in LoadingStrategy
    assert LoadingStrategy.STREAMING in LoadingStrategy


def test_loading_strategy_from_string():
    """Test creating LoadingStrategy from string value."""
    assert LoadingStrategy("memory") == LoadingStrategy.MEMORY
    assert LoadingStrategy("dynamic_load") == LoadingStrategy.DISK
    assert LoadingStrategy("iterable_only") == LoadingStrategy.STREAMING


def test_loading_strategy_invalid_string():
    """Test that invalid string raises ValueError."""
    with pytest.raises(ValueError):
        LoadingStrategy("invalid")


def test_loading_strategy_list_all():
    """Test listing all loading strategies."""
    strategies = list(LoadingStrategy)
    assert len(strategies) == 3
    assert LoadingStrategy.MEMORY in strategies
    assert LoadingStrategy.DISK in strategies
    assert LoadingStrategy.STREAMING in strategies

