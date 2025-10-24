"""Test the amber package initialization and basic functionality."""

import pytest
from amber import PACKAGE_NAME, ping


def test_package_name():
    """Test that the package name is correctly defined."""
    assert PACKAGE_NAME == "amber"


def test_ping_function():
    """Test the ping function returns expected value."""
    result = ping()
    assert result == "pong"
    assert isinstance(result, str)


def test_package_import():
    """Test that the package can be imported without side effects."""
    import amber
    assert hasattr(amber, 'PACKAGE_NAME')
    assert hasattr(amber, 'ping')
    assert amber.PACKAGE_NAME == "amber"
    assert amber.ping() == "pong"


def test_ping_function_type():
    """Test that ping function has correct type annotation."""
    import inspect
    sig = inspect.signature(ping)
    assert sig.return_annotation == str
