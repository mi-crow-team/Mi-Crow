
def pytest_addoption(parser):
    parser.addoption(
        "--unit",
        action="store_true",
        default=False,
        help="Run only unit tests (from tests/unit/)",
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run only integration tests (from tests/integration/)",
    )
    parser.addoption(
        "--e2e",
        action="store_true",
        default=False,
        help="Run only end-to-end tests (from tests/e2e/)",
    )

import pytest
from pathlib import Path

from amber.datasets.loading_strategy import LoadingStrategy
from tests.unit.fixtures.stores import create_temp_store, create_mock_store
from tests.unit.fixtures.models import create_mock_model
from tests.unit.fixtures.tokenizers import create_mock_tokenizer
from tests.unit.fixtures.datasets import create_sample_dataset
from tests.unit.fixtures.language_models import create_language_model_from_mock


def pytest_collection_modifyitems(config, items):
    if config.getoption("--unit"):
        selected = []
        deselected = []
        for item in items:
            # item.path is already a Path object
            # Convert to string and normalize separators for checking
            path_str = str(item.path).replace("\\", "/")
            if "/tests/unit/" in path_str:
                selected.append(item)
            else:
                deselected.append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)
    elif config.getoption("--integration"):
        selected = []
        deselected = []
        for item in items:
            # item.path is already a Path object
            # Convert to string and normalize separators for checking
            path_str = str(item.path).replace("\\", "/")
            if "/tests/integration/" in path_str:
                selected.append(item)
            else:
                deselected.append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)
    elif config.getoption("--e2e"):
        selected = []
        deselected = []
        for item in items:
            # item.path is already a Path object
            # Convert to string and normalize separators for checking
            path_str = str(item.path).replace("\\", "/")
            if "/tests/e2e/" in path_str:
                selected.append(item)
            else:
                deselected.append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)


# Store fixtures
@pytest.fixture
def temp_store(tmp_path):
    """Create a temporary LocalStore."""
    return create_temp_store(tmp_path)


@pytest.fixture
def mock_store():
    """Create a mock Store."""
    return create_mock_store()


# Model fixtures
@pytest.fixture
def mock_model():
    """Create a mock PyTorch model."""
    return create_mock_model()


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    return create_mock_tokenizer()


# Language Model fixtures
@pytest.fixture
def mock_language_model(temp_store):
    """Create a LanguageModel with mock model and tokenizer."""
    return create_language_model_from_mock(temp_store)


# Dataset fixtures
@pytest.fixture
def sample_dataset():
    """Create a sample dataset."""
    return create_sample_dataset()


@pytest.fixture
def sample_classification_dataset():
    """Create a sample classification dataset."""
    return create_sample_dataset(include_categories=True)


# Parametrized fixtures for loading strategies
@pytest.fixture(params=[LoadingStrategy.MEMORY, LoadingStrategy.DYNAMIC_LOAD, LoadingStrategy.ITERABLE_ONLY])
def loading_strategy(request):
    """Parametrized fixture for all loading strategies."""
    return request.param


@pytest.fixture(params=[LoadingStrategy.MEMORY, LoadingStrategy.DYNAMIC_LOAD])
def non_streaming_strategy(request):
    """Parametrized fixture for non-streaming loading strategies."""
    return request.param

