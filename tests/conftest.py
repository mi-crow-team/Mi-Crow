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
from mi_crow.datasets.loading_strategy import LoadingStrategy
from tests.unit.fixtures.datasets import create_sample_dataset
from tests.unit.fixtures.language_models import create_language_model_from_mock
from tests.unit.fixtures.models import create_mock_model
from tests.unit.fixtures.stores import create_mock_store, create_temp_store
from tests.unit.fixtures.tokenizers import create_mock_tokenizer


def pytest_collection_modifyitems(config, items):
    """
    Automatically add markers based on test file location and support --unit/--integration/--e2e flags.
    Uses marker-based selection when markers are present, falls back to path-based for backward compatibility.
    """
    # Auto-add markers based on file path if not already marked
    for item in items:
        path_str = str(item.path).replace("\\", "/")
        # Only add marker if test doesn't already have one
        if not any(mark.name in ("unit", "integration", "e2e") for mark in item.iter_markers()):
            if "/tests/unit/" in path_str:
                item.add_marker(pytest.mark.unit)
            elif "/tests/integration/" in path_str:
                item.add_marker(pytest.mark.integration)
            elif "/tests/e2e/" in path_str:
                item.add_marker(pytest.mark.e2e)

    # Support --unit, --integration, --e2e flags (backward compatibility)
    # Prefer marker-based selection if markers are present
    if config.getoption("--unit"):
        selected = []
        deselected = []
        for item in items:
            # Check for marker first, then fall back to path
            has_unit_marker = any(mark.name == "unit" for mark in item.iter_markers())
            path_str = str(item.path).replace("\\", "/")
            is_unit_path = "/tests/unit/" in path_str
            if has_unit_marker or is_unit_path:
                selected.append(item)
            else:
                deselected.append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)
    elif config.getoption("--integration"):
        selected = []
        deselected = []
        for item in items:
            has_integration_marker = any(mark.name == "integration" for mark in item.iter_markers())
            path_str = str(item.path).replace("\\", "/")
            is_integration_path = "/tests/integration/" in path_str
            if has_integration_marker or is_integration_path:
                selected.append(item)
            else:
                deselected.append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)
    elif config.getoption("--e2e"):
        selected = []
        deselected = []
        for item in items:
            has_e2e_marker = any(mark.name == "e2e" for mark in item.iter_markers())
            path_str = str(item.path).replace("\\", "/")
            is_e2e_path = "/tests/e2e/" in path_str
            if has_e2e_marker or is_e2e_path:
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
@pytest.fixture(params=[LoadingStrategy.MEMORY, LoadingStrategy.DISK, LoadingStrategy.STREAMING])
def loading_strategy(request):
    """Parametrized fixture for all loading strategies."""
    return request.param


@pytest.fixture(params=[LoadingStrategy.MEMORY, LoadingStrategy.DISK])
def non_streaming_strategy(request):
    """Parametrized fixture for non-streaming loading strategies."""
    return request.param
