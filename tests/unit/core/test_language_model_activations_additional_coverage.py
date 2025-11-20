import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from amber.language_model.language_model_activations import LanguageModelActivations
from amber.language_model.language_model_context import LanguageModelContext
from amber.store.local_store import LocalStore
from amber.adapters.text_dataset import TextDataset
from datasets import Dataset


@pytest.fixture
def mock_context(tmp_path):
    """Create a mock context for testing."""
    store = LocalStore(tmp_path / "store")
    mock_model = Mock()
    mock_model.model_name = "test_model"
    mock_model.__class__.__name__ = "TestModel"
    context = LanguageModelContext(
        model=mock_model,
        tokenizer=Mock(),
        store=store,
        device=torch.device("cpu"),
        language_model=Mock()
    )
    context.language_model._get_device = Mock(return_value=torch.device("cpu"))
    context.language_model._inference = Mock()
    context.language_model.layers = Mock()
    context.language_model.save_detector_metadata = Mock()
    return context


@pytest.fixture
def activations(mock_context):
    """Create LanguageModelActivations instance."""
    return LanguageModelActivations(context=mock_context)


def test_save_activations_dataset_with_dtype_conversion(activations, tmp_path):
    """Test save_activations_dataset with dtype conversion."""
    store = LocalStore(tmp_path / "store")
    activations.context.store = store
    
    # Create a simple dataset
    ds = Dataset.from_dict({"text": ["Hello", "World"]})
    text_ds = TextDataset(ds, store)
    
    # Mock detector with activations
    mock_detector = Mock()
    mock_detector.tensor_metadata = {"activations": torch.randn(2, 3, dtype=torch.float32)}
    activations.context.language_model.layers.get_detectors = Mock(return_value=[mock_detector])
    
    # Mock the setup and cleanup
    activations._setup_detector = Mock(return_value=(Mock(), "hook_id"))
    activations._cleanup_detector = Mock()
    
    # Run with dtype conversion
    run_name = activations.save_activations_dataset(
        text_ds,
        layer_signature="test_layer",
        batch_size=1,
        dtype=torch.float16
    )
    
    # Verify dtype conversion was applied
    assert mock_detector.tensor_metadata["activations"].dtype == torch.float16


def test_save_activations_dataset_with_max_length(activations, tmp_path):
    """Test save_activations_dataset with max_length parameter."""
    store = LocalStore(tmp_path / "store")
    activations.context.store = store
    
    ds = Dataset.from_dict({"text": ["Hello", "World"]})
    text_ds = TextDataset(ds, store)
    
    activations._setup_detector = Mock(return_value=(Mock(), "hook_id"))
    activations._cleanup_detector = Mock()
    
    run_name = activations.save_activations_dataset(
        text_ds,
        layer_signature="test_layer",
        batch_size=1,
        max_length=128
    )
    
    # Verify max_length was passed to _inference
    calls = activations.context.language_model._inference.call_args_list
    assert len(calls) > 0
    # Check that tok_kwargs contains max_length
    for call in calls:
        kwargs = call.kwargs
        if "tok_kwargs" in kwargs:
            assert kwargs["tok_kwargs"].get("max_length") == 128


def test_save_activations_dataset_with_empty_batches(activations, tmp_path):
    """Test save_activations_dataset handles empty batches."""
    store = LocalStore(tmp_path / "store")
    activations.context.store = store
    
    ds = Dataset.from_dict({"text": ["Hello"]})
    text_ds = TextDataset(ds, store)
    
    activations._setup_detector = Mock(return_value=(Mock(), "hook_id"))
    activations._cleanup_detector = Mock()
    
    # Mock iter_batches to return empty batch
    def mock_iter_batches(size):
        yield []  # Empty batch
        yield ["Hello"]  # Non-empty batch
    
    text_ds.iter_batches = mock_iter_batches
    
    run_name = activations.save_activations_dataset(
        text_ds,
        layer_signature="test_layer",
        batch_size=1
    )
    
    # Should skip empty batch and process non-empty one
    assert activations.context.language_model._inference.call_count == 1


def test_save_activations_dataset_with_autocast(activations, tmp_path):
    """Test save_activations_dataset with autocast."""
    store = LocalStore(tmp_path / "store")
    activations.context.store = store
    
    ds = Dataset.from_dict({"text": ["Hello"]})
    text_ds = TextDataset(ds, store)
    
    activations._setup_detector = Mock(return_value=(Mock(), "hook_id"))
    activations._cleanup_detector = Mock()
    
    run_name = activations.save_activations_dataset(
        text_ds,
        layer_signature="test_layer",
        batch_size=1,
        autocast=True,
        autocast_dtype=torch.float16
    )
    
    # Verify autocast parameters were passed
    calls = activations.context.language_model._inference.call_args_list
    assert len(calls) > 0
    for call in calls:
        kwargs = call.kwargs
        assert kwargs.get("autocast") is True
        assert kwargs.get("autocast_dtype") == torch.float16


def test_save_activations_dataset_with_verbose(activations, tmp_path, caplog):
    """Test save_activations_dataset with verbose logging."""
    store = LocalStore(tmp_path / "store")
    activations.context.store = store
    
    ds = Dataset.from_dict({"text": ["Hello"]})
    text_ds = TextDataset(ds, store)
    
    activations._setup_detector = Mock(return_value=(Mock(), "hook_id"))
    activations._cleanup_detector = Mock()
    
    run_name = activations.save_activations_dataset(
        text_ds,
        layer_signature="test_layer",
        batch_size=1,
        verbose=True
    )
    
    # Check that verbose logging occurred
    assert len(caplog.records) > 0


def test_save_activations_dataset_without_store(activations, tmp_path):
    """Test save_activations_dataset raises error when store is None."""
    activations.context.store = None
    
    store = LocalStore(tmp_path / "temp_store")
    ds = Dataset.from_dict({"text": ["Hello"]})
    text_ds = TextDataset(ds, store)
    
    with pytest.raises(ValueError, match="Store must be provided"):
        activations.save_activations_dataset(
            text_ds,
            layer_signature="test_layer"
        )


def test_save_activations_dataset_without_model(activations, tmp_path):
    """Test save_activations_dataset raises error when model is None."""
    store = LocalStore(tmp_path / "store")
    activations.context.store = store
    activations.context.model = None
    
    ds = Dataset.from_dict({"text": ["Hello"]})
    text_ds = TextDataset(ds, store)
    
    with pytest.raises(ValueError, match="Model must be initialized"):
        activations.save_activations_dataset(
            text_ds,
            layer_signature="test_layer"
        )


def test_normalize_layer_signatures_single_string(activations):
    """Test _normalize_layer_signatures with single string."""
    sig_str, sig_list = activations._normalize_layer_signatures("layer1")
    assert sig_str == "layer1"
    assert sig_list == ["layer1"]


def test_normalize_layer_signatures_single_int(activations):
    """Test _normalize_layer_signatures with single int."""
    sig_str, sig_list = activations._normalize_layer_signatures(0)
    assert sig_str == "0"
    assert sig_list == ["0"]


def test_normalize_layer_signatures_list(activations):
    """Test _normalize_layer_signatures with list."""
    sig_str, sig_list = activations._normalize_layer_signatures(["layer1", "layer2"])
    assert sig_str is None  # Multiple layers
    assert sig_list == ["layer1", "layer2"]


def test_normalize_layer_signatures_single_item_list(activations):
    """Test _normalize_layer_signatures with single-item list."""
    sig_str, sig_list = activations._normalize_layer_signatures(["layer1"])
    assert sig_str == "layer1"
    assert sig_list == ["layer1"]


def test_normalize_layer_signatures_none(activations):
    """Test _normalize_layer_signatures with None."""
    sig_str, sig_list = activations._normalize_layer_signatures(None)
    assert sig_str is None
    assert sig_list == []


def test_extract_dataset_info_with_valid_dataset(activations):
    """Test _extract_dataset_info with valid dataset."""
    mock_dataset = Mock()
    mock_dataset.dataset_dir = "/path/to/cache"
    mock_dataset.__len__ = Mock(return_value=100)
    
    info = activations._extract_dataset_info(mock_dataset)
    assert info["dataset_dir"] == "/path/to/cache"
    assert info["length"] == 100


def test_extract_dataset_info_with_none(activations):
    """Test _extract_dataset_info with None dataset."""
    info = activations._extract_dataset_info(None)
    assert info == {}


def test_extract_dataset_info_with_exception(activations):
    """Test _extract_dataset_info handles exceptions."""
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(side_effect=RuntimeError("Error"))
    
    info = activations._extract_dataset_info(mock_dataset)
    assert info["dataset_dir"] == ""
    assert info["length"] == -1


def test_prepare_run_metadata_with_run_name(activations):
    """Test _prepare_run_metadata with provided run name."""
    run_name, meta = activations._prepare_run_metadata(
        "layer1",
        run_name="custom_run"
    )
    assert run_name == "custom_run"
    assert "layer_signatures" in meta


def test_prepare_run_metadata_without_run_name(activations):
    """Test _prepare_run_metadata generates run name."""
    run_name, meta = activations._prepare_run_metadata("layer1")
    assert run_name.startswith("run_")
    assert "layer_signatures" in meta


def test_prepare_run_metadata_with_options(activations):
    """Test _prepare_run_metadata with options."""
    options = {"key": "value", "number": 42}
    run_name, meta = activations._prepare_run_metadata(
        "layer1",
        options=options
    )
    assert meta["options"] == options


def test_prepare_run_metadata_with_dataset(activations):
    """Test _prepare_run_metadata includes dataset info."""
    mock_dataset = Mock()
    mock_dataset.dataset_dir = "/cache"
    mock_dataset.__len__ = Mock(return_value=50)
    
    run_name, meta = activations._prepare_run_metadata(
        "layer1",
        dataset=mock_dataset
    )
    assert "dataset" in meta
    assert meta["dataset"]["length"] == 50
