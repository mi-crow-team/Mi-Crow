"""Store fixtures for testing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from unittest.mock import Mock, MagicMock

from mi_crow.store.local_store import LocalStore
from mi_crow.store.store import Store


def create_temp_store(tmp_path: Path, base_path: Optional[Path] = None) -> LocalStore:
    """
    Create a temporary LocalStore for testing.
    
    Args:
        tmp_path: pytest tmp_path fixture
        base_path: Optional base path (defaults to tmp_path / "store")
        
    Returns:
        LocalStore instance
    """
    if base_path is None:
        base_path = tmp_path / "store"
    return LocalStore(base_path=base_path)


def create_mock_store() -> Mock:
    """
    Create a mock Store for testing.
    
    Returns:
        Mock Store instance with common methods
    """
    mock_store = MagicMock(spec=Store)
    
    # Setup default return values
    mock_store.base_path = Path("/mock/store")
    mock_store.runs_prefix = "runs"
    mock_store.dataset_prefix = "datasets"
    mock_store.model_prefix = "models"
    
    # Setup common methods
    if hasattr(mock_store, 'put_tensor'):
        mock_store.put_tensor.return_value = None
    if hasattr(mock_store, 'get_tensor'):
        mock_store.get_tensor.return_value = None
    mock_store.put_run_batch.return_value = "runs/test_run/batch_000000.safetensors"
    mock_store.get_run_batch.return_value = {}
    mock_store.list_run_batches.return_value = []
    mock_store.put_run_metadata.return_value = "runs/test_run/meta.json"
    mock_store.get_run_metadata.return_value = {}
    mock_store.put_detector_metadata.return_value = "runs/test_run/batch_0"
    if hasattr(mock_store, "put_run_detector_metadata"):
        mock_store.put_run_detector_metadata.return_value = "runs/test_run/detectors"
    mock_store.get_detector_metadata.return_value = ({}, {})
    mock_store.get_detector_metadata_by_layer_by_key.return_value = None
    mock_store.delete_run.return_value = None
    
    return mock_store

