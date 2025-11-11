"""Additional tests to improve coverage for sae_trainer.py."""
import pytest
import torch
import logging
from unittest.mock import patch

try:
    from amber.mechanistic.sae.modules.topk_sae import TopKSae
    from amber.mechanistic.sae.sae_trainer import SaeTrainer, SaeTrainingConfig, StoreDataloader
    from amber.store.local_store import LocalStore
    OVERCOMPLETE_AVAILABLE = True
except ImportError:
    OVERCOMPLETE_AVAILABLE = False
    TopKSae = None  # type: ignore
    SaeTrainer = None  # type: ignore
    SaeTrainingConfig = None  # type: ignore
    StoreDataloader = None  # type: ignore
    LocalStore = None  # type: ignore


@pytest.mark.skipif(not OVERCOMPLETE_AVAILABLE, reason='Overcomplete not available')
def test_reusable_store_data_loader_logs_skipped_batches(tmp_path, caplog):
    """Test that ReusableStoreDataLoader logs when skipping batches (lines 67->69, 72->74)."""
    store = LocalStore(tmp_path)
    run_id = "test_run"
    
    # Create batches with missing 'activations' key (put_run_batch requires tensors, so we use other_key)
    store.put_run_batch(run_id, 0, {"other_key": torch.randn(5, 8)})
    # Create valid batch
    store.put_run_batch(run_id, 1, {"activations": torch.randn(5, 8)})
    
    # Set logger level to DEBUG to capture debug messages
    logger = logging.getLogger("amber.mechanistic.sae.sae_trainer")
    logger.setLevel(logging.DEBUG)
    
    loader = StoreDataloader(store, run_id, batch_size=5, logger_instance=logger)
    
    with caplog.at_level(logging.DEBUG):
        batches = list(loader)
    
    # Should skip invalid batches and only yield valid one
    assert len(batches) == 1
    # Check that debug messages were logged for missing 'activations' key
    log_messages = [record.message for record in caplog.records]
    assert any("Skipping non-dict or missing 'activations'" in msg for msg in log_messages)


@pytest.mark.skipif(not OVERCOMPLETE_AVAILABLE, reason='Overcomplete not available')
def test_reusable_store_data_loader_handles_1d_tensor(tmp_path):
    """Test ReusableStoreDataLoader handles 1D tensors (line 80)."""
    store = LocalStore(tmp_path)
    run_id = "test_run"
    
    # Create batch with 1D tensor
    store.put_run_batch(run_id, 0, {"activations": torch.randn(8)})  # 1D
    
    loader = StoreDataloader(store, run_id, batch_size=5)
    
    batches = list(loader)
    
    # Should reshape 1D to 2D [1, 8]
    assert len(batches) == 1
    assert batches[0].dim() == 2
    assert batches[0].shape == (1, 8)


@pytest.mark.skipif(not OVERCOMPLETE_AVAILABLE, reason='Overcomplete not available')
def test_sae_trainer_monitoring_override_with_verbose(tmp_path):
    """Test that SaeTrainer overrides monitoring when verbose is True (line 199)."""
    store = LocalStore(tmp_path)
    run_id = "test_run"
    
    # Create some batches
    for i in range(2):
        store.put_run_batch(run_id, i, {"activations": torch.randn(10, 8)})
    
    topk_sae = TopKSae(n_latents=8, n_inputs=8, k=4, device='cpu')
    trainer = SaeTrainer(topk_sae)
    
    # Config with monitoring=1 but verbose=True should override to 2
    config = SaeTrainingConfig(
        epochs=1,
        batch_size=5,
        monitoring=1,  # Low monitoring
        verbose=True,  # Should override to 2
        max_batches_per_epoch=2,
        use_amp=False
    )
    
    # Mock overcomplete's train_sae - it's imported inside the train method
    with patch('overcomplete.sae.train.train_sae') as mock_train:
        mock_train.return_value = {"avg_loss": [0.5]}
        
        trainer.train(store, run_id, config)
        
        # Verify monitoring was overridden to 2
        call_kwargs = mock_train.call_args[1]
        assert call_kwargs["monitoring"] == 2


@pytest.mark.skipif(not OVERCOMPLETE_AVAILABLE, reason='Overcomplete not available')
def test_sae_trainer_verbose_logging(tmp_path, caplog):
    """Test that SaeTrainer logs verbose messages (lines 202, 267)."""
    store = LocalStore(tmp_path)
    run_id = "test_run"
    
    # Create some batches
    for i in range(2):
        store.put_run_batch(run_id, i, {"activations": torch.randn(10, 8)})
    
    topk_sae = TopKSae(n_latents=8, n_inputs=8, k=4, device='cpu')
    trainer = SaeTrainer(topk_sae)
    
    config = SaeTrainingConfig(
        epochs=1,
        batch_size=5,
        verbose=True,  # Enable verbose logging
        max_batches_per_epoch=2,
        use_amp=False
    )
    
    # Set logger level to INFO to capture info messages
    logger = logging.getLogger("amber.mechanistic.sae.sae_trainer")
    logger.setLevel(logging.INFO)
    
    with caplog.at_level(logging.INFO):
        with patch('overcomplete.sae.train.train_sae') as mock_train:
            mock_train.return_value = {"avg_loss": [0.5]}
            
            trainer.train(store, run_id, config)
    
    # Check that verbose messages were logged
    log_messages = [record.message for record in caplog.records]
    assert any("Starting training" in msg for msg in log_messages)
    assert any("Completed training" in msg for msg in log_messages)


@pytest.mark.skipif(not OVERCOMPLETE_AVAILABLE, reason='Overcomplete not available')
def test_sae_trainer_history_with_z_batch_list(tmp_path):
    """Test SaeTrainer history conversion with z batch list (lines 251->257)."""
    store = LocalStore(tmp_path)
    run_id = "test_run"
    
    # Create some batches
    for i in range(2):
        store.put_run_batch(run_id, i, {"activations": torch.randn(10, 8)})
    
    topk_sae = TopKSae(n_latents=8, n_inputs=8, k=4, device='cpu')
    trainer = SaeTrainer(topk_sae)
    
    config = SaeTrainingConfig(
        epochs=2,
        batch_size=5,
        verbose=False,
        max_batches_per_epoch=2,
        use_amp=False,
        monitoring=2  # Enable z storage
    )
    
    # Mock overcomplete's train_sae to return logs with z batch list
    with patch('overcomplete.sae.train.train_sae') as mock_train:
        # Create mock logs with z batch list format
        z_batch1 = [torch.randn(5, 8), torch.randn(5, 8)]
        z_batch2 = [torch.randn(5, 8)]
        mock_train.return_value = {
            "avg_loss": [0.5, 0.4],
            "z": [z_batch1, z_batch2]  # List of lists of tensors
        }
        
        history = trainer.train(store, run_id, config)
        
        # Verify history has l1 computed from z batches
        assert "l1" in history
        assert len(history["l1"]) == 2  # One per epoch
        
        # Verify l1 values are computed (mean of absolute values)
        for l1_val in history["l1"]:
            assert isinstance(l1_val, float)
            assert l1_val >= 0


@pytest.mark.skipif(not OVERCOMPLETE_AVAILABLE, reason='Overcomplete not available')
def test_sae_trainer_history_with_empty_z_batch_list(tmp_path):
    """Test SaeTrainer history conversion with empty z batch list (line 257)."""
    store = LocalStore(tmp_path)
    run_id = "test_run"
    
    # Create some batches
    for i in range(2):
        store.put_run_batch(run_id, i, {"activations": torch.randn(10, 8)})
    
    topk_sae = TopKSae(n_latents=8, n_inputs=8, k=4, device='cpu')
    trainer = SaeTrainer(topk_sae)
    
    config = SaeTrainingConfig(
        epochs=2,
        batch_size=5,
        verbose=False,
        max_batches_per_epoch=2,
        use_amp=False,
        monitoring=2
    )
    
    # Mock overcomplete's train_sae to return logs with empty z batch list
    with patch('overcomplete.sae.train.train_sae') as mock_train:
        mock_train.return_value = {
            "avg_loss": [0.5, 0.4],
            "z": [[], []]  # Empty lists
        }
        
        history = trainer.train(store, run_id, config)
        
        # Verify history has l1 with zeros for empty batches
        assert "l1" in history
        assert len(history["l1"]) == 2
        assert history["l1"][0] == 0.0
        assert history["l1"][1] == 0.0


@pytest.mark.skipif(not OVERCOMPLETE_AVAILABLE, reason='Overcomplete not available')
def test_sae_trainer_history_with_non_list_z_batch(tmp_path):
    """Test SaeTrainer history conversion with non-list z batch (line 257)."""
    store = LocalStore(tmp_path)
    run_id = "test_run"
    
    # Create some batches
    for i in range(2):
        store.put_run_batch(run_id, i, {"activations": torch.randn(10, 8)})
    
    topk_sae = TopKSae(n_latents=8, n_inputs=8, k=4, device='cpu')
    trainer = SaeTrainer(topk_sae)
    
    config = SaeTrainingConfig(
        epochs=2,
        batch_size=5,
        verbose=False,
        max_batches_per_epoch=2,
        use_amp=False,
        monitoring=2
    )
    
    # Mock overcomplete's train_sae to return logs with non-list z batch
    with patch('overcomplete.sae.train.train_sae') as mock_train:
        mock_train.return_value = {
            "avg_loss": [0.5, 0.4],
            "z": [torch.randn(5, 8), "not a list"]  # Second is not a list
        }
        
        history = trainer.train(store, run_id, config)
        
        # Verify history handles non-list gracefully
        assert "l1" in history
        assert len(history["l1"]) == 2
        # First should have value, second should be 0.0 (not a list)
        assert history["l1"][1] == 0.0

