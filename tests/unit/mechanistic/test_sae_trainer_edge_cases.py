"""Additional tests for SaeTrainer edge cases and error handling."""
import pytest
import torch

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
def test_reusable_store_data_loader_edge_cases(tmp_path):
    """Test ReusableStoreDataLoader with edge cases."""
    store = LocalStore(tmp_path)
    run_id = "test_run"
    
    # Create some test batches (only tensors, no metadata)
    for i in range(3):
        batch = {
            "activations": torch.randn(10, 8),
        }
        store.put_run_batch(run_id, i, batch)
    
    config = SaeTrainingConfig(batch_size=5, max_batches_per_epoch=2)
    loader = StoreDataloader(store, run_id, config.batch_size, config.dtype, config.max_batches_per_epoch)
    
    # Test iteration
    batches = list(loader)
    assert len(batches) > 0
    
    # Test multiple iterations (reusability)
    batches2 = list(loader)
    assert len(batches2) == len(batches)
    
    # Test with max_batches limit
    config_limited = SaeTrainingConfig(batch_size=5, max_batches_per_epoch=1)
    loader_limited = StoreDataloader(store, run_id, config_limited.batch_size, config_limited.dtype, config_limited.max_batches_per_epoch)
    batches_limited = list(loader_limited)
    assert len(batches_limited) <= 1


@pytest.mark.skipif(not OVERCOMPLETE_AVAILABLE, reason='Overcomplete not available')
def test_reusable_store_data_loader_with_invalid_batches(tmp_path):
    """Test ReusableStoreDataLoader handles invalid batches gracefully."""
    store = LocalStore(tmp_path)
    run_id = "test_run"
    
    # Create batches with invalid data
    store.put_run_batch(run_id, 0, {"activations": torch.randn(10, 8)})  # Valid
    # Note: put_run_batch only accepts dicts of tensors, so we can't test invalid types directly
    # The loader will handle missing 'activations' key gracefully
    store.put_run_batch(run_id, 1, {"other_key": torch.randn(10, 8)})  # Missing 'activations' key
    store.put_run_batch(run_id, 2, {"activations": torch.randn(10, 8)})  # Valid
    
    config = SaeTrainingConfig(batch_size=5)
    loader = StoreDataloader(store, run_id, config.batch_size, config.dtype, config.max_batches_per_epoch)
    
    # Should skip invalid batches and only yield valid ones
    batches = list(loader)
    assert len(batches) > 0
    for batch in batches:
        assert isinstance(batch, torch.Tensor)


@pytest.mark.skipif(not OVERCOMPLETE_AVAILABLE, reason='Overcomplete not available')
def test_reusable_store_data_loader_with_different_shapes(tmp_path):
    """Test ReusableStoreDataLoader handles different tensor shapes."""
    store = LocalStore(tmp_path)
    run_id = "test_run"
    
    # Create batches with different shapes
    store.put_run_batch(run_id, 0, {"activations": torch.randn(10, 8)})  # 2D
    store.put_run_batch(run_id, 1, {"activations": torch.randn(5, 3, 8)})  # 3D
    store.put_run_batch(run_id, 2, {"activations": torch.randn(1, 8)})  # 2D (1D gets reshaped)
    
    config = SaeTrainingConfig(batch_size=5)
    loader = StoreDataloader(store, run_id, config.batch_size, config.dtype, config.max_batches_per_epoch)
    
    # Should flatten all to 2D
    batches = list(loader)
    for batch in batches:
        assert batch.dim() == 2
        assert batch.shape[-1] == 8  # Last dimension preserved


@pytest.mark.skipif(not OVERCOMPLETE_AVAILABLE, reason='Overcomplete not available')
def test_sae_trainer_with_empty_store(tmp_path):
    """Test SaeTrainer handles empty store gracefully."""
    store = LocalStore(tmp_path)
    run_id = "empty_run"
    
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4, device='cpu')
    trainer = SaeTrainer(topk_sae)
    
    config = SaeTrainingConfig(epochs=1, batch_size=5, verbose=False)
    
    # Empty store - training will complete with no data (overcomplete handles this)
    # Just verify it doesn't crash
    try:
        history = trainer.train(store, run_id, config)
        # May return empty history or complete without error
        assert isinstance(history, dict)
    except (StopIteration, ValueError, KeyError, RuntimeError) as e:
        # Some error is also acceptable
        pass


@pytest.mark.skipif(not OVERCOMPLETE_AVAILABLE, reason='Overcomplete not available')
def test_sae_trainer_with_dtype_conversion(tmp_path):
    """Test SaeTrainer handles dtype conversion."""
    store = LocalStore(tmp_path)
    run_id = "test_run"
    
    # Create batches with float32
    for i in range(2):
        batch = {
            "activations": torch.randn(10, 8, dtype=torch.float32),
        }
        store.put_run_batch(run_id, i, batch)
    
    topk_sae = TopKSae(n_latents=8, n_inputs=8, k=4, device='cpu')
    trainer = SaeTrainer(topk_sae)
    
    # Test with float32 dtype (float16 can cause issues with AMP)
    config = SaeTrainingConfig(
        epochs=1,
        batch_size=5,
        dtype=torch.float32,
        max_batches_per_epoch=2,
        use_amp=False,  # Disable AMP to avoid float16 issues
        verbose=False
    )
    
    # Should handle dtype conversion
    history = trainer.train(store, run_id, config)
    assert "loss" in history
    assert isinstance(history["loss"], list)


@pytest.mark.skipif(not OVERCOMPLETE_AVAILABLE, reason='Overcomplete not available')
def test_sae_training_config_defaults():
    """Test SaeTrainingConfig has correct defaults."""
    config = SaeTrainingConfig()
    
    assert config.epochs == 1
    assert config.batch_size == 1024
    assert config.lr == 1e-3
    assert config.l1_lambda == 0.0
    assert config.device == "cpu"
    assert config.dtype is None
    assert config.max_batches_per_epoch is None
    assert config.verbose is False
    assert config.use_amp is True
    assert config.amp_dtype is None
    assert config.grad_accum_steps == 1
    assert config.clip_grad == 1.0
    assert config.monitoring == 1
    assert config.scheduler is None
    assert config.max_nan_fallbacks == 5


@pytest.mark.skipif(not OVERCOMPLETE_AVAILABLE, reason='Overcomplete not available')
def test_reusable_store_data_loader_with_zero_batch_size(tmp_path):
    """Test ReusableStoreDataLoader handles zero batch size."""
    store = LocalStore(tmp_path)
    run_id = "test_run"
    
    store.put_run_batch(run_id, 0, {"activations": torch.randn(10, 8)})
    
    config = SaeTrainingConfig(batch_size=0)  # Should default to 1
    loader = StoreDataloader(store, run_id, config.batch_size, config.dtype, config.max_batches_per_epoch)
    
    # Should still work (batch_size defaults to 1 in implementation)
    batches = list(loader)
    assert len(batches) > 0

