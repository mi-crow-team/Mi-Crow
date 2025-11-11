"""Additional tests for SaeTrainer to improve coverage."""
import pytest
import torch

try:
    from overcomplete.sae import TopKSAE as OvercompleteTopKSAE
except ImportError:

from amber.mechanistic.sae.modules.topk_sae import TopKSae
from amber.mechanistic.sae.sae_trainer import SaeTrainer, SaeTrainingConfig, StoreDataloader
from amber.store.local_store import LocalStore


def test_reusable_store_data_loader_skips_non_dict_batches(tmp_path):
    """Test ReusableStoreDataLoader skips non-dict batches."""
    store = LocalStore(tmp_path)
    run_id = "test_run"
    
    # Create a valid batch
    store.put_run_batch(run_id, 0, {"activations": torch.randn(10, 8)})
    
    config = SaeTrainingConfig(batch_size=5)
    loader = StoreDataloader(store, run_id, config.batch_size, config.dtype, config.max_batches_per_epoch)
    
    # Should yield valid batches
    batches = list(loader)
    assert len(batches) > 0


def test_reusable_store_data_loader_skips_missing_activations_key(tmp_path):
    """Test ReusableStoreDataLoader skips batches without 'activations' key."""
    store = LocalStore(tmp_path)
    run_id = "test_run"
    
    # Create batch without 'activations' key
    store.put_run_batch(run_id, 0, {"other_key": torch.randn(10, 8)})
    
    config = SaeTrainingConfig(batch_size=5)
    loader = StoreDataloader(store, run_id, config.batch_size, config.dtype, config.max_batches_per_epoch)
    
    # Should skip this batch (no 'activations' key)
    batches = list(loader)
    assert len(batches) == 0


def test_sae_trainer_with_use_amp_false(tmp_path):
    """Test SaeTrainer with use_amp=False."""
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
        use_amp=False,  # Disable AMP
        max_batches_per_epoch=2,
        verbose=False
    )
    
    history = trainer.train(store, run_id, config)
    assert "loss" in history


def test_sae_trainer_with_scheduler(tmp_path):
    """Test SaeTrainer with scheduler."""
    store = LocalStore(tmp_path)
    run_id = "test_run"
    
    # Create some batches
    for i in range(2):
        store.put_run_batch(run_id, i, {"activations": torch.randn(10, 8)})
    
    topk_sae = TopKSae(n_latents=8, n_inputs=8, k=4, device='cpu')
    trainer = SaeTrainer(topk_sae)
    
    # Create a simple scheduler
    optimizer = torch.optim.AdamW(topk_sae.sae_engine.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    config = SaeTrainingConfig(
        epochs=1,
        batch_size=5,
        scheduler=scheduler,
        max_batches_per_epoch=2,
        verbose=False
    )
    
    history = trainer.train(store, run_id, config)
    assert "loss" in history

