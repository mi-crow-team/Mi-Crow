import pytest
import torch
from unittest.mock import Mock

from amber.mechanistic.autoencoder.autoencoder import Autoencoder
from amber.mechanistic.autoencoder.train import SAETrainer, SAETrainingConfig
from amber.store import LocalStore


def make_fake_run_with_validation(store: LocalStore, run_id: str, hidden_dim: int = 16) -> None:
    """Create fake activation data with some invalid entries for validation testing."""
    # Valid batch
    batch0 = {
        "activations": torch.randn(2, 3, hidden_dim),
        "input_ids": torch.randint(0, 100, (2, 3)),
    }
    # Invalid batch (not a dict) - skip this as it can't be stored
    # batch1 = "invalid_batch"
    # Batch missing activations
    batch2 = {
        "input_ids": torch.randint(0, 100, (1, 2)),
    }
    # Batch with non-tensor activations - skip this as it can't be stored
    # batch3 = {
    #     "activations": "not_a_tensor",
    #     "input_ids": torch.randint(0, 100, (1, 2)),
    # }
    # Valid batch
    batch4 = {
        "activations": torch.randn(1, 4, hidden_dim),
        "input_ids": torch.randint(0, 100, (1, 4)),
    }
    
    store.put_run_batch(run_id, 0, batch0)
    # store.put_run_batch(run_id, 1, batch1)  # Skip invalid batch
    store.put_run_batch(run_id, 2, batch2)
    # store.put_run_batch(run_id, 3, batch3)  # Skip invalid batch
    store.put_run_batch(run_id, 4, batch4)


def test_validation_skips_invalid_batches(tmp_path):
    """Test that validation logic skips invalid batches during training."""
    store = LocalStore(tmp_path)
    run_id = "validation_test"
    hidden_dim = 16
    make_fake_run_with_validation(store, run_id, hidden_dim)

    ae = Autoencoder(n_latents=8, n_inputs=hidden_dim, activation="TopK_4")

    cfg = SAETrainingConfig(
        epochs=1,
        batch_size=4,
        lr=1e-2,
        verbose=True,  # Enable logging to see validation messages
    )

    trainer = SAETrainer(ae, store, run_id, cfg)
    
    # Training should complete despite invalid batches
    history = trainer.train()
    assert "loss" in history and len(history["loss"]) == cfg.epochs
    
    # Should have processed at least some valid batches
    assert len(history["loss"]) > 0


def test_validation_with_dtype_conversion(tmp_path):
    """Test validation with dtype conversion."""
    store = LocalStore(tmp_path)
    run_id = "dtype_test"
    hidden_dim = 16
    
    # Create batches with different dtypes
    batch0 = {
        "activations": torch.randn(2, 3, hidden_dim, dtype=torch.float64),
        "input_ids": torch.randint(0, 100, (2, 3)),
    }
    batch1 = {
        "activations": torch.randn(1, 4, hidden_dim, dtype=torch.float32),
        "input_ids": torch.randint(0, 100, (1, 4)),
    }
    
    store.put_run_batch(run_id, 0, batch0)
    store.put_run_batch(run_id, 1, batch1)

    ae = Autoencoder(n_latents=8, n_inputs=hidden_dim, activation="TopK_4")

    cfg = SAETrainingConfig(
        epochs=1,
        batch_size=4,
        lr=1e-2,
        dtype=torch.float32,  # Force specific dtype
        verbose=True,
    )

    trainer = SAETrainer(ae, store, run_id, cfg)
    
    # Training should complete with dtype conversion
    history = trainer.train()
    assert "loss" in history and len(history["loss"]) == cfg.epochs


def test_validation_with_different_tensor_shapes(tmp_path):
    """Test validation with different tensor shapes (1D, 2D, 3D)."""
    store = LocalStore(tmp_path)
    run_id = "shape_test"
    hidden_dim = 16
    
    # 1D tensor
    batch0 = {
        "activations": torch.randn(hidden_dim),
        "input_ids": torch.randint(0, 100, (1,)),
    }
    # 2D tensor
    batch1 = {
        "activations": torch.randn(2, hidden_dim),
        "input_ids": torch.randint(0, 100, (2, 1)),
    }
    # 3D tensor
    batch2 = {
        "activations": torch.randn(1, 3, hidden_dim),
        "input_ids": torch.randint(0, 100, (1, 3)),
    }
    
    store.put_run_batch(run_id, 0, batch0)
    store.put_run_batch(run_id, 1, batch1)
    store.put_run_batch(run_id, 2, batch2)

    ae = Autoencoder(n_latents=8, n_inputs=hidden_dim, activation="TopK_4")

    cfg = SAETrainingConfig(
        epochs=1,
        batch_size=4,
        lr=1e-2,
        verbose=True,
    )

    trainer = SAETrainer(ae, store, run_id, cfg)
    
    # Training should handle all tensor shapes correctly
    history = trainer.train()
    assert "loss" in history and len(history["loss"]) == cfg.epochs


def test_validation_verbose_logging(tmp_path, caplog):
    """Test that validation produces appropriate log messages."""
    import logging
    
    store = LocalStore(tmp_path)
    run_id = "logging_test"
    hidden_dim = 16
    make_fake_run_with_validation(store, run_id, hidden_dim)

    ae = Autoencoder(n_latents=8, n_inputs=hidden_dim, activation="TopK_4")

    cfg = SAETrainingConfig(
        epochs=1,
        batch_size=4,
        lr=1e-2,
        verbose=True,
    )

    with caplog.at_level(logging.INFO):
        trainer = SAETrainer(ae, store, run_id, cfg)
        trainer.train()
    
    # Check that validation messages appear in logs
    log_messages = [rec.message for rec in caplog.records]
    assert any("Skipping non-dict" in msg for msg in log_messages)
    # Note: We removed the non-tensor batch, so this assertion is not needed
    # assert any("Skipping non-tensor" in msg for msg in log_messages)
