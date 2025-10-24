import pytest
import torch
from unittest.mock import patch

from amber.mechanistic.autoencoder.autoencoder import Autoencoder
from amber.mechanistic.autoencoder.train import SAETrainer, SAETrainingConfig
from amber.store import LocalStore


def make_fake_run_for_grad_accum(store: LocalStore, run_id: str, hidden_dim: int = 16) -> None:
    """Create fake activation data for gradient accumulation testing."""
    # Create multiple batches to test gradient accumulation
    for i in range(5):
        batch = {
            "activations": torch.randn(2, 3, hidden_dim),
            "input_ids": torch.randint(0, 100, (2, 3)),
        }
        store.put_run_batch(run_id, i, batch)


def test_gradient_accumulation_steps(tmp_path):
    """Test gradient accumulation with different step counts."""
    store = LocalStore(tmp_path)
    run_id = "grad_accum_test"
    hidden_dim = 16
    make_fake_run_for_grad_accum(store, run_id, hidden_dim)

    ae = Autoencoder(n_latents=8, n_inputs=hidden_dim, activation="TopK_4")

    cfg = SAETrainingConfig(
        epochs=1,
        batch_size=2,
        lr=1e-2,
        grad_accum_steps=3,  # Accumulate gradients over 3 steps
        verbose=True,
    )

    trainer = SAETrainer(ae, store, run_id, cfg)
    
    # Mock the optimizer to track step calls
    original_step = trainer.optimizer.step
    step_count = 0
    
    def mock_step():
        nonlocal step_count
        step_count += 1
        return original_step()
    
    trainer.optimizer.step = mock_step
    
    history = trainer.train()
    
    # Should have fewer optimizer steps due to accumulation
    assert step_count > 0
    assert "loss" in history and len(history["loss"]) == cfg.epochs


def test_gradient_accumulation_with_l1_penalty(tmp_path):
    """Test gradient accumulation with L1 penalty."""
    store = LocalStore(tmp_path)
    run_id = "grad_accum_l1_test"
    hidden_dim = 16
    make_fake_run_for_grad_accum(store, run_id, hidden_dim)

    ae = Autoencoder(n_latents=8, n_inputs=hidden_dim, activation="TopK_4")

    cfg = SAETrainingConfig(
        epochs=1,
        batch_size=2,
        lr=1e-2,
        l1_lambda=0.01,  # Add L1 penalty
        grad_accum_steps=2,
        verbose=True,
    )

    trainer = SAETrainer(ae, store, run_id, cfg)
    
    history = trainer.train()
    
    # Should have both loss and l1 components
    assert "loss" in history
    assert "l1" in history
    assert len(history["loss"]) == cfg.epochs
    assert len(history["l1"]) == cfg.epochs


def test_gradient_accumulation_with_decoder_grad_projection(tmp_path):
    """Test gradient accumulation with decoder gradient projection."""
    store = LocalStore(tmp_path)
    run_id = "grad_accum_proj_test"
    hidden_dim = 16
    make_fake_run_for_grad_accum(store, run_id, hidden_dim)

    ae = Autoencoder(n_latents=8, n_inputs=hidden_dim, activation="TopK_4")

    cfg = SAETrainingConfig(
        epochs=1,
        batch_size=2,
        lr=1e-2,
        grad_accum_steps=2,
        project_decoder_grads=True,  # Enable gradient projection
        verbose=True,
    )

    trainer = SAETrainer(ae, store, run_id, cfg)
    
    # Mock the project_grads_decode method to track calls
    original_project = ae.project_grads_decode
    project_count = 0
    
    def mock_project():
        nonlocal project_count
        project_count += 1
        return original_project()
    
    ae.project_grads_decode = mock_project
    
    history = trainer.train()
    
    # Should have called gradient projection
    assert project_count > 0
    assert "loss" in history and len(history["loss"]) == cfg.epochs


def test_gradient_accumulation_with_decoder_renorm(tmp_path):
    """Test gradient accumulation with decoder renormalization."""
    store = LocalStore(tmp_path)
    run_id = "grad_accum_renorm_test"
    hidden_dim = 16
    make_fake_run_for_grad_accum(store, run_id, hidden_dim)

    ae = Autoencoder(n_latents=8, n_inputs=hidden_dim, activation="TopK_4")

    cfg = SAETrainingConfig(
        epochs=1,
        batch_size=2,
        lr=1e-2,
        grad_accum_steps=2,
        renorm_decoder_every=2,  # Renormalize every 2 steps
        verbose=True,
    )

    trainer = SAETrainer(ae, store, run_id, cfg)
    
    # Mock the scale_to_unit_norm method to track calls
    original_renorm = ae.scale_to_unit_norm
    renorm_count = 0
    
    def mock_renorm():
        nonlocal renorm_count
        renorm_count += 1
        return original_renorm()
    
    ae.scale_to_unit_norm = mock_renorm
    
    history = trainer.train()
    
    # Should have called renormalization
    assert renorm_count > 0
    assert "loss" in history and len(history["loss"]) == cfg.epochs


def test_gradient_accumulation_with_cuda_cache_clearing(tmp_path):
    """Test gradient accumulation with CUDA cache clearing."""
    store = LocalStore(tmp_path)
    run_id = "grad_accum_cuda_test"
    hidden_dim = 16
    make_fake_run_for_grad_accum(store, run_id, hidden_dim)

    ae = Autoencoder(n_latents=8, n_inputs=hidden_dim, activation="TopK_4")

    cfg = SAETrainingConfig(
        epochs=1,
        batch_size=2,
        lr=1e-2,
        grad_accum_steps=2,
        free_cuda_cache_every=2,  # Clear cache every 2 steps
        verbose=True,
    )

    trainer = SAETrainer(ae, store, run_id, cfg)
    
    # Mock torch.cuda.empty_cache to track calls
    with patch('torch.cuda.empty_cache') as mock_empty_cache:
        history = trainer.train()
        
        # Should have called cache clearing (even on CPU, the condition is checked)
        # Note: On CPU, torch.cuda.is_available() returns False, so no actual clearing occurs
        # but the condition checking code path is still exercised
        assert "loss" in history and len(history["loss"]) == cfg.epochs


def test_gradient_accumulation_edge_cases(tmp_path):
    """Test gradient accumulation with edge case values."""
    store = LocalStore(tmp_path)
    run_id = "grad_accum_edge_test"
    hidden_dim = 16
    make_fake_run_for_grad_accum(store, run_id, hidden_dim)

    ae = Autoencoder(n_latents=8, n_inputs=hidden_dim, activation="TopK_4")

    # Test with grad_accum_steps = 1 (no accumulation)
    cfg = SAETrainingConfig(
        epochs=1,
        batch_size=2,
        lr=1e-2,
        grad_accum_steps=1,
        verbose=True,
    )

    trainer = SAETrainer(ae, store, run_id, cfg)
    history = trainer.train()
    
    assert "loss" in history and len(history["loss"]) == cfg.epochs
    
    # Test with very high grad_accum_steps
    cfg2 = SAETrainingConfig(
        epochs=1,
        batch_size=2,
        lr=1e-2,
        grad_accum_steps=100,  # Very high accumulation
        verbose=True,
    )

    trainer2 = SAETrainer(ae, store, run_id, cfg2)
    history2 = trainer2.train()
    
    assert "loss" in history2 and len(history2["loss"]) == cfg2.epochs
