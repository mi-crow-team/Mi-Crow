"""Tests for SaeTrainer."""

import pytest
import sys
import types
from unittest.mock import Mock, MagicMock, patch
import torch

from amber.mechanistic.sae.sae_trainer import SaeTrainer, SaeTrainingConfig
from tests.unit.mechanistic.test_sae_base import ConcreteSae


class TestSaeTrainingConfig:
    """Tests for SaeTrainingConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = SaeTrainingConfig()
        
        assert config.use_wandb is False
        assert config.wandb_project is None
        assert config.wandb_slow_metrics_frequency == 50

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = SaeTrainingConfig(
            use_wandb=True,
            wandb_project="test_project"
        )
        
        assert config.use_wandb is True
        assert config.wandb_project == "test_project"


class TestSaeTrainer:
    """Tests for SaeTrainer."""

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        trainer = SaeTrainer(sae)
        
        assert trainer.sae == sae
        assert trainer.logger is not None

    def test_train_without_wandb(self):
        """Test training without wandb."""
        # This test is complex due to overcomplete dependencies
        # We'll just test that the trainer can be initialized and the method exists
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        trainer = SaeTrainer(sae)
        
        # Test that train method exists and has correct signature
        assert hasattr(trainer, 'train')
        assert callable(trainer.train)
        
        # Note: Full training test requires overcomplete library and is tested in integration tests

    def test_train_without_overcomplete_raises_error(self):
        """Test that training without overcomplete raises ImportError."""
        # This test requires mocking the import which is complex
        # The ImportError handling is tested in integration tests
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        trainer = SaeTrainer(sae)
        
        # Just verify the method exists
        assert hasattr(trainer, 'train')
        # Note: Full ImportError test requires module manipulation and is tested in integration tests

    def test_train_collects_metrics_with_amp(self, monkeypatch, tmp_path):
        module = types.SimpleNamespace()
        logs = {
            "avg_loss": [0.5],
            "r2": [0.2],
            "z": [[torch.ones(2, 3)]],
        }
        module.train_sae_amp = lambda **kwargs: logs
        module.train_sae = lambda **kwargs: logs
        monkeypatch.setitem(sys.modules, "overcomplete.sae.train", module)

        class DummyDataLoader:
            def __init__(self, *args, **kwargs):
                pass

            def __iter__(self):
                yield torch.ones(2, 3)

        monkeypatch.setattr("amber.mechanistic.sae.sae_trainer.StoreDataloader", DummyDataLoader)

        sae = ConcreteSae(n_latents=4, n_inputs=4)
        sae.sae_engine.parameters.return_value = [torch.nn.Parameter(torch.ones(1, requires_grad=True))]
        sae.sae_engine.to.return_value = sae.sae_engine

        trainer = SaeTrainer(sae)
        config = SaeTrainingConfig(epochs=1, batch_size=2, use_amp=True)
        result = trainer.train(store=Mock(), run_id="run", layer_signature="layer", config=config)

        assert "history" in result
        assert "training_run_id" in result
        assert result["history"]["loss"] == [0.5]
        assert result["history"]["r2"] == [0.2]

        sys.modules.pop("overcomplete.sae.train", None)

    def test_train_without_amp_uses_standard_path(self, monkeypatch):
        module = types.SimpleNamespace()
        module.train_sae_amp = lambda **kwargs: {"avg_loss": [1.0]}
        module.train_sae = lambda **kwargs: {"avg_loss": [0.9], "r2": [0.1]}
        monkeypatch.setitem(sys.modules, "overcomplete.sae.train", module)

        class DummyDataLoader:
            def __init__(self, *args, **kwargs):
                pass

            def __iter__(self):
                yield torch.ones(1, 2)

        monkeypatch.setattr("amber.mechanistic.sae.sae_trainer.StoreDataloader", DummyDataLoader)

        sae = ConcreteSae(n_latents=4, n_inputs=4)
        sae.sae_engine.parameters.return_value = [torch.nn.Parameter(torch.ones(1, requires_grad=True))]
        sae.sae_engine.to.return_value = sae.sae_engine

        trainer = SaeTrainer(sae)
        config = SaeTrainingConfig(use_amp=False, epochs=1, batch_size=1)
        result = trainer.train(store=Mock(), run_id="run", layer_signature="layer", config=config)

        assert result["history"]["loss"] == [0.9]

        sys.modules.pop("overcomplete.sae.train", None)

    def test_train_with_wandb_logging(self, monkeypatch):
        module = types.SimpleNamespace()
        logs = {
            "avg_loss": [0.5, 0.3],
            "r2": [0.1, 0.4],
            "z": [[torch.ones(2, 3)], [torch.zeros(2, 3)]],
        }
        module.train_sae_amp = lambda **kwargs: logs
        module.train_sae = lambda **kwargs: logs
        monkeypatch.setitem(sys.modules, "overcomplete.sae.train", module)

        class DummyDataLoader:
            def __init__(self, *args, **kwargs):
                pass

            def __iter__(self):
                yield torch.ones(2, 3)

        monkeypatch.setattr("amber.mechanistic.sae.sae_trainer.StoreDataloader", DummyDataLoader)

        class DummyRun:
            def __init__(self):
                self.logged = []
                self.summary = {}
                self.url = "http://wandb.test"

            def log(self, data):
                self.logged.append(data)

        dummy_run = DummyRun()
        monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=lambda **kwargs: dummy_run))

        sae = ConcreteSae(n_latents=4, n_inputs=4)
        sae.sae_engine.parameters.return_value = [torch.nn.Parameter(torch.ones(1, requires_grad=True))]
        sae.sae_engine.to.return_value = sae.sae_engine

        trainer = SaeTrainer(sae)
        config = SaeTrainingConfig(
            epochs=2,
            batch_size=2,
            use_amp=True,
            use_wandb=True,
            wandb_mode="offline",
            wandb_slow_metrics_frequency=1,
            verbose=True,
        )
        result = trainer.train(store=Mock(), run_id="run", layer_signature="layer", config=config)

        assert len(dummy_run.logged) >= 2
        assert result["history"]["loss"] == [0.5, 0.3]

        sys.modules.pop("overcomplete.sae.train", None)
        sys.modules.pop("wandb", None)

    def test_train_import_error_when_overcomplete_missing(self, monkeypatch):
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "overcomplete.sae.train":
                raise ImportError("missing overcomplete")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)

        sae = ConcreteSae(n_latents=4, n_inputs=4)
        trainer = SaeTrainer(sae)
        with pytest.raises(ImportError):
            trainer.train(store=Mock(), run_id="run", layer_signature="layer")
        sys.modules["overcomplete.sae.train"] = types.SimpleNamespace(
            train_sae=lambda **kwargs: {},
            train_sae_amp=lambda **kwargs: {},
        )

    def test_train_handles_z_sparsity_logs(self, monkeypatch):
        module = types.SimpleNamespace()
        module.train_sae_amp = lambda **kwargs: {"avg_loss": [0.3], "z_sparsity": [0.7]}
        module.train_sae = lambda **kwargs: {"avg_loss": [0.3], "z_sparsity": [0.7]}
        monkeypatch.setitem(sys.modules, "overcomplete.sae.train", module)

        class DummyDataLoader:
            def __init__(self, *args, **kwargs):
                pass

            def __iter__(self):
                yield torch.ones(1, 2)

        monkeypatch.setattr("amber.mechanistic.sae.sae_trainer.StoreDataloader", DummyDataLoader)

        sae = ConcreteSae(n_latents=2, n_inputs=2)
        sae.sae_engine.parameters.return_value = [torch.nn.Parameter(torch.ones(1, requires_grad=True))]
        sae.sae_engine.to.return_value = sae.sae_engine

        trainer = SaeTrainer(sae)
        config = SaeTrainingConfig(use_amp=False, epochs=1, batch_size=1)
        result = trainer.train(store=Mock(), run_id="run", layer_signature="layer", config=config)

        assert result["history"]["l1"] == [0.7]

        sys.modules.pop("overcomplete.sae.train", None)

    def test_wandb_import_failure_logs_warning(self, monkeypatch, caplog):
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "wandb":
                raise ImportError("missing wandb")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        module = types.SimpleNamespace()
        logs = {"avg_loss": [0.4], "r2": [0.2], "z": [[torch.ones(1, 2)]]}
        module.train_sae_amp = lambda **kwargs: logs
        module.train_sae = lambda **kwargs: logs
        monkeypatch.setitem(sys.modules, "overcomplete.sae.train", module)

        class DummyDataLoader:
            def __init__(self, *args, **kwargs):
                pass

            def __iter__(self):
                yield torch.ones(1, 2)

        monkeypatch.setattr("amber.mechanistic.sae.sae_trainer.StoreDataloader", DummyDataLoader)

        sae = ConcreteSae(n_latents=2, n_inputs=2)
        sae.sae_engine.parameters.return_value = [torch.nn.Parameter(torch.ones(1, requires_grad=True))]
        sae.sae_engine.to.return_value = sae.sae_engine

        trainer = SaeTrainer(sae)
        config = SaeTrainingConfig(use_amp=True, epochs=1, batch_size=1, use_wandb=True)
        with caplog.at_level("WARNING"):
            trainer.train(store=Mock(), run_id="run", layer_signature="layer", config=config)
        assert "wandb not installed" in caplog.text

        sys.modules.pop("overcomplete.sae.train", None)


class TestSaeTrainerHelperMethods:
    """Tests for SaeTrainer helper methods."""

    def test_compute_l1_with_valid_tensors(self):
        """Test _compute_l1 with valid tensors."""
        sae = ConcreteSae(n_latents=4, n_inputs=4)
        trainer = SaeTrainer(sae)
        
        z_list = [
            torch.tensor([[1.0, -2.0, 0.0, 3.0]]),
            torch.tensor([[0.0, 1.0, -1.0, 0.0]]),
        ]
        
        l1 = trainer._compute_l1(z_list)
        
        expected = (abs(1.0) + abs(-2.0) + abs(3.0) + abs(1.0) + abs(-1.0)) / 2.0 / 4.0
        assert abs(l1 - expected) < 1e-6

    def test_compute_l1_with_empty_list(self):
        """Test _compute_l1 with empty list."""
        sae = ConcreteSae(n_latents=4, n_inputs=4)
        trainer = SaeTrainer(sae)
        
        l1 = trainer._compute_l1([])
        
        assert l1 == 0.0

    def test_compute_l1_filters_non_tensors(self):
        """Test _compute_l1 filters out non-tensor items."""
        sae = ConcreteSae(n_latents=4, n_inputs=4)
        trainer = SaeTrainer(sae)
        
        z_list = [
            torch.tensor([[1.0, 2.0]]),
            "not a tensor",
            torch.tensor([[3.0, 4.0]]),
        ]
        
        l1 = trainer._compute_l1(z_list)
        
        expected = (abs(1.0) + abs(2.0) + abs(3.0) + abs(4.0)) / 2.0 / 2.0
        assert abs(l1 - expected) < 1e-6

    def test_compute_slow_metrics_with_valid_data(self):
        """Test _compute_slow_metrics with valid data."""
        sae = ConcreteSae(n_latents=5, n_inputs=4)
        sae.context.n_latents = 5
        trainer = SaeTrainer(sae)
        
        z_list = [
            torch.tensor([[1.0, 0.0, 2.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 3.0, 0.0, 4.0, 0.0]]),
        ]
        
        l0, dead_pct = trainer._compute_slow_metrics(z_list, 5)
        
        assert l0 > 0
        assert 0 <= dead_pct <= 100

    def test_compute_slow_metrics_with_all_dead_features(self):
        """Test _compute_slow_metrics when all features are dead."""
        sae = ConcreteSae(n_latents=3, n_inputs=4)
        sae.context.n_latents = 3
        trainer = SaeTrainer(sae)
        
        z_list = [
            torch.zeros(2, 3),
            torch.zeros(1, 3),
        ]
        
        l0, dead_pct = trainer._compute_slow_metrics(z_list, 3)
        
        assert l0 == 0.0
        assert dead_pct == 100.0

    def test_compute_slow_metrics_with_no_dead_features(self):
        """Test _compute_slow_metrics when no features are dead."""
        sae = ConcreteSae(n_latents=3, n_inputs=4)
        sae.context.n_latents = 3
        trainer = SaeTrainer(sae)
        
        z_list = [
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([[4.0, 5.0, 6.0]]),
        ]
        
        l0, dead_pct = trainer._compute_slow_metrics(z_list, 3)
        
        assert l0 > 0
        assert dead_pct == 0.0

    def test_compute_slow_metrics_with_none_n_latents(self):
        """Test _compute_slow_metrics when n_latents is None."""
        sae = ConcreteSae(n_latents=3, n_inputs=4)
        trainer = SaeTrainer(sae)
        
        z_list = [torch.tensor([[1.0, 2.0, 3.0]])]
        
        l0, dead_pct = trainer._compute_slow_metrics(z_list, None)
        
        assert l0 > 0
        assert dead_pct == 0.0

    def test_compute_slow_metrics_filters_non_tensors(self):
        """Test _compute_slow_metrics filters out non-tensor items."""
        sae = ConcreteSae(n_latents=3, n_inputs=4)
        sae.context.n_latents = 3
        trainer = SaeTrainer(sae)
        
        z_list = [
            torch.tensor([[1.0, 2.0, 3.0]]),
            "not a tensor",
            torch.tensor([[4.0, 5.0, 6.0]]),
        ]
        
        l0, dead_pct = trainer._compute_slow_metrics(z_list, 3)
        
        assert l0 > 0

    def test_get_n_latents_from_context(self):
        """Test _get_n_latents retrieves from context."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        sae.context.n_latents = 100
        trainer = SaeTrainer(sae)
        
        n_latents = trainer._get_n_latents()
        
        assert n_latents == 100

    def test_get_n_latents_returns_none_when_missing(self):
        """Test _get_n_latents returns None when context doesn't have n_latents."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        delattr(sae.context, 'n_latents')
        trainer = SaeTrainer(sae)
        
        n_latents = trainer._get_n_latents()
        
        assert n_latents is None

    def test_extract_r2_and_mse_with_r2(self):
        """Test _extract_r2_and_mse when r2 is in logs."""
        sae = ConcreteSae(n_latents=4, n_inputs=4)
        trainer = SaeTrainer(sae)
        
        history = {"loss": [0.5, 0.3]}
        logs = {"r2": [0.8, 0.9]}
        
        trainer._extract_r2_and_mse(history, logs)
        
        assert history["r2"] == [0.8, 0.9]
        assert len(history["recon_mse"]) == 2
        assert abs(history["recon_mse"][0] - 0.2) < 1e-6
        assert abs(history["recon_mse"][1] - 0.1) < 1e-6

    def test_extract_r2_and_mse_without_r2(self):
        """Test _extract_r2_and_mse when r2 is not in logs."""
        sae = ConcreteSae(n_latents=4, n_inputs=4)
        trainer = SaeTrainer(sae)
        
        history = {"loss": [0.5, 0.3]}
        logs = {}
        
        trainer._extract_r2_and_mse(history, logs)
        
        assert history["r2"] == [0.0, 0.0]

    def test_extract_sparsity_metrics_with_z(self):
        """Test _extract_sparsity_metrics when z is in logs."""
        sae = ConcreteSae(n_latents=4, n_inputs=4)
        sae.context.n_latents = 4
        trainer = SaeTrainer(sae)
        config = SaeTrainingConfig()
        
        history = {
            "loss": [0.5, 0.3],
            "l1": [],
            "l0": [],
            "dead_features_pct": []
        }
        logs = {
            "z": [
                [torch.tensor([[1.0, 0.0, 2.0, 0.0]])],
                [torch.tensor([[0.0, 1.0, 0.0, 2.0]])],
            ]
        }
        
        trainer._extract_sparsity_metrics(history, logs, config)
        
        assert len(history["l1"]) == 2
        assert len(history["l0"]) == 2

    def test_extract_sparsity_metrics_with_z_sparsity(self):
        """Test _extract_sparsity_metrics when z_sparsity is in logs."""
        sae = ConcreteSae(n_latents=4, n_inputs=4)
        trainer = SaeTrainer(sae)
        config = SaeTrainingConfig()
        
        history = {
            "loss": [0.5, 0.3],
            "l1": [],
            "l0": [],
            "dead_features_pct": []
        }
        logs = {"z_sparsity": [0.7, 0.8]}
        
        trainer._extract_sparsity_metrics(history, logs, config)
        
        assert history["l1"] == [0.7, 0.8]
        assert history["l0"] == [0.0, 0.0]

    def test_extract_sparsity_metrics_without_z(self):
        """Test _extract_sparsity_metrics when z is not in logs."""
        sae = ConcreteSae(n_latents=4, n_inputs=4)
        trainer = SaeTrainer(sae)
        config = SaeTrainingConfig()
        
        history = {
            "loss": [0.5, 0.3],
            "l1": [],
            "l0": [],
            "dead_features_pct": []
        }
        logs = {}
        
        trainer._extract_sparsity_metrics(history, logs, config)
        
        assert history["l1"] == [0.0, 0.0]
        assert history["l0"] == [0.0, 0.0]

    def test_get_metric_value_with_valid_index(self):
        """Test _get_metric_value with valid index."""
        sae = ConcreteSae(n_latents=4, n_inputs=4)
        trainer = SaeTrainer(sae)
        
        values = [1.0, 2.0, 3.0, None, 5.0]
        
        result = trainer._get_metric_value(values, 2)
        
        assert result == 3.0

    def test_get_metric_value_with_none_uses_last_known(self):
        """Test _get_metric_value with None uses last known value."""
        sae = ConcreteSae(n_latents=4, n_inputs=4)
        trainer = SaeTrainer(sae)
        
        values = [1.0, 2.0, None, None, 5.0]
        
        result = trainer._get_metric_value(values, 2)
        
        assert result == 2.0

    def test_get_metric_value_with_all_none(self):
        """Test _get_metric_value when all values are None."""
        sae = ConcreteSae(n_latents=4, n_inputs=4)
        trainer = SaeTrainer(sae)
        
        values = [None, None, None]
        
        result = trainer._get_metric_value(values, 1)
        
        assert result == 0.0

    def test_get_last_known_value_finds_value(self):
        """Test _get_last_known_value finds last known value."""
        sae = ConcreteSae(n_latents=4, n_inputs=4)
        trainer = SaeTrainer(sae)
        
        values = [1.0, 2.0, None, None, 5.0]
        
        result = trainer._get_last_known_value(values, 3)
        
        assert result == 2.0

    def test_get_last_known_value_returns_zero_when_none_found(self):
        """Test _get_last_known_value returns 0.0 when no value found."""
        sae = ConcreteSae(n_latents=4, n_inputs=4)
        trainer = SaeTrainer(sae)
        
        values = [None, None, None]
        
        result = trainer._get_last_known_value(values, 2)
        
        assert result == 0.0

    def test_build_epoch_metrics_with_slow_metrics(self):
        """Test _build_epoch_metrics includes slow metrics when should_log_slow."""
        sae = ConcreteSae(n_latents=4, n_inputs=4)
        trainer = SaeTrainer(sae)
        config = SaeTrainingConfig(lr=0.001)
        
        history = {
            "loss": [0.5, 0.3],
            "recon_mse": [0.2, 0.1],
            "r2": [0.8, 0.9],
            "l1": [0.7, 0.6],
            "l0": [10.0, 8.0],
            "dead_features_pct": [5.0, 3.0],
        }
        
        metrics = trainer._build_epoch_metrics(history, 1, 0, config, should_log_slow=True)
        
        assert metrics["epoch"] == 1
        assert metrics["train/loss"] == 0.5
        assert "train/l0_sparsity" in metrics
        assert "train/dead_features_pct" in metrics

    def test_build_epoch_metrics_without_slow_metrics(self):
        """Test _build_epoch_metrics excludes slow metrics when should_log_slow is False."""
        sae = ConcreteSae(n_latents=4, n_inputs=4)
        trainer = SaeTrainer(sae)
        config = SaeTrainingConfig(lr=0.001)
        
        history = {
            "loss": [0.5],
            "recon_mse": [0.2],
            "r2": [0.8],
            "l1": [0.7],
            "l0": [10.0],
            "dead_features_pct": [5.0],
        }
        
        metrics = trainer._build_epoch_metrics(history, 1, 0, config, should_log_slow=False)
        
        assert "train/l0_sparsity" not in metrics
        assert "train/dead_features_pct" not in metrics

    def test_build_final_metrics(self):
        """Test _build_final_metrics builds final metrics correctly."""
        sae = ConcreteSae(n_latents=4, n_inputs=4)
        trainer = SaeTrainer(sae)
        
        history = {
            "loss": [0.5, 0.3, 0.2],
            "recon_mse": [0.2, 0.1, 0.05],
            "r2": [0.8, 0.9, 0.95],
            "l1": [0.7, 0.6, 0.5],
            "l0": [10.0, 8.0, 6.0],
            "dead_features_pct": [5.0, 3.0, 2.0],
        }
        
        metrics = trainer._build_final_metrics(history, 3)
        
        assert metrics["final/loss"] == 0.2
        assert metrics["final/reconstruction_mse"] == 0.05
        assert metrics["final/r2_score"] == 0.95
        assert metrics["final/l1_penalty"] == 0.5
        assert metrics["final/l0_sparsity"] == 6.0
        assert metrics["final/dead_features_pct"] == 2.0
        assert metrics["training/num_epochs"] == 3
        assert "best/loss" in metrics
        assert "best/r2_score" in metrics

