"""Integration tests for overcomplete library."""

import pytest
import tempfile
from pathlib import Path
import torch
from unittest.mock import patch, MagicMock

from amber.mechanistic.sae.sae_trainer import SaeTrainer, SaeTrainingConfig
from tests.unit.mechanistic.test_sae_base import ConcreteSae
from tests.unit.fixtures.stores import create_temp_store


class TestOvercompleteIntegration:
    """Tests for overcomplete library integration."""

    @patch('overcomplete.sae.train.train_sae')
    @patch('overcomplete.sae.train.train_sae_amp')
    def test_sae_trainer_with_overcomplete(self, mock_train_sae_amp, mock_train_sae):
        """Test SaeTrainer integration with overcomplete."""
        logs = {
            "avg_loss": [1.0, 0.5, 0.3],
            "r2": [0.1, 0.5, 0.7],
            "z": [[torch.ones(2, 2)]],
        }
        mock_train_sae.return_value = logs
        mock_train_sae_amp.return_value = logs
        
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        sae.sae_engine.parameters.return_value = [torch.nn.Parameter(torch.ones(1, requires_grad=True))]
        sae.sae_engine.to.return_value = sae.sae_engine
        trainer = SaeTrainer(sae)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_temp_store(Path(tmpdir))
            tensor_metadata = {
                "layer_0": {"activations": torch.randn(10, 200)}
            }
            store.put_detector_metadata("run_1", 0, {}, tensor_metadata)
            
            config = SaeTrainingConfig(use_wandb=False)
            result = trainer.train(store, "run_1", "layer_0", config)
            
            assert result["loss"] == [1.0, 0.5, 0.3]
            mock_train_sae_amp.assert_called_once()

    def test_overcomplete_unavailable_raises_error(self):
        """Test error handling when overcomplete is unavailable."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        trainer = SaeTrainer(sae)
        
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "overcomplete.sae.train":
                raise ImportError("No module named 'overcomplete'")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with tempfile.TemporaryDirectory() as tmpdir:
                store = create_temp_store(Path(tmpdir))
                with pytest.raises(ImportError, match="overcomplete.sae.train module not available"):
                    trainer.train(store, "run_1", "layer_0")

