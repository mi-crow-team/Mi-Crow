"""Integration tests for overcomplete library."""

import pytest
import tempfile
from unittest.mock import patch, MagicMock

from amber.mechanistic.sae.sae_trainer import SaeTrainer, SaeTrainingConfig
from tests.unit.mechanistic.test_sae_base import ConcreteSae
from tests.unit.fixtures.stores import create_temp_store


class TestOvercompleteIntegration:
    """Tests for overcomplete library integration."""

    @patch('amber.mechanistic.sae.sae_trainer.train_sae')
    def test_sae_trainer_with_overcomplete(self, mock_train_sae):
        """Test SaeTrainer integration with overcomplete."""
        mock_train_sae.return_value = {"loss": [1.0, 0.5, 0.3]}
        
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        trainer = SaeTrainer(sae)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_temp_store(tmpdir)
            import torch
            tensor_metadata = {
                "layer_0": {"activations": torch.randn(10, 200)}
            }
            store.put_detector_metadata("run_1", 0, {}, tensor_metadata)
            
            config = SaeTrainingConfig(use_wandb=False)
            result = trainer.train(store, "run_1", "layer_0", config)
            
            assert result is not None
            mock_train_sae.assert_called_once()

    def test_overcomplete_unavailable_raises_error(self):
        """Test error handling when overcomplete is unavailable."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        trainer = SaeTrainer(sae)
        
        with patch('amber.mechanistic.sae.sae_trainer.train_sae', side_effect=ImportError("No module named 'overcomplete'")):
            with tempfile.TemporaryDirectory() as tmpdir:
                store = create_temp_store(tmpdir)
                with pytest.raises(ImportError, match="overcomplete.sae.train module not available"):
                    trainer.train(store, "run_1", "layer_0")

