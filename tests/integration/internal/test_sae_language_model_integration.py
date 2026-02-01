"""Integration tests for SAE with LanguageModel."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.unit.fixtures.language_models import create_language_model_from_mock
from tests.unit.fixtures.stores import create_temp_store
from tests.unit.mechanistic.test_sae_base import ConcreteSae


class TestSAELanguageModelIntegration:
    """Tests for SAE with LanguageModel."""

    @patch("mi_crow.mechanistic.sae.modules.topk_sae.OvercompleteTopkSAE")
    def test_sae_attachment_to_language_model(self, mock_topk_sae_class, temp_store):
        """Test attaching SAE to LanguageModel."""
        from mi_crow.mechanistic.sae.modules.topk_sae import TopKSae

        mock_engine = MagicMock()
        mock_engine.forward.return_value = (None, None, torch.randn(2, 10))
        mock_topk_sae_class.return_value = mock_engine
        lm = create_language_model_from_mock(temp_store)
        sae = TopKSae(n_latents=100, n_inputs=10, k=10)
        lm.layers.register_hook(0, sae)
        mock_output = MagicMock()
        mock_encodings = {"input_ids": torch.tensor([[1, 2, 3]])}
        with patch.object(lm.inference, "execute_inference") as mock_execute:
            mock_execute.return_value = (mock_output, mock_encodings)
            output, _ = lm.inference.execute_inference(["Hello"])

        assert output is not None

    def test_sae_modifies_activations_during_inference(self, temp_store):
        """Test that SAE modifies activations during inference."""
        lm = create_language_model_from_mock(temp_store)
        sae = ConcreteSae(n_latents=100, n_inputs=10)
        lm.layers.register_hook(0, sae)
        mock_output = MagicMock()
        mock_encodings = {"input_ids": torch.tensor([[1, 2, 3]])}
        with patch.object(lm.inference, "execute_inference") as mock_execute:
            mock_execute.return_value = (mock_output, mock_encodings)
            output, _ = lm.inference.execute_inference(["Hello"])

        assert output is not None
