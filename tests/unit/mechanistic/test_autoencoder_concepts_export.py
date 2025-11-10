"""Tests for AutoencoderConcepts export methods."""
import pytest
import torch
import csv
from pathlib import Path
import tempfile

from amber.mechanistic.autoencoder.autoencoder import Autoencoder
from amber.mechanistic.autoencoder.concepts.autoencoder_concepts import AutoencoderConcepts


class _FakeLM:
    def __init__(self):
        self._layers = type('obj', (object,), {})()
        self._input_tracker = None

    def get_input_tracker(self):
        return self._input_tracker
    
    def _ensure_input_tracker(self):
        if self._input_tracker is None:
            from amber.mechanistic.autoencoder.concepts.input_tracker import InputTracker
            self._input_tracker = InputTracker(self)
        return self._input_tracker


def test_autoencoder_concepts_export_top_texts_to_csv(tmp_path):
    """Test export_top_texts_to_csv method."""
    autoencoder = Autoencoder(n_latents=4, n_inputs=10)
    concepts = AutoencoderConcepts(autoencoder.context)
    
    lm = _FakeLM()
    autoencoder.context.lm = lm
    autoencoder.context.text_tracking_enabled = True
    autoencoder.context.text_tracking_k = 3
    concepts.enable_text_tracking()
    
    # Add some texts
    latents = torch.randn(2, 4)
    texts = ["text1", "text2"]
    concepts.update_top_texts_from_latents(latents, texts, original_shape=(2, 4))
    
    # Export to CSV
    csv_path = tmp_path / "top_texts.csv"
    result_path = concepts.export_top_texts_to_csv(csv_path)
    
    assert result_path == csv_path
    assert csv_path.exists()
    
    # Verify CSV content
    with csv_path.open('r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) > 1  # Header + at least one data row
        assert rows[0] == ["neuron_idx", "text", "score", "token_str", "token_idx"]


def test_autoencoder_concepts_export_top_texts_to_csv_no_texts():
    """Test export_top_texts_to_csv raises error when no texts available."""
    autoencoder = Autoencoder(n_latents=4, n_inputs=10)
    concepts = AutoencoderConcepts(autoencoder.context)
    
    # No texts tracked
    with pytest.raises(ValueError, match="No top texts available"):
        concepts.export_top_texts_to_csv("/tmp/test.csv")

