"""Tests that verify actual behavior of AutoencoderConcepts, not just compilation."""
import pytest
import torch
import json
import csv
from pathlib import Path

from amber.mechanistic.autoencoder.concepts.autoencoder_concepts import AutoencoderConcepts
from amber.mechanistic.autoencoder.autoencoder_context import AutoencoderContext
from amber.mechanistic.autoencoder.autoencoder import Autoencoder


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


class TestAutoencoderConceptsUpdateTopTexts:
    """Test update_top_texts_from_latents behavior."""
    
    def test_update_top_texts_updates_existing_if_better(self):
        """Verify existing text entry is updated if new activation is better."""
        lm = _FakeLM()
        autoencoder = Autoencoder(n_latents=3, n_inputs=10)
        autoencoder.context.lm = lm
        autoencoder.context.lm_layer_signature = "sig"
        concepts = AutoencoderConcepts(autoencoder.context)
        
        autoencoder.context.text_tracking_enabled = True
        autoencoder.context.text_tracking_k = 5
        autoencoder.context.text_tracking_negative = False
        concepts.enable_text_tracking()
        
        text = "same text"
        lm.get_input_tracker().set_current_texts([text])
        
        # First update with score 1.0
        tens1 = torch.tensor([[[1.0, 0.0, 0.0]]])  # [B=1, T=1, D=3]
        flat1 = tens1.reshape(-1, 3)
        concepts.update_top_texts_from_latents(flat1, [text], original_shape=tens1.shape)
        
        # Verify text is in heap with score 1.0
        tops = concepts.get_top_texts_for_neuron(0)
        assert len(tops) == 1
        assert tops[0].text == text
        assert tops[0].score == pytest.approx(1.0)
        
        # Second update with same text but better score 2.0
        tens2 = torch.tensor([[[2.0, 0.0, 0.0]]])  # [B=1, T=1, D=3]
        flat2 = tens2.reshape(-1, 3)
        concepts.update_top_texts_from_latents(flat2, [text], original_shape=tens2.shape)
        
        # Verify only one entry exists with better score
        tops = concepts.get_top_texts_for_neuron(0)
        assert len(tops) == 1, "Should have only one entry (updated, not duplicated)"
        assert tops[0].text == text
        assert tops[0].score == pytest.approx(2.0), "Score should be updated to 2.0"

    def test_update_top_texts_skips_zero_scores(self):
        """Verify zero scores are not added to heap."""
        lm = _FakeLM()
        autoencoder = Autoencoder(n_latents=2, n_inputs=10)
        autoencoder.context.lm = lm
        autoencoder.context.lm_layer_signature = "sig"
        concepts = AutoencoderConcepts(autoencoder.context)
        
        autoencoder.context.text_tracking_enabled = True
        autoencoder.context.text_tracking_k = 5
        concepts.enable_text_tracking()
        
        text = "test text"
        lm.get_input_tracker().set_current_texts([text])
        
        # Update with all-zero activations
        tens = torch.zeros(1, 1, 2)  # [B=1, T=1, D=2]
        flat = tens.reshape(-1, 2)
        concepts.update_top_texts_from_latents(flat, [text], original_shape=tens.shape)
        
        # Verify heap remains empty
        tops0 = concepts.get_top_texts_for_neuron(0)
        tops1 = concepts.get_top_texts_for_neuron(1)
        assert len(tops0) == 0, "Zero scores should not be added"
        assert len(tops1) == 0, "Zero scores should not be added"

    def test_update_top_texts_respects_k_limit(self):
        """Verify heap doesn't exceed k limit."""
        lm = _FakeLM()
        autoencoder = Autoencoder(n_latents=1, n_inputs=10)
        autoencoder.context.lm = lm
        autoencoder.context.lm_layer_signature = "sig"
        concepts = AutoencoderConcepts(autoencoder.context)
        
        k = 3
        autoencoder.context.text_tracking_enabled = True
        autoencoder.context.text_tracking_k = k
        concepts.enable_text_tracking()
        
        # Add k+1 different texts
        for i in range(k + 1):
            text = f"text_{i}"
            lm.get_input_tracker().set_current_texts([text])
            # Use decreasing scores so later ones are worse
            score = 10.0 - i
            tens = torch.tensor([[[score]]])  # [B=1, T=1, D=1]
            flat = tens.reshape(-1, 1)
            concepts.update_top_texts_from_latents(flat, [text], original_shape=tens.shape)
        
        # Verify only k entries in heap
        tops = concepts.get_top_texts_for_neuron(0)
        assert len(tops) == k, f"Should have exactly {k} entries, got {len(tops)}"
        
        # Verify lowest-scoring entry was removed (text_3 with score 7.0 should be gone)
        texts = [t.text for t in tops]
        assert "text_3" not in texts, "Lowest-scoring text should be removed"
        assert "text_0" in texts, "Highest-scoring text should be kept"
        assert "text_1" in texts, "Second-highest text should be kept"
        assert "text_2" in texts, "Third-highest text should be kept"

    def test_update_top_texts_keeps_existing_if_worse(self):
        """Verify existing text entry is NOT updated if new activation is worse."""
        lm = _FakeLM()
        autoencoder = Autoencoder(n_latents=1, n_inputs=10)
        autoencoder.context.lm = lm
        autoencoder.context.lm_layer_signature = "sig"
        concepts = AutoencoderConcepts(autoencoder.context)
        
        autoencoder.context.text_tracking_enabled = True
        autoencoder.context.text_tracking_k = 5
        concepts.enable_text_tracking()
        
        text = "same text"
        lm.get_input_tracker().set_current_texts([text])
        
        # First update with score 2.0
        tens1 = torch.tensor([[[2.0]]])  # [B=1, T=1, D=1]
        flat1 = tens1.reshape(-1, 1)
        concepts.update_top_texts_from_latents(flat1, [text], original_shape=tens1.shape)
        
        # Second update with same text but worse score 1.0
        tens2 = torch.tensor([[[1.0]]])  # [B=1, T=1, D=1]
        flat2 = tens2.reshape(-1, 1)
        concepts.update_top_texts_from_latents(flat2, [text], original_shape=tens2.shape)
        
        # Verify score is still 2.0 (not updated to worse)
        tops = concepts.get_top_texts_for_neuron(0)
        assert len(tops) == 1
        assert tops[0].score == pytest.approx(2.0), "Score should remain 2.0 (not updated to worse)"


class TestAutoencoderConceptsExport:
    """Test export functionality."""
    
    def test_export_json_valid_and_parseable(self, tmp_path):
        """Verify exported JSON is valid and parseable."""
        lm = _FakeLM()
        autoencoder = Autoencoder(n_latents=2, n_inputs=10)
        autoencoder.context.lm = lm
        autoencoder.context.lm_layer_signature = "sig"
        concepts = AutoencoderConcepts(autoencoder.context)
        
        autoencoder.context.text_tracking_enabled = True
        autoencoder.context.text_tracking_k = 5
        concepts.enable_text_tracking()
        
        # Add some texts
        texts = ["text1", "text2"]
        lm.get_input_tracker().set_current_texts(texts)
        tens = torch.tensor([
            [[1.0, 0.5], [0.8, 0.3]],  # text1: max per neuron = [1.0, 0.5]
            [[0.9, 0.7], [0.6, 0.4]],  # text2: max per neuron = [0.9, 0.7]
        ])
        flat = tens.reshape(-1, 2)
        concepts.update_top_texts_from_latents(flat, texts, original_shape=tens.shape)
        
        # Export to JSON
        json_path = tmp_path / "top_texts.json"
        concepts.export_top_texts_to_json(json_path)
        
        # Verify file exists
        assert json_path.exists()
        
        # Verify JSON is valid and parseable
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Verify structure
        assert isinstance(data, dict)
        assert "0" in data  # neuron 0
        assert "1" in data  # neuron 1
        
        # Verify required fields
        for neuron_idx, texts_list in data.items():
            assert isinstance(texts_list, list)
            for entry in texts_list:
                assert "text" in entry
                assert "score" in entry
                assert "token_str" in entry
                assert "token_idx" in entry
                assert isinstance(entry["text"], str)
                assert isinstance(entry["score"], (int, float))
                assert isinstance(entry["token_str"], str)
                assert isinstance(entry["token_idx"], int)

    def test_export_csv_correct_columns(self, tmp_path):
        """Verify exported CSV has correct columns."""
        lm = _FakeLM()
        autoencoder = Autoencoder(n_latents=2, n_inputs=10)
        autoencoder.context.lm = lm
        autoencoder.context.lm_layer_signature = "sig"
        concepts = AutoencoderConcepts(autoencoder.context)
        
        autoencoder.context.text_tracking_enabled = True
        autoencoder.context.text_tracking_k = 5
        concepts.enable_text_tracking()
        
        # Add some texts
        texts = ["text1", "text2"]
        lm.get_input_tracker().set_current_texts(texts)
        tens = torch.tensor([
            [[1.0, 0.5], [0.8, 0.3]],
            [[0.9, 0.7], [0.6, 0.4]],
        ])
        flat = tens.reshape(-1, 2)
        concepts.update_top_texts_from_latents(flat, texts, original_shape=tens.shape)
        
        # Export to CSV
        csv_path = tmp_path / "top_texts.csv"
        concepts.export_top_texts_to_csv(csv_path)
        
        # Verify file exists
        assert csv_path.exists()
        
        # Read CSV and verify columns
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            assert columns == ["neuron_idx", "text", "score", "token_str", "token_idx"]
            
            # Verify data rows
            rows = list(reader)
            assert len(rows) > 0
            for row in rows:
                assert "neuron_idx" in row
                assert "text" in row
                assert "score" in row
                assert "token_str" in row
                assert "token_idx" in row


class TestAutoencoderConceptsManipulateConcept:
    """Test manipulate_concept parameter updates."""
    
    def test_manipulate_concept_updates_parameters(self):
        """Verify manipulate_concept actually updates parameter values."""
        autoencoder = Autoencoder(n_latents=3, n_inputs=10)
        concepts = AutoencoderConcepts(autoencoder.context)
        
        # Default values should be 1.0
        assert torch.allclose(concepts.multiplication.data, torch.ones(3))
        assert torch.allclose(concepts.bias.data, torch.ones(3))
        
        # Update parameters
        concepts.manipulate_concept(neuron_idx=0, multiplier=2.0, bias=0.5)
        concepts.manipulate_concept(neuron_idx=1, multiplier=1.5, bias=-0.3)
        
        # Verify values are updated
        assert concepts.multiplication.data[0] == pytest.approx(2.0)
        assert concepts.bias.data[0] == pytest.approx(0.5)
        assert concepts.multiplication.data[1] == pytest.approx(1.5)
        assert concepts.bias.data[1] == pytest.approx(-0.3)
        
        # Verify other neurons unchanged
        assert concepts.multiplication.data[2] == pytest.approx(1.0)
        assert concepts.bias.data[2] == pytest.approx(1.0)

