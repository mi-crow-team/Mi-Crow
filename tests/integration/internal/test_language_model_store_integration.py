"""Integration tests for LanguageModel with Store."""

import pytest
import torch

from tests.unit.fixtures.language_models import create_language_model_from_mock
from tests.unit.fixtures.stores import create_temp_store
from tests.unit.fixtures.hooks import create_activation_detector


class TestLanguageModelStoreIntegration:
    """Tests for LanguageModel with Store."""

    def test_activation_saving_to_store(self, temp_store):
        """Test that activations are saved to store."""
        lm = create_language_model_from_mock(temp_store)
        detector = create_activation_detector(layer_signature=0)
        
        lm.layers.register_hook(0, detector)
        lm.forwards(["Hello world"])
        
        # Check that activations were captured
        captured = detector.get_captured()
        assert captured is not None
        assert isinstance(captured, torch.Tensor)

    def test_metadata_persistence(self, temp_store):
        """Test metadata persistence in store."""
        run_id = "test_run"
        batch_index = 0
        
        metadata = {"test_key": "test_value"}
        tensor_metadata = {
            "layer_0": {"activations": torch.randn(2, 10)}
        }
        
        temp_store.put_detector_metadata(run_id, batch_index, metadata, tensor_metadata)
        
        retrieved_metadata, retrieved_tensors = temp_store.get_detector_metadata(run_id, batch_index)
        
        assert retrieved_metadata["test_key"] == "test_value"
        assert "layer_0" in retrieved_tensors
        assert "activations" in retrieved_tensors["layer_0"]

    def test_run_metadata_management(self, temp_store):
        """Test run metadata management."""
        run_id = "test_run"
        metadata = {"model_id": "test_model", "dataset": "test_dataset"}
        
        temp_store.put_run_metadata(run_id, metadata)
        retrieved = temp_store.get_run_metadata(run_id)
        
        assert retrieved == metadata

    def test_unified_detector_metadata_saving_via_language_model(self, temp_store):
        """Test unified detector metadata saving path."""
        lm = create_language_model_from_mock(temp_store)
        detector = create_activation_detector(layer_signature=0)

        lm.layers.register_hook(0, detector)
        lm.inference.infer_texts(["Hello unified"], run_name="unified_run", save_in_batches=False)

        base = temp_store.base_path / temp_store.runs_prefix / "unified_run" / "detectors"
        assert (base / "metadata.json").exists()

