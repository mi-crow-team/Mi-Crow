"""Additional tests for AutoencoderConcepts to improve coverage."""
import pytest
import torch

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


def test_autoencoder_concepts_load_concepts_from_csv():
    """Test load_concepts_from_csv method."""
    import tempfile
    import csv
    
    autoencoder = Autoencoder(n_latents=4, n_inputs=10)
    concepts = AutoencoderConcepts(autoencoder.context)
    
    # Create a CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(["neuron_idx", "concept_name", "score"])
        writer.writerow([0, "concept_0", "0.8"])
        writer.writerow([1, "concept_1", "0.6"])
        csv_path = f.name
    
    try:
        concepts.load_concepts_from_csv(csv_path)
        assert concepts.dictionary is not None
        concept = concepts.dictionary.get(0)
        assert concept is not None
        assert concept.name == "concept_0"
    finally:
        import os
        os.unlink(csv_path)


def test_autoencoder_concepts_load_concepts_from_json():
    """Test load_concepts_from_json method."""
    import tempfile
    import json
    
    autoencoder = Autoencoder(n_latents=4, n_inputs=10)
    concepts = AutoencoderConcepts(autoencoder.context)
    
    # Create a JSON file
    json_data = {
        "0": [{"name": "concept_0", "score": 0.8}],
        "1": [{"name": "concept_1", "score": 0.6}]
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_data, f)
        json_path = f.name
    
    try:
        concepts.load_concepts_from_json(json_path)
        assert concepts.dictionary is not None
        concept = concepts.dictionary.get(0)
        assert concept is not None
        assert concept.name == "concept_0"
    finally:
        import os
        os.unlink(json_path)


def test_autoencoder_concepts_generate_concepts_with_llm_no_texts():
    """Test generate_concepts_with_llm raises error when no texts available."""
    autoencoder = Autoencoder(n_latents=4, n_inputs=10)
    concepts = AutoencoderConcepts(autoencoder.context)
    
    # No texts tracked yet
    with pytest.raises(ValueError, match="No top texts available"):
        concepts.generate_concepts_with_llm()


def test_autoencoder_concepts_ensure_dictionary():
    """Test _ensure_dictionary creates dictionary if None."""
    autoencoder = Autoencoder(n_latents=4, n_inputs=10)
    concepts = AutoencoderConcepts(autoencoder.context)
    
    assert concepts.dictionary is None
    
    # Accessing dictionary should create it
    dict_ref = concepts._ensure_dictionary()
    assert dict_ref is not None
    assert concepts.dictionary is not None

