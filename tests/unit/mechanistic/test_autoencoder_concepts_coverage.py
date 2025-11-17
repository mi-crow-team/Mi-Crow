import pytest
import torch
import json
from pathlib import Path
from unittest.mock import Mock, patch

from amber.mechanistic.sae.concepts.autoencoder_concepts import AutoencoderConcepts, NeuronText
from amber.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary
from amber.mechanistic.sae.autoencoder_context import AutoencoderContext


@pytest.fixture
def mock_context():
    """Create a mock AutoencoderContext for testing."""
    mock_autoencoder = Mock()
    mock_autoencoder._text_tracking_enabled = False
    context = AutoencoderContext(
        autoencoder=mock_autoencoder,
        n_latents=10,
        n_inputs=16,
        device="cpu"
    )
    context.lm = Mock()
    context.lm.tokenizer = Mock()
    return context


@pytest.fixture
def autoencoder_concepts(mock_context):
    """Create AutoencoderConcepts instance."""
    concepts = AutoencoderConcepts(context=mock_context)
    concepts._text_tracking_k = 3
    concepts._text_tracking_negative = False
    return concepts


def test_update_top_texts_from_latents_3d_shape(autoencoder_concepts):
    """Test update_top_texts_from_latents with 3D original shape."""
    latents = torch.randn(6, 10)  # B*T=6, n_latents=10
    texts = ["text1", "text2", "text3"]
    original_shape = (3, 2, 5)  # B=3, T=2, D=5
    
    autoencoder_concepts.update_top_texts_from_latents(latents, texts, original_shape)
    
    # Should have initialized heaps
    assert autoencoder_concepts._top_texts_heaps is not None
    assert len(autoencoder_concepts._top_texts_heaps) == 10


def test_update_top_texts_from_latents_2d_shape(autoencoder_concepts):
    """Test update_top_texts_from_latents with 2D original shape."""
    latents = torch.randn(3, 10)  # B=3, n_latents=10
    texts = ["text1", "text2", "text3"]
    original_shape = (3, 5)  # B=3, D=5
    
    autoencoder_concepts.update_top_texts_from_latents(latents, texts, original_shape)
    
    assert autoencoder_concepts._top_texts_heaps is not None


def test_update_top_texts_from_latents_no_shape(autoencoder_concepts):
    """Test update_top_texts_from_latents without original_shape."""
    latents = torch.randn(3, 10)
    texts = ["text1", "text2", "text3"]
    
    autoencoder_concepts.update_top_texts_from_latents(latents, texts, None)
    
    assert autoencoder_concepts._top_texts_heaps is not None


def test_update_top_texts_from_latents_empty_texts(autoencoder_concepts):
    """Test update_top_texts_from_latents with empty texts."""
    latents = torch.randn(3, 10)
    texts = []
    
    autoencoder_concepts.update_top_texts_from_latents(latents, texts)
    
    # Should not crash, heaps may or may not be initialized


def test_update_top_texts_from_latents_batch_mismatch(autoencoder_concepts, caplog):
    """Test update_top_texts_from_latents with batch size mismatch."""
    latents = torch.randn(6, 10)  # B*T=6
    texts = ["text1", "text2"]  # Only 2 texts, but shape says B=3
    original_shape = (3, 2, 5)  # B=3
    
    autoencoder_concepts.update_top_texts_from_latents(latents, texts, original_shape)
    
    # Should log warning but continue
    assert len(caplog.records) > 0


def test_update_top_texts_from_latents_negative_tracking(autoencoder_concepts):
    """Test update_top_texts_from_latents with negative tracking enabled."""
    autoencoder_concepts._text_tracking_negative = True
    latents = torch.randn(3, 10)
    texts = ["text1", "text2", "text3"]
    
    autoencoder_concepts.update_top_texts_from_latents(latents, texts)
    
    # Should track minimum (most negative) activations
    assert autoencoder_concepts._top_texts_heaps is not None


def test_update_top_texts_from_latents_zero_score_skip(autoencoder_concepts):
    """Test update_top_texts_from_latents skips zero scores."""
    latents = torch.zeros(3, 10)
    texts = ["text1", "text2", "text3"]
    
    autoencoder_concepts.update_top_texts_from_latents(latents, texts)
    
    # Heaps should be initialized but empty (all zeros skipped)
    assert autoencoder_concepts._top_texts_heaps is not None


def test_update_top_texts_from_latents_existing_text_update(autoencoder_concepts):
    """Test update_top_texts_from_latents updates existing text if better."""
    latents1 = torch.randn(3, 10)
    latents1[0, 0] = 0.5  # Small activation
    texts = ["text1", "text2", "text3"]
    
    autoencoder_concepts.update_top_texts_from_latents(latents1, texts)
    
    # Update with better activation
    latents2 = torch.randn(3, 10)
    latents2[0, 0] = 0.9  # Better activation for same text
    
    autoencoder_concepts.update_top_texts_from_latents(latents2, texts)
    
    # Should have updated the entry
    heap = autoencoder_concepts._top_texts_heaps[0]
    assert len(heap) > 0


def test_get_top_texts_for_neuron(autoencoder_concepts):
    """Test get_top_texts_for_neuron."""
    latents = torch.randn(3, 10)
    texts = ["text1", "text2", "text3"]
    autoencoder_concepts.update_top_texts_from_latents(latents, texts)
    
    texts_list = autoencoder_concepts.get_top_texts_for_neuron(0)
    assert isinstance(texts_list, list)
    assert all(isinstance(nt, NeuronText) for nt in texts_list)


def test_get_top_texts_for_neuron_with_limit(autoencoder_concepts):
    """Test get_top_texts_for_neuron with top_m limit."""
    latents = torch.randn(3, 10)
    texts = ["text1", "text2", "text3"]
    autoencoder_concepts.update_top_texts_from_latents(latents, texts)
    
    texts_list = autoencoder_concepts.get_top_texts_for_neuron(0, top_m=2)
    assert len(texts_list) <= 2


def test_get_top_texts_for_neuron_invalid_index(autoencoder_concepts):
    """Test get_top_texts_for_neuron with invalid index."""
    result = autoencoder_concepts.get_top_texts_for_neuron(-1)
    assert result == []
    
    result = autoencoder_concepts.get_top_texts_for_neuron(100)
    assert result == []


def test_get_all_top_texts(autoencoder_concepts):
    """Test get_all_top_texts."""
    latents = torch.randn(3, 10)
    texts = ["text1", "text2", "text3"]
    autoencoder_concepts.update_top_texts_from_latents(latents, texts)
    
    all_texts = autoencoder_concepts.get_all_top_texts()
    assert len(all_texts) == 10  # One list per neuron
    assert all(isinstance(neuron_texts, list) for neuron_texts in all_texts)


def test_get_all_top_texts_no_heaps(autoencoder_concepts):
    """Test get_all_top_texts when heaps are None."""
    result = autoencoder_concepts.get_all_top_texts()
    assert result == []


def test_reset_top_texts(autoencoder_concepts):
    """Test reset_top_texts."""
    latents = torch.randn(3, 10)
    texts = ["text1", "text2", "text3"]
    autoencoder_concepts.update_top_texts_from_latents(latents, texts)
    
    assert autoencoder_concepts._top_texts_heaps is not None
    autoencoder_concepts.reset_top_texts()
    assert autoencoder_concepts._top_texts_heaps is None


def test_export_top_texts_to_json(autoencoder_concepts, tmp_path):
    """Test export_top_texts_to_json."""
    latents = torch.randn(3, 10)
    texts = ["text1", "text2", "text3"]
    autoencoder_concepts.update_top_texts_from_latents(latents, texts)
    
    filepath = tmp_path / "export.json"
    result_path = autoencoder_concepts.export_top_texts_to_json(filepath)
    
    assert result_path == filepath
    assert filepath.exists()
    
    # Verify JSON structure
    with filepath.open() as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert "0" in data  # Neuron index as string key


def test_export_top_texts_to_json_no_heaps(autoencoder_concepts, tmp_path):
    """Test export_top_texts_to_json raises error when no heaps."""
    filepath = tmp_path / "export.json"
    with pytest.raises(ValueError, match="No top texts available"):
        autoencoder_concepts.export_top_texts_to_json(filepath)


def test_decode_token_with_tokenizer(autoencoder_concepts):
    """Test _decode_token with available tokenizer."""
    autoencoder_concepts.context.lm.tokenizer.encode = Mock(return_value=[1, 2, 3])
    autoencoder_concepts.context.lm.tokenizer.decode = Mock(return_value="token")
    
    result = autoencoder_concepts._decode_token("test text", 1)
    assert result == "token"


def test_decode_token_out_of_range(autoencoder_concepts):
    """Test _decode_token with out-of-range token index."""
    autoencoder_concepts.context.lm.tokenizer.encode = Mock(return_value=[1, 2])
    
    result = autoencoder_concepts._decode_token("test", 5)
    assert "out_of_range" in result


def test_decode_token_no_tokenizer(autoencoder_concepts):
    """Test _decode_token without tokenizer."""
    autoencoder_concepts.context.lm = None
    
    result = autoencoder_concepts._decode_token("test", 0)
    assert result.startswith("<token_")


def test_decode_token_decode_error(autoencoder_concepts):
    """Test _decode_token handles decode errors."""
    autoencoder_concepts.context.lm.tokenizer.encode = Mock(side_effect=Exception("Error"))
    
    result = autoencoder_concepts._decode_token("test", 0)
    assert "decode_error" in result


def test_generate_concepts_with_llm_no_heaps(autoencoder_concepts):
    """Test generate_concepts_with_llm raises error when no heaps."""
    with pytest.raises(ValueError, match="No top texts available"):
        autoencoder_concepts.generate_concepts_with_llm()


def test_generate_concepts_with_llm(autoencoder_concepts):
    """Test generate_concepts_with_llm."""
    latents = torch.randn(3, 10)
    texts = ["text1", "text2", "text3"]
    autoencoder_concepts.update_top_texts_from_latents(latents, texts)
    
    with patch("amber.mechanistic.sae.concepts.concept_dictionary.ConceptDictionary") as mock_dict:
        mock_dict_instance = Mock()
        mock_dict.from_llm = Mock(return_value=mock_dict_instance)
        autoencoder_concepts.generate_concepts_with_llm()
        
        assert autoencoder_concepts.dictionary is not None
        assert autoencoder_concepts.dictionary == mock_dict_instance
        mock_dict.from_llm.assert_called_once()


def test_manipulate_concept_with_dictionary(autoencoder_concepts):
    """Test manipulate_concept with dictionary."""
    autoencoder_concepts.dictionary = Mock()
    autoencoder_concepts.multiplication = Mock()
    autoencoder_concepts.multiplication.data = torch.zeros(10)
    autoencoder_concepts.bias = Mock()
    autoencoder_concepts.bias.data = torch.zeros(10)
    
    autoencoder_concepts.manipulate_concept(0, multiplier=2.0, bias=1.0)
    
    assert autoencoder_concepts.multiplication.data[0] == 2.0
    assert autoencoder_concepts.bias.data[0] == 1.0


def test_manipulate_concept_no_dictionary(autoencoder_concepts, caplog):
    """Test manipulate_concept without dictionary logs warning."""
    autoencoder_concepts.dictionary = None
    
    autoencoder_concepts.manipulate_concept(0, multiplier=2.0, bias=0.0)
    
    # Should log warning
    assert len(caplog.records) > 0


def test_ensure_heaps(autoencoder_concepts):
    """Test _ensure_heaps initializes heaps."""
    autoencoder_concepts._ensure_heaps(10)
    assert autoencoder_concepts._top_texts_heaps is not None
    assert len(autoencoder_concepts._top_texts_heaps) == 10


def test_ensure_heaps_already_initialized(autoencoder_concepts):
    """Test _ensure_heaps doesn't reinitialize existing heaps."""
    autoencoder_concepts._ensure_heaps(10)
    original_heaps = autoencoder_concepts._top_texts_heaps
    
    autoencoder_concepts._ensure_heaps(10)
    assert autoencoder_concepts._top_texts_heaps is original_heaps

