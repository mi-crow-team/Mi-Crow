"""Additional tests to improve coverage for sae.py."""
import pytest
import torch

from amber.mechanistic.sae.modules.topk_sae import TopKSae
from amber.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary


def test_sae_attach_dictionary():
    """Test attach_dictionary method (lines 84-85)."""
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4)
    
    # Create a concept dictionary
    concept_dict = ConceptDictionary(n_size=8)
    concept_dict.add(0, "concept1", 0.5)
    concept_dict.add(1, "concept2", 0.8)
    
    # Attach dictionary
    topk_sae.attach_dictionary(concept_dict)
    
    # Verify dictionary is attached
    assert topk_sae.concepts.dictionary is concept_dict
    assert topk_sae.concepts.dictionary.get(0) is not None
    assert topk_sae.concepts.dictionary.get(0).name == "concept1"

