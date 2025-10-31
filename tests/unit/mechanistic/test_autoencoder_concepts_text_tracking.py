import logging
from pathlib import Path
import pytest

from amber.mechanistic.autoencoder.concepts.autoencoder_concepts import AutoencoderConcepts
from amber.mechanistic.autoencoder.autoencoder import Autoencoder


def test_enable_text_tracking_requires_lm_and_signature():
    autoencoder = Autoencoder(n_latents=3, n_inputs=10)
    ac = AutoencoderConcepts(autoencoder.context)
    with pytest.raises(ValueError):
        ac.enable_text_tracking()


def test_multiply_concept_warns_without_dictionary(caplog):
    autoencoder = Autoencoder(n_latents=3, n_inputs=10)
    ac = AutoencoderConcepts(autoencoder.context)
    with caplog.at_level(logging.WARNING):
        ac.manipulate_concept(1, multiplier=2.0, bias=0.5)
    assert any("No dictionary was created yet" in rec.message for rec in caplog.records)


def test_ensure_dictionary_creates_or_loads(tmp_path):
    # Without path -> creates new ConceptDictionary
    autoencoder1 = Autoencoder(n_latents=4, n_inputs=10)
    ac1 = AutoencoderConcepts(autoencoder1.context)
    d1 = ac1._ensure_dictionary()
    assert d1.n_size == 4

    # With path -> from_directory (which creates empty when no file)
    p = tmp_path / "dict"
    p.mkdir(parents=True)
    autoencoder2 = Autoencoder(n_latents=5, n_inputs=10)
    ac2 = AutoencoderConcepts(autoencoder2.context)
    d2 = ac2._ensure_dictionary()
    assert d2.n_size == 5  # Should be 5 since we set n_latents=5


def test_disable_text_tracking_is_safe_when_no_tracker():
    autoencoder = Autoencoder(n_latents=2, n_inputs=10)
    ac = AutoencoderConcepts(autoencoder.context)
    # Should be no-op
    ac.disable_text_tracking()
