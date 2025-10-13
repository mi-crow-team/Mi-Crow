import logging
from pathlib import Path
import pytest

from amber.mechanistic.autoencoder.concepts.autoencoder_concepts import AutoencoderConcepts


def test_enable_text_tracking_requires_lm_and_signature():
    ac = AutoencoderConcepts(n_size=3)
    with pytest.raises(ValueError):
        ac.enable_text_tracking()


def test_multiply_concept_warns_without_dictionary(caplog):
    ac = AutoencoderConcepts(n_size=3)
    with caplog.at_level(logging.WARNING):
        ac.multiply_concept(1, 2.0)
    assert any("No dictionary was created yet" in rec.message for rec in caplog.records)


def test_ensure_dictionary_creates_or_loads(tmp_path):
    # Without path -> creates new ConceptDictionary
    ac1 = AutoencoderConcepts(n_size=4)
    d1 = ac1._ensure_dictionary()
    assert d1.n_size == 4

    # With path -> from_directory (which creates empty when no file)
    p = tmp_path / "dict"
    p.mkdir(parents=True)
    ac2 = AutoencoderConcepts(n_size=5, dictionary_path=p)
    d2 = ac2._ensure_dictionary()
    assert d2.n_size == 0  # from_directory on empty dir sets n_size=0 per implementation


def test_disable_text_tracking_is_safe_when_no_tracker():
    ac = AutoencoderConcepts(n_size=2)
    # Should be no-op
    ac.disable_text_tracking()
