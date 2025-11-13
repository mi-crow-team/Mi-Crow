import json
from pathlib import Path
import pytest

from amber.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary, Concept


def test_add_and_get_enforces_one_concept_per_neuron(tmp_path):
    cd = ConceptDictionary(n_size=5)
    # Only 1 concept per neuron is allowed - adding a new one replaces the old one
    cd.add(1, "a", 0.1)
    got = cd.get(1)
    assert got is not None
    assert got.name == "a"
    assert got.score == 0.1
    
    # Adding a new concept with higher score replaces the old one
    cd.add(1, "b", 0.9)
    got = cd.get(1)
    assert got is not None
    assert got.name == "b"
    assert got.score == 0.9
    
    # Adding a new concept with lower score still replaces (only 1 allowed)
    cd.add(1, "c", 0.5)
    got = cd.get(1)
    assert got is not None
    assert got.name == "c"
    assert got.score == 0.5

    # Out-of-bounds add/get raise IndexError
    with pytest.raises(IndexError):
        cd.add(10, "x", 1.0)
    with pytest.raises(IndexError):
        cd.get(10)

    # get_many returns mapping for requested indices
    many = cd.get_many([0, 1, 2])
    assert set(many.keys()) == {0, 1, 2}
    assert many[0] is None  # No concept for neuron 0
    assert many[1] is not None  # Has concept "c"
    assert many[2] is None  # No concept for neuron 2


def test_save_and_load_roundtrip(tmp_path):
    base = tmp_path / "concepts_dir"
    cd = ConceptDictionary(n_size=3)
    cd.add(0, "x", 0.2)
    cd.add(2, "y", 0.8)

    path = cd.save(directory=base)
    assert path.exists()

    cd2 = ConceptDictionary(n_size=0)
    cd2.load(directory=base)
    assert cd2.n_size == 3
    concept = cd2.get(2)
    assert concept is not None
    assert concept.name == "y"


def test_save_load_errors_and_from_directory_behaviors(tmp_path):
    base = tmp_path / "empty_dir"

    # load without directory set raises
    cd = ConceptDictionary(n_size=1)
    with pytest.raises(ValueError):
        cd.load()

    # save without directory set raises
    cd2 = ConceptDictionary(n_size=1)
    with pytest.raises(ValueError):
        cd2.save()

    # from_directory on non-existing path raises
    with pytest.raises(FileNotFoundError):
        cd_temp = ConceptDictionary(n_size=0)
        cd_temp.load(directory=base)

    # Create directory without concepts.json -> load raises FileNotFoundError
    base.mkdir(parents=True, exist_ok=True)
    cd3 = ConceptDictionary(n_size=0)
    with pytest.raises(FileNotFoundError):
        cd3.load(directory=base)

    # Write a minimal concepts.json and ensure load parses it
    meta = {
        "n_size": 4,
        "concepts": {"1": [{"name": "z", "score": 0.7}]},
    }
    (base / "concepts.json").write_text(json.dumps(meta), encoding="utf-8")
    cd4 = ConceptDictionary(n_size=0)
    cd4.load(directory=base)
    assert cd4.n_size == 4
    concept = cd4.get(1)
    assert concept is not None
    assert isinstance(concept, Concept)
    assert concept.name == "z"
    assert concept.score == 0.7
