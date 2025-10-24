import json
from pathlib import Path
import pytest

from amber.mechanistic.autoencoder.concepts.concept_dictionary import ConceptDictionary, Concept


def test_add_and_get_and_max_concepts_enforces_topk(tmp_path):
    cd = ConceptDictionary(n_size=5, max_concepts=2)
    # Add 3 concepts to index 1; only top-2 by score are kept
    cd.add(1, "a", 0.1)
    cd.add(1, "b", 0.9)
    cd.add(1, "c", 0.5)
    got = cd.get(1)
    assert [c.name for c in got] == ["b", "c"]

    # Out-of-bounds add/get raise IndexError
    with pytest.raises(IndexError):
        cd.add(10, "x", 1.0)
    with pytest.raises(IndexError):
        cd.get(10)

    # get_many returns mapping for requested indices
    many = cd.get_many([0, 1, 2])
    assert set(many.keys()) == {0, 1, 2}
    assert many[0] == [] and isinstance(many[1], list)


def test_save_and_load_roundtrip(tmp_path):
    base = tmp_path / "concepts_dir"
    cd = ConceptDictionary(n_size=3, max_concepts=3)
    cd.add(0, "x", 0.2)
    cd.add(2, "y", 0.8)

    path = cd.save(directory=base)
    assert path.exists()

    cd2 = ConceptDictionary(n_size=0)
    cd2.load(directory=base)
    assert cd2.n_size == 3
    assert [c.name for c in cd2.get(2)] == ["y"]


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
        ConceptDictionary.from_directory(base)

    # Create directory without concepts.json -> returns empty dictionary with n_size=0
    base.mkdir(parents=True)
    cd3 = ConceptDictionary.from_directory(base)
    assert cd3.n_size == 0
    assert cd3.get_many([]) == {}

    # Write a minimal concepts.json and ensure from_directory parses it
    meta = {
        "n_size": 4,
        "max_concepts": 2,
        "concepts": {"1": [{"name": "z", "score": 0.7}]},
    }
    (base / "concepts.json").write_text(json.dumps(meta), encoding="utf-8")
    cd4 = ConceptDictionary.from_directory(base)
    assert cd4.n_size == 4 and cd4.max_concepts == 2
    assert isinstance(cd4.get(1)[0], Concept)
