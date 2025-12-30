import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mi_crow.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary, Concept
from mi_crow.mechanistic.sae.concepts.concept_models import NeuronText


def test_add_and_get_enforce_bounds():
    dictionary = ConceptDictionary(n_size=2)
    dictionary.add(0, "alpha", 0.9)
    assert dictionary.get(0) == Concept(name="alpha", score=0.9)

    with pytest.raises(IndexError):
        dictionary.add(5, "beta", 0.5)
    with pytest.raises(IndexError):
        dictionary.get(-1)


def test_save_and_load_roundtrip(tmp_path):
    dictionary = ConceptDictionary(n_size=3)
    dictionary.add(2, "gamma", 0.8)

    save_path = dictionary.save(directory=tmp_path / "concepts")
    assert save_path.exists()

    new_dict = ConceptDictionary(n_size=1)
    new_dict.load(directory=tmp_path / "concepts")
    assert new_dict.n_size == 3
    assert new_dict.get(2).name == "gamma"


def test_save_without_directory_raises():
    dictionary = ConceptDictionary(n_size=1)
    with pytest.raises(ValueError):
        dictionary.save()


def test_from_csv_picks_best_score(tmp_path):
    csv_path = tmp_path / "concepts.csv"
    csv_path.write_text(
        "neuron_idx,concept_name,score\n0,a,0.1\n0,b,0.8\n1,c,0.5\n",
        encoding="utf-8",
    )

    dictionary = ConceptDictionary.from_csv(csv_path, n_size=2)

    assert dictionary.get(0).name == "b"
    assert dictionary.get(1).name == "c"


def test_from_json_supports_old_and_new_formats(tmp_path):
    json_path = tmp_path / "concepts.json"
    json_path.write_text(
        json.dumps({
            "0": [{"name": "old", "score": 0.2}, {"name": "better", "score": 0.9}],
            "1": {"name": "single", "score": 0.5},
        }),
        encoding="utf-8",
    )

    dictionary = ConceptDictionary.from_json(json_path, n_size=2)
    assert dictionary.get(0).name == "better"
    assert dictionary.get(1).name == "single"


def test_from_llm_uses_generated_names(monkeypatch):
    neuron_texts = [
        [NeuronText(score=1.0, text="text", token_idx=0, token_str="token")],
        [],
    ]

    with patch.object(
        ConceptDictionary,
        "_generate_concept_names_llm",
        return_value=[("llm_concept", 0.7)],
    ) as mock_llm:
        dictionary = ConceptDictionary.from_llm(neuron_texts, n_size=2)

    mock_llm.assert_called_once()
    assert dictionary.get(0).name == "llm_concept"
    assert dictionary.get(1) is None

