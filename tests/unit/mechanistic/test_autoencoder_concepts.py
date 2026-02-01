import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from mi_crow.mechanistic.sae.autoencoder_context import AutoencoderContext
from mi_crow.mechanistic.sae.concepts.autoencoder_concepts import AutoencoderConcepts
from mi_crow.mechanistic.sae.concepts.concept_models import NeuronText


class DummyAutoencoder:
    def __init__(self):
        self._text_tracking_enabled = False


def build_context(tmp_lm=None, tracking=False, negative=False):
    lm = tmp_lm
    context = AutoencoderContext(
        autoencoder=DummyAutoencoder(),
        n_latents=3,
        n_inputs=3,
        lm=lm,
        text_tracking_enabled=tracking,
        text_tracking_k=2,
        text_tracking_negative=negative,
    )
    return context


def test_enable_text_tracking_requires_language_model():
    context = build_context(tmp_lm=None, tracking=True)
    concepts = AutoencoderConcepts(context)
    with pytest.raises(ValueError, match="LanguageModel must be set"):
        concepts.enable_text_tracking()


def test_enable_and_disable_text_tracking(monkeypatch):
    class LM:
        def __init__(self):
            self.tracker = Mock()

        def _ensure_input_tracker(self):
            return self.tracker

        tokenizer = None

    lm = LM()
    context = build_context(tmp_lm=lm, tracking=True, negative=True)
    concepts = AutoencoderConcepts(context)
    concepts.enable_text_tracking()
    assert concepts._text_tracking_k == context.text_tracking_k
    assert concepts._text_tracking_negative is True
    assert context.autoencoder._text_tracking_enabled is True
    lm.tracker.enable.assert_called_once()
    concepts.disable_text_tracking()
    assert context.autoencoder._text_tracking_enabled is False


def test_update_top_texts_and_getters(tmp_path):
    lm = Mock()
    lm.tokenizer = None
    context = build_context(tmp_lm=lm, tracking=True)
    concepts = AutoencoderConcepts(context)
    concepts._ensure_heaps(3)
    latents = torch.tensor(
        [
            [0.1, 0.0, 0.3],
            [0.4, -0.2, 0.0],
        ]
    )
    texts = ["alpha", "beta"]
    concepts.update_top_texts_from_latents(latents, texts)
    top_neuron0 = concepts.get_top_texts_for_neuron(0)
    assert isinstance(top_neuron0[0], NeuronText)
    assert top_neuron0[0].text in {"alpha", "beta"}
    assert concepts.get_all_top_texts()[0]
    json_path = concepts.export_top_texts_to_json(tmp_path / "top.json")
    csv_path = concepts.export_top_texts_to_csv(tmp_path / "top.csv")
    assert json.loads(Path(json_path).read_text()) != {}
    assert csv_path.exists()
    concepts.reset_top_texts()
    assert concepts.get_all_top_texts() == []


def test_update_top_texts_replaces_existing_entry():
    context = build_context(tmp_lm=None, tracking=True)
    concepts = AutoencoderConcepts(context)
    concepts._ensure_heaps(1)
    latents_first = torch.tensor([[0.2], [0.1]])
    texts = ["same", "same"]
    concepts.update_top_texts_from_latents(latents_first, texts)
    latents_second = torch.tensor([[0.5], [0.4]])
    concepts.update_top_texts_from_latents(latents_second, texts)
    result = concepts.get_top_texts_for_neuron(0)
    assert result[0].score >= 0.5


def test_generate_concepts_requires_top_texts():
    context = build_context(tmp_lm=None, tracking=False)
    concepts = AutoencoderConcepts(context)
    with pytest.raises(ValueError, match="No top texts available"):
        concepts.generate_concepts_with_llm()


def test_load_concepts_from_files(tmp_path):
    csv_path = tmp_path / "concepts.csv"
    csv_path.write_text("neuron_idx,concept_name,score\n0,a,0.5\n", encoding="utf-8")
    json_path = tmp_path / "concepts.json"
    json_path.write_text(json.dumps({"1": {"name": "b", "score": 0.7}}), encoding="utf-8")
    context = build_context(tmp_lm=None, tracking=False)
    concepts = AutoencoderConcepts(context)
    concepts.load_concepts_from_csv(csv_path)
    assert concepts.dictionary.get(0).name == "a"
    concepts.load_concepts_from_json(json_path)
    assert concepts.dictionary.get(1).name == "b"
