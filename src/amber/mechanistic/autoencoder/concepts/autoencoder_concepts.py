from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import json
import csv

import torch
from torch import nn

from amber.mechanistic.autoencoder.concepts.concept_models import NeuronText
from amber.mechanistic.autoencoder.concepts.top_neuron_texts import TopNeuronTexts
from amber.mechanistic.autoencoder.autoencoder_context import AutoencoderContext
from amber.utils import get_logger

if TYPE_CHECKING:
    from amber.mechanistic.autoencoder.concepts.concept_dictionary import ConceptDictionary

logger = get_logger(__name__)


class AutoencoderConcepts:
    def __init__(
            self,
            context: AutoencoderContext
    ):
        self.context = context
        self._n_size = context.n_latents
        self.dictionary: ConceptDictionary | None = None

        # Concept manipulation parameters
        self.multiplication = nn.Parameter(torch.ones(self._n_size))
        self.bias = nn.Parameter(torch.ones(self._n_size))

        # Top texts tracking
        self.top_texts_tracker: TopNeuronTexts | None = None

    def enable_text_tracking(self):
        """Enable text tracking using context parameters."""
        if not (self.context.text_tracking_enabled and
                self.context.lm is not None and
                self.context.lm_layer_signature is not None):
            raise ValueError("LanguageModel and layer signature must be set in context to enable tracking")

        if self.top_texts_tracker is not None:
            try:
                self.top_texts_tracker.detach()
            except Exception:
                pass
            self.top_texts_tracker = None

        self.top_texts_tracker = TopNeuronTexts(
            self.context,
            k=self.context.text_tracking_k,
            negative=self.context.text_tracking_negative,
        )

    def disable_text_tracking(self):
        if self.top_texts_tracker is not None:
            try:
                self.top_texts_tracker.detach()
            except Exception:
                pass
            self.top_texts_tracker = None

    def _ensure_dictionary(self):
        if self.dictionary is None:
            from amber.mechanistic.autoencoder.concepts.concept_dictionary import ConceptDictionary
            self.dictionary = ConceptDictionary(self._n_size)
        return self.dictionary

    def load_concepts_from_csv(self, csv_filepath: str | Path):
        from amber.mechanistic.autoencoder.concepts.concept_dictionary import ConceptDictionary
        self.dictionary = ConceptDictionary.from_csv(
            csv_filepath=csv_filepath,
            n_size=self._n_size,
            store=self.dictionary.store if self.dictionary else None
        )

    def load_concepts_from_json(self, json_filepath: str | Path):
        from amber.mechanistic.autoencoder.concepts.concept_dictionary import ConceptDictionary
        self.dictionary = ConceptDictionary.from_json(
            json_filepath=json_filepath,
            n_size=self._n_size,
            store=self.dictionary.store if self.dictionary else None
        )

    def generate_concepts_with_llm(self, llm_provider: str | None = None):
        """Generate concepts using LLM based on current top texts"""
        if self.top_texts_tracker is None:
            raise ValueError("No text tracker available. Enable text tracking first.")

        from amber.mechanistic.autoencoder.concepts.concept_dictionary import ConceptDictionary
        neuron_texts = self.top_texts_tracker.get_all()

        self.dictionary = ConceptDictionary.from_llm(
            neuron_texts=neuron_texts,
            n_size=self._n_size,
            store=self.dictionary.store if self.dictionary else None,
            llm_provider=llm_provider
        )

    def manipulate_concept(
            self,
            neuron_idx: int,
            multiplier: float | None = None,
            bias: float | None = None
    ):
        if self.dictionary is None:
            logger.warning("No dictionary was created yet")
        self.multiplication.data[neuron_idx] = multiplier
        self.bias.data[neuron_idx] = bias

    def get_top_texts_for_neuron(self, neuron_idx: int, top_m: int | None = None) -> list[NeuronText]:
        if self.top_texts_tracker is None:
            return []
        return self.top_texts_tracker.get_top_texts(neuron_idx, top_m)

    def get_all_top_texts(self) -> list[list[NeuronText]]:
        if self.top_texts_tracker is None:
            return []
        return self.top_texts_tracker.get_all()

    def reset_top_texts(self) -> None:
        if self.top_texts_tracker is not None:
            self.top_texts_tracker.reset()

    def export_top_texts_to_json(self, filepath: Path | str) -> Path:
        if self.top_texts_tracker is None:
            raise ValueError("No text tracker available. Enable text tracking first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        all_texts = self.top_texts_tracker.get_all()
        export_data = {}

        for neuron_idx, neuron_texts in enumerate(all_texts):
            export_data[neuron_idx] = [
                {
                    "text": nt.text,
                    "score": nt.score,
                    "token_str": nt.token_str,
                    "token_idx": nt.token_idx
                }
                for nt in neuron_texts
            ]

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        return filepath

    def export_top_texts_to_csv(self, filepath: Path | str) -> Path:
        if self.top_texts_tracker is None:
            raise ValueError("No text tracker available. Enable text tracking first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        all_texts = self.top_texts_tracker.get_all()

        with filepath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["neuron_idx", "text", "score", "token_str", "token_idx"])

            for neuron_idx, neuron_texts in enumerate(all_texts):
                for nt in neuron_texts:
                    writer.writerow([neuron_idx, nt.text, nt.score, nt.token_str, nt.token_idx])

        return filepath
