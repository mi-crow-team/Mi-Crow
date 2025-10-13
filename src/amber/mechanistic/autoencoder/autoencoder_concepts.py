from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from amber.mechanistic.autoencoder.concepts.concept_models import NeuronText
from amber.mechanistic.autoencoder.concepts.top_neuron_texts import TopNeuronTexts
from amber.utils import get_logger

if TYPE_CHECKING:
    from amber.mechanistic.autoencoder.concept_dictionary import ConceptDictionary

logger = get_logger(__name__)


class AutoencoderConcepts:
    def __init__(
            self,
            n_size: int,
            dictionary_path: str | Path | None = None,
            lm: "LanguageModel | None" = None,
            lm_layer_signature: int | str | None = None
    ):
        self._n_size = n_size
        self._dictionary_path = Path(dictionary_path) if dictionary_path is not None else None
        self.dictionary = None

        # Concept manipulation parameters
        self.multiplication = nn.Parameter(torch.ones(n_size))
        self.bias = nn.Parameter(torch.ones(n_size))

        # Language model refs
        self.lm = lm
        self.lm_layer_signature = lm_layer_signature

        # Top texts tracking
        self.top_texts_tracker: TopNeuronTexts | None = None

    def enable_text_tracking(self, k: int = 5, *, negative: bool = False):
        if self.lm is None or self.lm_layer_signature is None:
            raise ValueError("LanguageModel and layer signature must be set to enable tracking")
        self.lm.enable_input_text_tracking()
        # Detach any existing tracker first
        if self.top_texts_tracker is not None:
            try:
                self.top_texts_tracker.detach()
            except Exception:
                pass
            self.top_texts_tracker = None
        self.top_texts_tracker = TopNeuronTexts(
            self.lm,
            self.lm_layer_signature,
            k=k,
            negative=negative,
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
            from amber.mechanistic.autoencoder.concept_dictionary import ConceptDictionary
            if self._dictionary_path is not None:
                self.dictionary = ConceptDictionary.from_directory(self._dictionary_path)
            else:
                self.dictionary = ConceptDictionary(self._n_size)
        return self.dictionary

    def multiply_concept(
            self,
            concept_idx: int,
            multiplier: float
    ):
        if self.dictionary is None:
            logger.warning("No dictionary was created yet")
        self.multiplication.data[concept_idx] = multiplier

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
