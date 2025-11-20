from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from amber.mechanistic.sae.concepts.concept_models import NeuronText
from amber.mechanistic.sae.concepts.text_tracker import TextTracker
from amber.mechanistic.sae.autoencoder_context import AutoencoderContext
from amber.utils import get_logger

if TYPE_CHECKING:
    from amber.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary

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

        # Text tracking (delegated to TextTracker)
        self._text_tracker: TextTracker | None = None

    def enable_text_tracking(self):
        """Enable text tracking using context parameters."""
        if not (self.context.text_tracking_enabled and
                self.context.lm is not None):
            raise ValueError("LanguageModel must be set in context to enable tracking")

        # Create TextTracker with context parameters
        self._text_tracker = TextTracker(
            context=self.context,
            k=self.context.text_tracking_k,
            track_negative=self.context.text_tracking_negative
        )

        # Ensure InputTracker singleton exists on LanguageModel and enable it
        input_tracker = self.context.lm._ensure_input_tracker()
        input_tracker.enable()

        # Enable text tracking on the SAE instance
        if hasattr(self.context.autoencoder, '_text_tracking_enabled'):
            self.context.autoencoder._text_tracking_enabled = True

    def disable_text_tracking(self):
        """Disable text tracking."""
        if hasattr(self.context.autoencoder, '_text_tracking_enabled'):
            self.context.autoencoder._text_tracking_enabled = False
        self._text_tracker = None

    def _ensure_dictionary(self):
        if self.dictionary is None:
            from amber.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary
            self.dictionary = ConceptDictionary(self._n_size)
        return self.dictionary

    def load_concepts_from_csv(self, csv_filepath: str | Path):
        from amber.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary
        self.dictionary = ConceptDictionary.from_csv(
            csv_filepath=csv_filepath,
            n_size=self._n_size,
            store=self.dictionary.store if self.dictionary else None
        )

    def load_concepts_from_json(self, json_filepath: str | Path):
        from amber.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary
        self.dictionary = ConceptDictionary.from_json(
            json_filepath=json_filepath,
            n_size=self._n_size,
            store=self.dictionary.store if self.dictionary else None
        )

    def generate_concepts_with_llm(self, llm_provider: str | None = None):
        """Generate concepts using LLM based on current top texts"""
        if self._text_tracker is None:
            raise ValueError("No top texts available. Enable text tracking and run inference first.")

        from amber.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary
        neuron_texts = self.get_all_top_texts()

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

    def update_top_texts_from_latents(
            self,
            latents: torch.Tensor,
            texts: Sequence[str],
            original_shape: tuple[int, ...] | None = None
    ) -> None:
        """
        Update top texts from latents and texts (delegates to TextTracker).
        
        Args:
            latents: Latent activations tensor, shape [B*T, n_latents] or [B, n_latents] (already flattened)
            texts: List of texts corresponding to the batch
            original_shape: Original shape before flattening, e.g., (B, T, D) or (B, D)
        """
        if self._text_tracker is not None:
            self._text_tracker.update_from_latents(latents, texts, original_shape)

    def get_top_texts_for_neuron(self, neuron_idx: int, top_m: int | None = None) -> list[NeuronText]:
        """
        Get top texts for a specific neuron (delegates to TextTracker).
        
        Args:
            neuron_idx: Index of the neuron
            top_m: Optional limit on number of texts to return
            
        Returns:
            List of NeuronText objects for the neuron
        """
        if self._text_tracker is None:
            return []
        return self._text_tracker.get_top_texts_for_neuron(neuron_idx, top_m)

    def get_all_top_texts(self) -> list[list[NeuronText]]:
        """
        Get top texts for all neurons (delegates to TextTracker).
        
        Returns:
            List of lists of NeuronText objects, one per neuron
        """
        if self._text_tracker is None:
            return []
        return self._text_tracker.get_all_top_texts()

    def reset_top_texts(self) -> None:
        """Reset all tracked top texts (delegates to TextTracker)."""
        if self._text_tracker is not None:
            self._text_tracker.reset()

    def export_top_texts_to_json(self, filepath: Path | str) -> Path:
        """
        Export top texts to JSON file (delegates to TextTracker).
        
        Args:
            filepath: Path to output JSON file
            
        Returns:
            Path to the created file
            
        Raises:
            ValueError: If no top texts are available
        """
        if self._text_tracker is None:
            raise ValueError("No top texts available. Enable text tracking and run inference first.")
        return self._text_tracker.export_to_json(filepath)

    def export_top_texts_to_csv(self, filepath: Path | str) -> Path:
        """
        Export top texts to CSV file (delegates to TextTracker).
        
        Args:
            filepath: Path to output CSV file
            
        Returns:
            Path to the created file
            
        Raises:
            ValueError: If no top texts are available
        """
        if self._text_tracker is None:
            raise ValueError("No top texts available. Enable text tracking and run inference first.")
        return self._text_tracker.export_to_csv(filepath)
