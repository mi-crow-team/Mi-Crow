from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence
import json
import csv
import heapq

import torch
from torch import nn

from mi_crow.mechanistic.sae.concepts.concept_models import NeuronText
from mi_crow.mechanistic.sae.concepts.text_heap import TextHeap
from mi_crow.mechanistic.sae.autoencoder_context import AutoencoderContext
from mi_crow.utils import get_logger

if TYPE_CHECKING:
    from mi_crow.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary

logger = get_logger(__name__)


class AutoencoderConcepts:
    def __init__(
            self,
            context: AutoencoderContext
    ):
        self.context = context
        self._n_size = context.n_latents
        self.dictionary: ConceptDictionary | None = None

        self.multiplication = nn.Parameter(torch.ones(self._n_size))
        self.bias = nn.Parameter(torch.ones(self._n_size))

        self._text_heaps_positive: list[TextHeap] | None = None
        self._text_heaps_negative: list[TextHeap] | None = None
        self._text_tracking_k: int = 5
        self._text_tracking_negative: bool = False

    def enable_text_tracking(self):
        """Enable text tracking using context parameters."""
        if self.context.lm is None:
            raise ValueError("LanguageModel must be set in context to enable tracking")

        # Store tracking parameters
        self._text_tracking_k = self.context.text_tracking_k
        self._text_tracking_negative = self.context.text_tracking_negative

        # Ensure InputTracker singleton exists on LanguageModel and enable it
        input_tracker = self.context.lm._ensure_input_tracker()
        input_tracker.enable()

        # Enable text tracking on the SAE instance
        if hasattr(self.context.autoencoder, '_text_tracking_enabled'):
            self.context.autoencoder._text_tracking_enabled = True

    def disable_text_tracking(self):
        self.context.autoencoder._text_tracking_enabled = False

    def _ensure_dictionary(self):
        if self.dictionary is None:
            from mi_crow.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary
            self.dictionary = ConceptDictionary(self._n_size)
        return self.dictionary

    def load_concepts_from_csv(self, csv_filepath: str | Path):
        from mi_crow.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary
        self.dictionary = ConceptDictionary.from_csv(
            csv_filepath=csv_filepath,
            n_size=self._n_size,
            store=self.dictionary.store if self.dictionary else None
        )

    def load_concepts_from_json(self, json_filepath: str | Path):
        from mi_crow.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary
        self.dictionary = ConceptDictionary.from_json(
            json_filepath=json_filepath,
            n_size=self._n_size,
            store=self.dictionary.store if self.dictionary else None
        )

    def generate_concepts_with_llm(self, llm_provider: str | None = None):
        """Generate concepts using LLM based on current top texts"""
        if self._text_heaps_positive is None:
            raise ValueError("No top texts available. Enable text tracking and run inference first.")

        from mi_crow.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary
        neuron_texts = self.get_all_top_texts()

        self.dictionary = ConceptDictionary.from_llm(
            neuron_texts=neuron_texts,
            n_size=self._n_size,
            store=self.dictionary.store if self.dictionary else None,
            llm_provider=llm_provider
        )

    def _ensure_heaps(self, n_neurons: int) -> None:
        """Ensure heaps are initialized for the given number of neurons."""
        if self._text_heaps_positive is None:
            self._text_heaps_positive = [TextHeap(self._text_tracking_k) for _ in range(n_neurons)]
        if self._text_tracking_negative and self._text_heaps_negative is None:
            self._text_heaps_negative = [TextHeap(self._text_tracking_k) for _ in range(n_neurons)]

    def _decode_token(self, text: str, token_idx: int) -> str:
        """
        Decode a specific token from the text using the language model's tokenizer.
        
        The token_idx is relative to the sequence length T that the model saw during inference.
        However, there's a mismatch: during inference, texts are tokenized with 
        add_special_tokens=True (which adds BOS/EOS), but the token_idx appears to be
        calculated relative to the sequence without special tokens.
        
        We tokenize the text the same way as _decode_token originally did (without special tokens)
        to match the token_idx calculation, but we also account for truncation that may have
        occurred during inference (max_length).
        """
        if self.context.lm is None:
            return f"<token_{token_idx}>"

        try:
            if self.context.lm.tokenizer is None:
                return f"<token_{token_idx}>"

            # Use the raw tokenizer (not the wrapper) to encode and decode
            tokenizer = self.context.lm.tokenizer

            # Tokenize without special tokens (matching original behavior)
            # This matches how token_idx was calculated in update_top_texts_from_latents
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Check if token_idx is valid
            if 0 <= token_idx < len(tokens):
                token_id = tokens[token_idx]
                # Decode the specific token
                token_str = tokenizer.decode([token_id])
                return token_str
            else:
                return f"<token_{token_idx}_out_of_range>"
        except Exception as e:
            # If tokenization fails, return a placeholder
            logger.debug(f"Token decode error for token_idx={token_idx} in text (len={len(text)}): {e}")
            return f"<token_{token_idx}_decode_error>"

    def update_top_texts_from_latents(
            self,
            latents: torch.Tensor,
            texts: Sequence[str],
            original_shape: tuple[int, ...] | None = None
    ) -> None:
        """
        Update top texts heaps from latents and texts.
        
        Optimized version that:
        - Only processes active neurons (non-zero activations)
        - Vectorizes argmax/argmin operations
        - Eliminates per-neuron tensor slicing
        
        Args:
            latents: Latent activations tensor, shape [B*T, n_latents] or [B, n_latents] (already flattened)
            texts: List of texts corresponding to the batch
            original_shape: Original shape before flattening, e.g., (B, T, D) or (B, D)
        """
        if not texts:
            return

        n_neurons = latents.shape[-1]
        self._ensure_heaps(n_neurons)

        # Calculate batch and token dimensions
        original_B = len(texts)
        BT = latents.shape[0]  # Total positions (B*T if 3D original, or B if 2D original)

        # Determine if original was 3D or 2D
        if original_shape is not None and len(original_shape) == 3:
            # Original was [B, T, D], latents are [B*T, n_latents]
            B, T, _ = original_shape
            # Verify batch size matches
            if B != original_B:
                logger.warning(f"Batch size mismatch: original_shape has B={B}, but {original_B} texts provided")
                # Use the actual number of texts as batch size
                B = original_B
                T = BT // B if B > 0 else 1
        else:
            # Original was [B, D], latents are [B, n_latents]
            B = original_B
            T = 1

        # OPTIMIZATION 1: Find active neurons (have any non-zero activation across batch)
        # Shape: [n_neurons] - boolean mask
        active_neurons_mask = (latents.abs().sum(dim=0) > 0)
        active_neuron_indices = torch.nonzero(active_neurons_mask, as_tuple=False).flatten().tolist()
        
        if not active_neuron_indices:
            return  # No active neurons, skip

        # OPTIMIZATION 2: Vectorize argmax/argmin for all neurons at once
        if original_shape is not None and len(original_shape) == 3:
            # Reshape to [B, T, n_neurons]
            latents_3d = latents.view(B, T, n_neurons)
            # For each text, find max/min across tokens for each neuron
            # Shape: [B, n_neurons] - max activation per text per neuron
            max_activations, max_token_indices_3d = latents_3d.max(dim=1)  # [B, n_neurons]
            min_activations, min_token_indices_3d = latents_3d.min(dim=1)  # [B, n_neurons]
            # max_token_indices_3d is already the token index (0 to T-1)
            max_token_indices = max_token_indices_3d
            min_token_indices = min_token_indices_3d
        else:
            # Shape: [B, n_neurons]
            latents_2d = latents.view(B, n_neurons)
            max_activations = latents_2d  # [B, n_neurons]
            max_token_indices = torch.zeros(B, n_neurons, dtype=torch.long, device=latents.device)
            min_activations = latents_2d
            min_token_indices = torch.zeros(B, n_neurons, dtype=torch.long, device=latents.device)

        # Convert to numpy for faster CPU access (already on CPU from l1_sae.py)
        max_activations_np = max_activations.cpu().numpy()
        min_activations_np = min_activations.cpu().numpy()
        max_token_indices_np = max_token_indices.cpu().numpy()
        min_token_indices_np = min_token_indices.cpu().numpy()

        # OPTIMIZATION 3: Only process active neurons
        for j in active_neuron_indices:
            heap_positive = self._text_heaps_positive[j]
            heap_negative = self._text_heaps_negative[j] if self._text_tracking_negative else None

            # OPTIMIZATION 4: Batch process all texts for this neuron
            for batch_idx in range(original_B):
                if batch_idx >= len(texts):
                    continue

                text = texts[batch_idx]
                
                # Use pre-computed max/min (no tensor slicing needed!)
                max_score_positive = float(max_activations_np[batch_idx, j])
                token_idx_positive = int(max_token_indices_np[batch_idx, j])

                if max_score_positive > 0.0:
                    heap_positive.update(text, max_score_positive, token_idx_positive)

                if self._text_tracking_negative and heap_negative is not None:
                    min_score_negative = float(min_activations_np[batch_idx, j])
                    if min_score_negative != 0.0:
                        token_idx_negative = int(min_token_indices_np[batch_idx, j])
                        heap_negative.update(text, min_score_negative, token_idx_negative, adjusted_score=-min_score_negative)
    
    def _extract_activations(
        self,
        latents: torch.Tensor,
        token_indices: torch.Tensor,
        batch_idx: int,
        neuron_idx: int,
        original_shape: tuple[int, ...] | None,
        T: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract activations for a specific batch item and neuron.
        
        Returns:
            Tuple of (text_activations, text_token_indices)
        """
        if original_shape is not None and len(original_shape) == 3:
            start_idx = batch_idx * T
            end_idx = start_idx + T
            text_activations = latents[start_idx:end_idx, neuron_idx]
            text_token_indices = token_indices[start_idx:end_idx]
        else:
            text_activations = latents[batch_idx:batch_idx + 1, neuron_idx]
            text_token_indices = token_indices[batch_idx:batch_idx + 1]
        
        return text_activations, text_token_indices

    def get_top_texts_for_neuron(self, neuron_idx: int, top_m: int | None = None) -> list[NeuronText]:
        """Get top texts for a specific neuron (positive activations)."""
        if self._text_heaps_positive is None or neuron_idx < 0 or neuron_idx >= len(self._text_heaps_positive):
            return []
        heap = self._text_heaps_positive[neuron_idx]
        items = heap.get_items()
        items_sorted = sorted(items, key=lambda s_t: s_t[0], reverse=True)
        if top_m is not None:
            items_sorted = items_sorted[: top_m]

        neuron_texts = []
        for score, text, token_idx in items_sorted:
            token_str = self._decode_token(text, token_idx)
            neuron_texts.append(NeuronText(score=score, text=text, token_idx=token_idx, token_str=token_str))
        return neuron_texts

    def get_bottom_texts_for_neuron(self, neuron_idx: int, top_m: int | None = None) -> list[NeuronText]:
        """Get bottom texts for a specific neuron (negative activations)."""
        if not self._text_tracking_negative:
            return []
        if self._text_heaps_negative is None or neuron_idx < 0 or neuron_idx >= len(self._text_heaps_negative):
            return []
        heap = self._text_heaps_negative[neuron_idx]
        items = heap.get_items()
        items_sorted = sorted(items, key=lambda s_t: s_t[0], reverse=False)
        if top_m is not None:
            items_sorted = items_sorted[: top_m]

        neuron_texts = []
        for score, text, token_idx in items_sorted:
            token_str = self._decode_token(text, token_idx)
            neuron_texts.append(NeuronText(score=score, text=text, token_idx=token_idx, token_str=token_str))
        return neuron_texts

    def get_all_top_texts(self) -> list[list[NeuronText]]:
        """Get top texts for all neurons (positive activations)."""
        if self._text_heaps_positive is None:
            return []
        return [self.get_top_texts_for_neuron(i) for i in range(len(self._text_heaps_positive))]

    def get_all_bottom_texts(self) -> list[list[NeuronText]]:
        """Get bottom texts for all neurons (negative activations)."""
        if not self._text_tracking_negative or self._text_heaps_negative is None:
            return []
        return [self.get_bottom_texts_for_neuron(i) for i in range(len(self._text_heaps_negative))]

    def reset_top_texts(self) -> None:
        """Reset all tracked top texts."""
        self._text_heaps_positive = None
        self._text_heaps_negative = None

    def export_top_texts_to_json(self, filepath: Path | str) -> Path:
        """Export top texts (positive activations) to JSON file."""
        if self._text_heaps_positive is None:
            raise ValueError("No top texts available. Enable text tracking and run inference first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        all_texts = self.get_all_top_texts()
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

    def export_bottom_texts_to_json(self, filepath: Path | str) -> Path:
        """Export bottom texts (negative activations) to JSON file."""
        if not self._text_tracking_negative or self._text_heaps_negative is None:
            raise ValueError("No bottom texts available. Enable negative text tracking and run inference first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        all_texts = self.get_all_bottom_texts()
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
        if self._text_heaps_positive is None:
            raise ValueError("No top texts available. Enable text tracking and run inference first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        all_texts = self.get_all_top_texts()

        with filepath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["neuron_idx", "text", "score", "token_str", "token_idx"])

            for neuron_idx, neuron_texts in enumerate(all_texts):
                for nt in neuron_texts:
                    writer.writerow([neuron_idx, nt.text, nt.score, nt.token_str, nt.token_idx])

        return filepath
