from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence
import json
import csv
import heapq

import torch
from torch import nn

from mi_crow.mechanistic.sae.concepts.concept_models import NeuronText
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

        # Concept manipulation parameters
        self.multiplication = nn.Parameter(torch.ones(self._n_size))
        self.bias = nn.Parameter(torch.ones(self._n_size))

        # Top texts tracking
        self._top_texts_heaps: list[list[tuple[float, tuple[float, str, int]]]] | None = None
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
        if self._top_texts_heaps is None:
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
        if self._top_texts_heaps is None:
            self._top_texts_heaps = [[] for _ in range(n_neurons)]

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
            # Create token indices: [0, 1, 2, ..., T-1, 0, 1, 2, ..., T-1, ...]
            token_indices = torch.arange(T, device='cpu').unsqueeze(0).expand(B, T).contiguous().view(B * T)
        else:
            # Original was [B, D], latents are [B, n_latents]
            # All tokens are at index 0
            T = 1
            token_indices = torch.zeros(BT, dtype=torch.long, device='cpu')

        # For each neuron, find the maximum activation per text
        # This ensures we only track the best activation for each text, not every token position
        for j in range(n_neurons):
            heap = self._top_texts_heaps[j]

            # For each text in the batch, find the max activation and its token position
            texts_processed = 0
            texts_added = 0
            texts_updated = 0
            texts_skipped_duplicate = 0
            for batch_idx in range(original_B):
                if batch_idx >= len(texts):
                    continue

                text = texts[batch_idx]
                texts_processed += 1

                # Get activations for this text (all token positions)
                if original_shape is not None and len(original_shape) == 3:
                    # 3D case: [B, T, D] -> get slice for this batch
                    start_idx = batch_idx * T
                    end_idx = start_idx + T
                    text_activations = latents[start_idx:end_idx, j]  # [T]
                    text_token_indices = token_indices[start_idx:end_idx]  # [T]
                else:
                    # 2D case: [B, D] -> single token
                    text_activations = latents[batch_idx:batch_idx + 1, j]  # [1]
                    text_token_indices = token_indices[batch_idx:batch_idx + 1]  # [1]

                # Find the maximum activation (or minimum if tracking negative)
                if self._text_tracking_negative:
                    # For negative tracking, find the most negative (minimum) value
                    max_idx = torch.argmin(text_activations)
                    max_score = float(text_activations[max_idx].item())
                    adj = -max_score  # Negate for heap ordering
                else:
                    # For positive tracking, find the maximum value
                    max_idx = torch.argmax(text_activations)
                    max_score = float(text_activations[max_idx].item())
                    adj = max_score

                # Skip if score is zero (no activation)
                if max_score == 0.0:
                    continue

                token_idx = int(text_token_indices[max_idx].item())

                # Check if we already have this text in the heap
                # If so, only update if this activation is better
                existing_entry = None
                heap_texts = []
                for heap_idx, (heap_adj, (heap_score, heap_text, heap_token_idx)) in enumerate(heap):
                    heap_texts.append(heap_text[:50] if len(heap_text) > 50 else heap_text)
                    if heap_text == text:
                        existing_entry = (heap_idx, heap_adj, heap_score, heap_token_idx)
                        break

                if existing_entry is not None:
                    # Update existing entry if this activation is better
                    heap_idx, heap_adj, heap_score, heap_token_idx = existing_entry
                    if adj > heap_adj:
                        # Replace with better activation
                        heap[heap_idx] = (adj, (max_score, text, token_idx))
                        heapq.heapify(heap)  # Re-heapify after modification
                        texts_updated += 1
                    else:
                        texts_skipped_duplicate += 1
                else:
                    # New text, add to heap
                    if len(heap) < self._text_tracking_k:
                        heapq.heappush(heap, (adj, (max_score, text, token_idx)))
                        texts_added += 1
                    else:
                        # Compare with smallest adjusted score; replace if better
                        if adj > heap[0][0]:
                            heapq.heapreplace(heap, (adj, (max_score, text, token_idx)))
                            texts_added += 1

    def get_top_texts_for_neuron(self, neuron_idx: int, top_m: int | None = None) -> list[NeuronText]:
        """Get top texts for a specific neuron."""
        if self._top_texts_heaps is None or neuron_idx < 0 or neuron_idx >= len(self._top_texts_heaps):
            return []
        heap = self._top_texts_heaps[neuron_idx]
        items = [val for (_, val) in heap]
        reverse = not self._text_tracking_negative
        items_sorted = sorted(items, key=lambda s_t: s_t[0], reverse=reverse)
        if top_m is not None:
            items_sorted = items_sorted[: top_m]

        neuron_texts = []
        for score, text, token_idx in items_sorted:
            token_str = self._decode_token(text, token_idx)
            neuron_texts.append(NeuronText(score=score, text=text, token_idx=token_idx, token_str=token_str))
        return neuron_texts

    def get_all_top_texts(self) -> list[list[NeuronText]]:
        """Get top texts for all neurons."""
        if self._top_texts_heaps is None:
            return []
        return [self.get_top_texts_for_neuron(i) for i in range(len(self._top_texts_heaps))]

    def reset_top_texts(self) -> None:
        """Reset all tracked top texts."""
        self._top_texts_heaps = None

    def export_top_texts_to_json(self, filepath: Path | str) -> Path:
        if self._top_texts_heaps is None:
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

    def export_top_texts_to_csv(self, filepath: Path | str) -> Path:
        if self._top_texts_heaps is None:
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
