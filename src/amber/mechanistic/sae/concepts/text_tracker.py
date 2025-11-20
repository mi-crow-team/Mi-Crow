"""Text tracking functionality for SAE concepts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, TYPE_CHECKING
import json
import csv
import heapq

import torch

from amber.mechanistic.sae.concepts.concept_models import NeuronText
from amber.utils import get_logger

if TYPE_CHECKING:
    from amber.mechanistic.sae.autoencoder_context import AutoencoderContext

logger = get_logger(__name__)


class TextTracker:
    """
    Manages heap-based text tracking for SAE neurons.
    
    Tracks top-k activating texts for each neuron using min-heaps,
    supporting both positive and negative activation tracking.
    """

    def __init__(
            self,
            context: "AutoencoderContext",
            k: int = 5,
            track_negative: bool = False
    ):
        """
        Initialize TextTracker.
        
        Args:
            context: AutoencoderContext instance
            k: Number of top texts to track per neuron
            track_negative: Whether to track negative activations
        """
        self.context = context
        self._k = k
        self._track_negative = track_negative
        self._top_texts_heaps: list[list[tuple[float, tuple[float, str, int]]]] | None = None

    @property
    def k(self) -> int:
        """Number of top texts to track per neuron."""
        return self._k

    @k.setter
    def k(self, value: int) -> None:
        """Set number of top texts to track per neuron."""
        if value < 1:
            raise ValueError(f"k must be >= 1, got {value}")
        self._k = value

    @property
    def track_negative(self) -> bool:
        """Whether to track negative activations."""
        return self._track_negative

    @track_negative.setter
    def track_negative(self, value: bool) -> None:
        """Set whether to track negative activations."""
        self._track_negative = value

    def _ensure_heaps(self, n_neurons: int) -> None:
        """Ensure heaps are initialized for the given number of neurons."""
        if self._top_texts_heaps is None:
            self._top_texts_heaps = [[] for _ in range(n_neurons)]

    def _decode_token(self, text: str, token_idx: int) -> str:
        """
        Decode a specific token from the text using the language model's tokenizer.
        
        Args:
            text: Input text
            token_idx: Token index to decode
            
        Returns:
            Decoded token string or placeholder if decoding fails
        """
        if self.context.lm is None:
            return f"<token_{token_idx}>"

        try:
            if self.context.lm.tokenizer is None:
                return f"<token_{token_idx}>"

            # Use the raw tokenizer (not the wrapper) to encode and decode
            tokenizer = self.context.lm.tokenizer

            # Tokenize the text to get token IDs
            tokens = tokenizer.encode(text, add_special_tokens=False)
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

    def update_from_latents(
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
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not texts:
            return
        
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(latents)}")
        
        if len(latents.shape) != 2:
            raise ValueError(f"Expected 2D tensor [batch*seq_len, n_latents], got shape {latents.shape}")
        
        if latents.shape[0] == 0:
            raise ValueError("Cannot update from empty latents tensor")

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
            for batch_idx in range(original_B):
                if batch_idx >= len(texts):
                    continue

                text = texts[batch_idx]

                # Get activations for this text (all token positions)
                if original_shape is not None and len(original_shape) == 3:
                    # 3D case: [B, T, D] -> get slice for this batch
                    start_idx = batch_idx * T
                    end_idx = start_idx + T
                    text_activations = latents[start_idx:end_idx, j]  # [T]
                    text_token_indices = token_indices[start_idx:end_idx]  # [T]
                else:
                    # 2D case: [B, D] -> single token
                    text_activations = latents[batch_idx:batch_idx+1, j]  # [1]
                    text_token_indices = token_indices[batch_idx:batch_idx+1]  # [1]

                # Find the maximum activation (or minimum if tracking negative)
                if self._track_negative:
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
                for heap_idx, (heap_adj, (heap_score, heap_text, heap_token_idx)) in enumerate(heap):
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
                else:
                    # New text, add to heap
                    if len(heap) < self._k:
                        heapq.heappush(heap, (adj, (max_score, text, token_idx)))
                    else:
                        # Compare with smallest adjusted score; replace if better
                        if adj > heap[0][0]:
                            heapq.heapreplace(heap, (adj, (max_score, text, token_idx)))

    def get_top_texts_for_neuron(self, neuron_idx: int, top_m: int | None = None) -> list[NeuronText]:
        """
        Get top texts for a specific neuron.
        
        Args:
            neuron_idx: Index of the neuron
            top_m: Optional limit on number of texts to return
            
        Returns:
            List of NeuronText objects for the neuron
        """
        if self._top_texts_heaps is None or neuron_idx < 0 or neuron_idx >= len(self._top_texts_heaps):
            return []
        heap = self._top_texts_heaps[neuron_idx]
        items = [val for (_, val) in heap]
        reverse = not self._track_negative
        items_sorted = sorted(items, key=lambda s_t: s_t[0], reverse=reverse)
        if top_m is not None:
            items_sorted = items_sorted[:top_m]

        neuron_texts = []
        for score, text, token_idx in items_sorted:
            token_str = self._decode_token(text, token_idx)
            neuron_texts.append(NeuronText(score=score, text=text, token_idx=token_idx, token_str=token_str))
        return neuron_texts

    def get_all_top_texts(self) -> list[list[NeuronText]]:
        """
        Get top texts for all neurons.
        
        Returns:
            List of lists of NeuronText objects, one per neuron
        """
        if self._top_texts_heaps is None:
            return []
        return [self.get_top_texts_for_neuron(i) for i in range(len(self._top_texts_heaps))]

    def reset(self) -> None:
        """Reset all tracked top texts."""
        self._top_texts_heaps = None

    def export_to_json(self, filepath: Path | str) -> Path:
        """
        Export top texts to JSON file.
        
        Args:
            filepath: Path to output JSON file
            
        Returns:
            Path to the created file
            
        Raises:
            ValueError: If no top texts are available
        """
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

    def export_to_csv(self, filepath: Path | str) -> Path:
        """
        Export top texts to CSV file.
        
        Args:
            filepath: Path to output CSV file
            
        Returns:
            Path to the created file
            
        Raises:
            ValueError: If no top texts are available
        """
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

