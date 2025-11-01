from typing import TYPE_CHECKING, Sequence, Any

import heapq
import torch

from amber.utils import get_logger
from amber.mechanistic.autoencoder.concepts.concept_models import NeuronText
from amber.mechanistic.autoencoder.autoencoder_context import AutoencoderContext
from amber.hooks.detector import Detector
from amber.hooks.hook import HookType

if TYPE_CHECKING:
    from amber.core.language_model import LanguageModel
    from torch import nn

logger = get_logger(__name__)


class TopNeuronTexts(Detector):

    def __init__(
            self,
            context: AutoencoderContext,
            k: int = 5,
            *,
            nth_tensor: int = 1,
            negative: bool = False,
            hook_id: str | None = None
    ) -> None:
        # Initialize detector with layer signature
        super().__init__(
            layer_signature=context.lm_layer_signature,
            hook_type=HookType.FORWARD,
            hook_id=hook_id,
            store=None  # We don't use store in TopNeuronTexts
        )
        
        self.context = context
        if self.context.text_tracking_k <= 0:
            raise ValueError("k must be positive")
        self.negative = bool(negative)

        # runtime state
        self._current_texts: Sequence[str] | None = None
        self._heaps: list[list[tuple[float, tuple[float, str]]]] | None = None
        self.last_activations: torch.Tensor | None = None

        # Register with LM for texts and use new hook system
        self.nth_tensor = nth_tensor
        self._hook_handle = None
        
        # Register using the new hook system
        try:
            hook_id = self.context.lm.layers.register_hook(
                self.context.lm_layer_signature,
                self
            )
            self.id = hook_id  # Update our ID to match what was registered
        except Exception as e:
            logger.error(f"Failed to register TopNeuronTexts hook: {e}")
            raise
        
        try:
            self.context.lm.register_activation_text_tracker(self)
        except Exception:
            pass
        logger.debug("Registered activations hook and tracker for TopNeuronTexts")

    def detach(self) -> None:
        """Unregister this hook from the language model."""
        try:
            # Use new hook system to unregister
            self.context.lm.layers.unregister_hook(self.id)
        except Exception:
            pass
        try:
            self.context.lm.unregister_activation_text_tracker(self)
        except Exception:
            pass

    def reset(self) -> None:
        self._heaps = None
        self.last_activations = None

    def set_current_texts(self, texts: Sequence[str]) -> None:
        self._current_texts = list(texts)

    def _ensure_heaps(self, n_neurons: int) -> None:
        if self._heaps is None:
            self._heaps = [[] for _ in range(n_neurons)]

    def _reduce_over_tokens(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Expect tensor of shape [B, T, D] or [B, D]
        if tensor.dim() == 3:
            # For 3D tensors, we want to track all token-neuron pairs
            # Reshape to [B*T, D] to treat each token as a separate batch item
            B, T, D = tensor.shape
            tensor_flat = tensor.view(B * T, D)

            # Create token indices for each position
            token_indices = torch.arange(T, device=tensor.device).unsqueeze(0).expand(B, T).contiguous().view(B * T)

            return tensor_flat, token_indices
        # For [B, D] case, assume all tokens are at index 0
        indices = torch.zeros(tensor.shape[0], dtype=torch.long, device=tensor.device)
        return tensor, indices

    def _decode_token(self, text: str, token_idx: int) -> str:
        """Decode a specific token from the text using the language model's tokenizer."""
        try:
            if self.context.lm.tokenizer is None:
                return f"<token_{token_idx}>"

            # Tokenize the text to get token IDs
            tokens = self.context.lm.lm_tokenizer.encode(text, add_special_tokens=False)
            if 0 <= token_idx < len(tokens):
                token_id = tokens[token_idx]
                # Decode the specific token
                token_str = self.context.lm.lm_tokenizer.decode([token_id])
                return token_str
            else:
                return f"<token_{token_idx}_out_of_range>"
        except Exception:
            # If tokenization fails, return a placeholder
            return f"<token_{token_idx}_decode_error>"

    def _update_heaps(self, scores_bt: torch.Tensor, token_indices: torch.Tensor) -> None:
        assert self._current_texts is not None, "Current texts not set before forward"
        BT, D = scores_bt.shape
        self._ensure_heaps(D)

        # Calculate original batch size from the flattened tensor
        # We need to figure out how many tokens per text
        original_B = len(self._current_texts)
        T = BT // original_B

        for i in range(BT):
            # Map flattened index back to original batch and token
            batch_idx = i // T
            token_idx = int(token_indices[i].item())
            text = self._current_texts[batch_idx]

            for j in range(D):
                score = float(scores_bt[i, j].item())
                adj = -score if self.negative else score
                heap = self._heaps[j]
                if len(heap) < self.context.text_tracking_k:
                    heapq.heappush(heap, (adj, (score, text, token_idx)))
                else:
                    # Compare with smallest adjusted score; replace if better
                    if adj > heap[0][0]:
                        heapq.heapreplace(heap, (adj, (score, text, token_idx)))

    def process_activations(self, module: "nn.Module", inputs: tuple, output: Any) -> None:
        """
        Process activations from the hooked layer.
        
        Extracts tensor from output, updates heaps with top activations.
        """
        # Normalize output to a tensor
        tensor: torch.Tensor | None = None
        if isinstance(output, torch.Tensor):
            tensor = output
        elif isinstance(output, (tuple, list)):
            if len(output) < self.nth_tensor + 1:
                raise ValueError(f"Expected at least {self.nth_tensor + 1} tensors in output, got {len(output)}")
            tensor = output[self.nth_tensor]
        else:
            if hasattr(output, "last_hidden_state"):
                maybe = getattr(output, "last_hidden_state")
                if isinstance(maybe, torch.Tensor):
                    tensor = maybe
        if tensor is None:
            return
        self.last_activations = tensor.detach().to("cpu")
        try:
            scores_bt, token_indices = self._reduce_over_tokens(self.last_activations)
            self._update_heaps(scores_bt, token_indices)
        except Exception:
            # Do not raise from hooks
            pass

    def get_top_texts(self, neuron_idx: int, top_m: int | None = None) -> list[NeuronText]:
        if self._heaps is None or neuron_idx < 0 or neuron_idx >= len(self._heaps):
            return []
        heap = self._heaps[neuron_idx]
        items = [val for (_, val) in heap]
        # Sort presentation: positive -> desc by score; negative -> asc by score
        reverse = not self.negative
        items_sorted = sorted(items, key=lambda s_t: s_t[0], reverse=reverse)
        if top_m is not None:
            items_sorted = items_sorted[: top_m]

        # Decode tokens for each item
        neuron_texts = []
        for score, text, token_idx in items_sorted:
            token_str = self._decode_token(text, token_idx)
            neuron_texts.append(NeuronText(score=score, text=text, token_idx=token_idx, token_str=token_str))
        return neuron_texts

    def get_all(self) -> list[list[NeuronText]]:
        if self._heaps is None:
            return []
        return [self.get_top_texts(i) for i in range(len(self._heaps))]
