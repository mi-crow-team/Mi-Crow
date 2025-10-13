from typing import TYPE_CHECKING, Sequence

import heapq
import torch

from amber.utils import get_logger

from amber.mechanistic.autoencoder.concepts.concept_models import NeuronText

if TYPE_CHECKING:
    from amber.core.language_model import LanguageModel

logger = get_logger(__name__)


# TODO: Maybe we can make it inherit Tracker ABC?
class TopNeuronTexts:

    def __init__(
            self,
            lm: "LanguageModel",
            layer_signature: str | int,
            k: int = 5,
            *,
            nth_tensor: int = 1,
            negative: bool = False
    ) -> None:
        self.lm = lm
        self.layer_signature = layer_signature
        self.k = int(k)
        if self.k <= 0:
            raise ValueError("k must be positive")
        self.negative = bool(negative)

        # runtime state
        self._current_texts: Sequence[str] | None = None
        self._heaps: list[list[tuple[float, tuple[float, str]]]] | None = None
        self.last_activations: torch.Tensor | None = None

        # Register with LM for texts and forward hook for activations
        self.nth_tensor = nth_tensor
        self._hook_handle = self.lm.layers.register_forward_hook_for_layer(self.layer_signature, self._activations_hook)
        try:
            self.lm.register_activation_text_tracker(self)
        except Exception:
            pass
        logger.debug("Registered activations hook and tracker for TopNeuronTexts")

    def detach(self) -> None:
        try:
            if hasattr(self, "_hook_handle") and self._hook_handle is not None:
                self._hook_handle.remove()
        except Exception:
            pass
        try:
            self.lm.unregister_activation_text_tracker(self)
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

    def _reduce_over_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        # Expect tensor of shape [B, T, D] or [B, D]
        if tensor.dim() == 3:
            if self.negative:
                # For negative tracking pick minimum across tokens per neuron
                return tensor.min(dim=1).values
            return tensor.max(dim=1).values
        return tensor  # [B, D]

    def _update_heaps(self, scores_bt: torch.Tensor) -> None:
        assert self._current_texts is not None, "Current texts not set before forward"
        B, D = scores_bt.shape
        self._ensure_heaps(D)
        for i in range(B):
            text = self._current_texts[i]
            for j in range(D):
                score = float(scores_bt[i, j].item())
                adj = -score if self.negative else score
                heap = self._heaps[j]
                if len(heap) < self.k:
                    heapq.heappush(heap, (adj, (score, text)))
                else:
                    # Compare with smallest adjusted score; replace if better
                    if adj > heap[0][0]:
                        heapq.heapreplace(heap, (adj, (score, text)))

    def _activations_hook(self, _module, _input, output):
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
            scores_bt = self._reduce_over_tokens(self.last_activations)
            if scores_bt.dim() == 1:
                scores_bt = scores_bt.unsqueeze(0)
            self._update_heaps(scores_bt)
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
        return [NeuronText(score=score, text=text) for score, text in items_sorted]

    def get_all(self) -> list[list[NeuronText]]:
        if self._heaps is None:
            return []
        return [self.get_top_texts(i) for i in range(len(self._heaps))]
