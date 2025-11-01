from typing import Dict, List, Callable, Sequence, Any, TYPE_CHECKING

from torch import nn
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from amber.core.language_model_context import LanguageModelContext


class LanguageModelTokenizer:

    def __init__(
            self,
            context: "LanguageModelContext"
    ):
        self.context = context

    def tokenize(self, texts: Sequence[str], **kwargs: Any):
        """Robust batch tokenization that works across tokenizer variants.

        Tries, in order:
        - callable tokenizer (most HF tokenizers)
        - batch_encode_plus
        - encode_plus per item + tokenizer.pad to collate
        """
        tokenizer = self.context.tokenizer
        model = self.context.model

        # Ensure a padding token exists if padding is requested
        padding_requested = kwargs.get("padding", False)
        if padding_requested and hasattr(tokenizer, "pad_token") and getattr(tokenizer, "pad_token", None) is None:
            # Prefer using eos_token as pad for causal LMs (no embedding resize needed)
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is not None:
                tokenizer.pad_token = eos_token
                if hasattr(model, "config"):
                    # Keep model config in sync
                    model.config.pad_token_id = getattr(tokenizer, "eos_token_id", None)
            else:
                # Fall back to adding a new [PAD] token and resizing embeddings
                if hasattr(tokenizer, "add_special_tokens"):
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    if hasattr(model, "resize_token_embeddings"):
                        # Resize model embeddings to account for the new token
                        model.resize_token_embeddings(len(tokenizer))  # type: ignore[arg-type]
                    if hasattr(model, "config"):
                        model.config.pad_token_id = getattr(tokenizer, "pad_token_id", None)

        # Most HF tokenizers are callable
        if callable(tokenizer):  # type: ignore[call-arg]
            try:
                return tokenizer(texts, **kwargs)
            except TypeError:
                # Some tests simulate non-callable behavior by raising here; fall back gracefully
                pass
        # Fallbacks for non-callable tokenizers
        if hasattr(tokenizer, "batch_encode_plus"):
            return tokenizer.batch_encode_plus(texts, **kwargs)
        if hasattr(tokenizer, "encode_plus"):
            encoded = [tokenizer.encode_plus(t, **kwargs) for t in texts]
            if hasattr(tokenizer, "pad"):
                # Honor return_tensors option if provided; default to pt which callers expect
                rt = kwargs.get("return_tensors") or "pt"
                return tokenizer.pad(encoded, return_tensors=rt)
            # As a last resort, return the raw list of dicts
            return encoded
        raise TypeError("Tokenizer object on LanguageModel is not usable for batch tokenization")
