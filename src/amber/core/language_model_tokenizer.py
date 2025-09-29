from typing import Dict, List, Callable, Sequence, Any

from torch import nn
from transformers import AutoTokenizer


class LanguageModelTokenizer:

    def __init__(
            self,
            model: nn.Module,
            tokenizer: AutoTokenizer
    ):
        self._model = model
        self._tokenizer = tokenizer

    def tokenize(self, texts: Sequence[str], **kwargs: Any):
        """Robust batch tokenization that works across tokenizer variants.

        Tries, in order:
        - callable tokenizer (most HF tokenizers)
        - batch_encode_plus
        - encode_plus per item + tokenizer.pad to collate
        """
        tok = self._tokenizer

        # Ensure a padding token exists if padding is requested
        padding_requested = kwargs.get("padding", False)
        if padding_requested and hasattr(tok, "pad_token") and getattr(tok, "pad_token", None) is None:
            # Prefer using eos_token as pad for causal LMs (no embedding resize needed)
            eos_token = getattr(tok, "eos_token", None)
            if eos_token is not None:
                tok.pad_token = eos_token
                if hasattr(self._model, "config"):
                    # Keep model config in sync
                    self._model.config.pad_token_id = getattr(tok, "eos_token_id", None)
            else:
                # Fall back to adding a new [PAD] token and resizing embeddings
                if hasattr(tok, "add_special_tokens"):
                    tok.add_special_tokens({"pad_token": "[PAD]"})
                    if hasattr(self._model, "resize_token_embeddings"):
                        # Resize model embeddings to account for the new token
                        self._model.resize_token_embeddings(len(tok))  # type: ignore[arg-type]
                    if hasattr(self._model, "config"):
                        self._model.config.pad_token_id = getattr(tok, "pad_token_id", None)

        # Most HF tokenizers are callable
        if callable(tok):  # type: ignore[call-arg]
            return tok(texts, **kwargs)
        # Fallbacks for non-callable tokenizers
        if hasattr(tok, "batch_encode_plus"):
            return tok.batch_encode_plus(texts, **kwargs)
        if hasattr(tok, "encode_plus"):
            encoded = [tok.encode_plus(t, **kwargs) for t in texts]
            if hasattr(tok, "pad"):
                # Honor return_tensors option if provided; default to pt which callers expect
                rt = kwargs.get("return_tensors") or "pt"
                return tok.pad(encoded, return_tensors=rt)
            # As a last resort, return the raw list of dicts
            return encoded
        raise TypeError("Tokenizer object on LanguageModel is not usable for batch tokenization")
