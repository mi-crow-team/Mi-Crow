from typing import Dict, List, Callable, Sequence, Any, TYPE_CHECKING, Union

from torch import nn
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from mi_crow.language_model.context import LanguageModelContext


class LanguageModelTokenizer:
    """Handles tokenization for LanguageModel."""

    def __init__(
            self,
            context: "LanguageModelContext"
    ):
        """
        Initialize LanguageModelTokenizer.
        
        Args:
            context: LanguageModelContext instance
        """
        self.context = context

    def _setup_pad_token(self, tokenizer: Any, model: Any) -> None:
        """
        Setup pad token for tokenizer if not already set.
        
        Args:
            tokenizer: Tokenizer instance
            model: Model instance
        """
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
            if hasattr(model, "config"):
                model.config.pad_token_id = getattr(tokenizer, "eos_token_id", None)
        else:
            if hasattr(tokenizer, "add_special_tokens"):
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                if hasattr(model, "resize_token_embeddings"):
                    model.resize_token_embeddings(len(tokenizer))
                if hasattr(model, "config"):
                    model.config.pad_token_id = getattr(tokenizer, "pad_token_id", None)

    def split_to_tokens(
            self,
            text: Union[str, Sequence[str]],
            add_special_tokens: bool = False
    ) -> Union[List[str], List[List[str]]]:
        """
        Split text into token strings.
        
        Args:
            text: Single string or sequence of strings to tokenize
            add_special_tokens: Whether to add special tokens (e.g., BOS, EOS)
        
        Returns:
            For a single string: list of token strings
            For a sequence of strings: list of lists of token strings
        """
        if isinstance(text, str):
            return self._split_single_text_to_tokens(text, add_special_tokens)

        return [self._split_single_text_to_tokens(t, add_special_tokens) for t in text]

    def _try_tokenize_with_method(
            self,
            tokenizer: Any,
            text: str,
            add_special_tokens: bool,
            method_name: str,
            fallback_method: str | None = None
    ) -> List[str] | None:
        """
        Try tokenizing using a specific tokenizer method.
        
        Args:
            tokenizer: Tokenizer instance
            text: Text to tokenize
            add_special_tokens: Whether to add special tokens
            method_name: Primary method to try (e.g., "tokenize", "encode")
            fallback_method: Optional fallback method (e.g., "convert_ids_to_tokens")
            
        Returns:
            List of token strings or None if method fails
        """
        if not hasattr(tokenizer, method_name):
            return None
        
        try:
            if method_name == "tokenize":
                return tokenizer.tokenize(text, add_special_tokens=add_special_tokens)
            elif method_name == "encode":
                if fallback_method and hasattr(tokenizer, fallback_method):
                    token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
                    return tokenizer.convert_ids_to_tokens(token_ids)
            elif method_name == "encode_plus":
                if fallback_method and hasattr(tokenizer, fallback_method):
                    encoded = tokenizer.encode_plus(text, add_special_tokens=add_special_tokens)
                    if isinstance(encoded, dict) and "input_ids" in encoded:
                        token_ids = encoded["input_ids"]
                        return tokenizer.convert_ids_to_tokens(token_ids)
        except (TypeError, ValueError, AttributeError):
            pass
        
        return None

    def _split_single_text_to_tokens(self, text: str, add_special_tokens: bool) -> List[str]:
        """
        Split a single text into token strings.
        
        Uses the tokenizer from LanguageModelContext to split text into tokens.
        
        Args:
            text: Text string to tokenize
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token strings
        """
        tokenizer = self.context.tokenizer

        if tokenizer is None:
            return text.split()
        
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)}")

        # Try different tokenization methods in order
        tokens = self._try_tokenize_with_method(tokenizer, text, add_special_tokens, "tokenize")
        if tokens is not None:
            return tokens

        tokens = self._try_tokenize_with_method(tokenizer, text, add_special_tokens, "encode", "convert_ids_to_tokens")
        if tokens is not None:
            return tokens

        tokens = self._try_tokenize_with_method(tokenizer, text, add_special_tokens, "encode_plus", "convert_ids_to_tokens")
        if tokens is not None:
            return tokens

        return text.split()

    def tokenize(
            self,
            texts: Sequence[str],
            padding: bool = False,
            pad_token: str = "[PAD]",
            **kwargs: Any
    ) -> Any:
        """
        Robust batch tokenization that works across tokenizer variants.

        Tries methods in order:
        - callable tokenizer (most HF tokenizers)
        - batch_encode_plus
        - encode_plus per item + tokenizer.pad to collate
        
        Args:
            texts: Sequence of text strings to tokenize
            padding: Whether to pad sequences
            pad_token: Pad token string
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Tokenized encodings
            
        Raises:
            ValueError: If tokenizer is not initialized
            TypeError: If tokenizer is not usable for batch tokenization
        """
        tokenizer = self.context.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer must be initialized before tokenization")
        
        model = self.context.model

        if padding and pad_token and getattr(tokenizer, "pad_token", None) is None:
            self._setup_pad_token(tokenizer, model)

        kwargs["padding"] = padding
        
        # Try callable tokenizer first (most common case)
        if callable(tokenizer):
            try:
                return tokenizer(texts, **kwargs)
            except (TypeError, NotImplementedError):
                pass

        # Try batch_encode_plus
        if hasattr(tokenizer, "batch_encode_plus"):
            return tokenizer.batch_encode_plus(texts, **kwargs)
        
        # Fallback to encode_plus per item
        if hasattr(tokenizer, "encode_plus"):
            encoded = [tokenizer.encode_plus(t, **kwargs) for t in texts]
            if hasattr(tokenizer, "pad"):
                rt = kwargs.get("return_tensors") or "pt"
                return tokenizer.pad(encoded, return_tensors=rt)
            return encoded
        
        raise TypeError("Tokenizer object on LanguageModel is not usable for batch tokenization")
