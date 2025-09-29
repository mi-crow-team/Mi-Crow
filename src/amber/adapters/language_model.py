from typing import Dict, List, Callable, Sequence, Any

from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class LanguageModel:
    model = None
    tokenizer = None
    name_to_layer: Dict[str, nn.Module] = {}
    idx_to_layer: Dict[int, nn.Module] = {}

    # Allow calling the wrapper like a torch.nn.Module by delegating to the underlying model
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __init__(
            self,
            model: nn.Module,
            tokenizer: AutoTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self._flatten_layer_names()

    def _flatten_layer_names(self):
        self.name_to_layer.clear()
        self.idx_to_layer.clear()

        def _recurse(module: nn.Module, prefix: str, idx: List[int]):
            for name, child in module.named_children():
                clean_name = f"{prefix}_{name}".replace(".", "_")
                idx_val = len(self.idx_to_layer)
                self.name_to_layer[clean_name] = child
                self.idx_to_layer[idx_val] = child
                _recurse(child, clean_name, idx)

        _recurse(self.model, self.model.__class__.__name__.lower(), [])

        return self.name_to_layer, self.idx_to_layer

    # --- Layer access ---

    def _get_layer_by_name(self, layer_name: str):
        if not self.name_to_layer:
            self._flatten_layer_names()
        if layer_name not in self.name_to_layer:
            raise ValueError(f"Layer name '{layer_name}' not found in model.")
        return self.name_to_layer[layer_name]

    def _get_layer_by_index(self, layer_index: int):
        if not self.idx_to_layer:
            self._flatten_layer_names()
        if layer_index not in self.idx_to_layer:
            raise ValueError(f"Layer index '{layer_index}' not found in model.")
        return self.idx_to_layer[layer_index]

    def get_layer_names(self):
        return list(self.name_to_layer.keys())

    def register_forward_hook_for_layer(
            self,
            layer_signature: str | int,
            hook: Callable,  # TODO: perhaps we could make some better signature
            hook_args: dict = None
    ):
        if isinstance(layer_signature, int):
            layer = self._get_layer_by_index(layer_signature)
        else:
            layer = self._get_layer_by_name(layer_signature)
        return layer.register_forward_hook(hook, **(hook_args or {}))

    # --- Hooks ---

    def register_pre_forward_hook_for_layer(
            self,
            layer_signature: str | int,
            hook: Callable,  # TODO: perhaps we could make some better signature
            hook_args: dict = None
    ):
        if isinstance(layer_signature, int):
            layer = self._get_layer_by_index(layer_signature)
        else:
            layer = self._get_layer_by_name(layer_signature)
        return layer.register_forward_pre_hook(hook, **(hook_args or {}))

    def register_new_layer(
            self,
            layer_name: str,
            layer: nn.Module,
            after_layer_signature: str | int,
    ):
        if isinstance(after_layer_signature, int):
            after_layer = self._get_layer_by_index(after_layer_signature)
        else:
            after_layer = self._get_layer_by_name(after_layer_signature)
        after_layer.add_module(layer_name, layer)
        self._flatten_layer_names()

    # --- Factory methods ---

    @classmethod
    def from_huggingface(
            cls,
            model_name: str,
            tokenizer_params: dict = None,
            model_params: dict = None,
    ) -> "LanguageModel":
        if tokenizer_params is None:
            tokenizer_params = {}
        if model_params is None:
            model_params = {}
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_params)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_params)
        return cls(model=model, tokenizer=tokenizer)  # TODO: fix the type hinting

    # --- Tokenization wrapper ---
    def tokenize(self, texts: Sequence[str], **kwargs: Any):
        """Robust batch tokenization that works across tokenizer variants.

        Tries, in order:
        - callable tokenizer (most HF tokenizers)
        - batch_encode_plus
        - encode_plus per item + tokenizer.pad to collate
        """
        tok = self.tokenizer

        # Ensure a padding token exists if padding is requested
        padding_requested = kwargs.get("padding", False)
        if padding_requested and hasattr(tok, "pad_token") and getattr(tok, "pad_token", None) is None:
            # Prefer using eos_token as pad for causal LMs (no embedding resize needed)
            eos_token = getattr(tok, "eos_token", None)
            if eos_token is not None:
                tok.pad_token = eos_token
                if hasattr(self.model, "config"):
                    # Keep model config in sync
                    self.model.config.pad_token_id = getattr(tok, "eos_token_id", None)
            else:
                # Fall back to adding a new [PAD] token and resizing embeddings
                if hasattr(tok, "add_special_tokens"):
                    tok.add_special_tokens({"pad_token": "[PAD]"})
                    if hasattr(self.model, "resize_token_embeddings"):
                        # Resize model embeddings to account for the new token
                        self.model.resize_token_embeddings(len(tok))  # type: ignore[arg-type]
                    if hasattr(self.model, "config"):
                        self.model.config.pad_token_id = getattr(tok, "pad_token_id", None)

        # Most HF tokenizers are callable
        if callable(tok):  # type: ignore[call-arg]
            return tok(texts, **kwargs)
        # Fallbacks for non-callable tokenizers
        if hasattr(tok, "batch_encode_plus"):
            return tok.batch_encode_plus(texts, **kwargs)
        if hasattr(tok, "encode_plus"):
            encoded = [tok.encode_plus(t, **kwargs) for t in texts]
            if hasattr(tok, "pad"):
                # Honor return_tensors option if provided; default to pt which is expected by callers
                rt = kwargs.get("return_tensors") or "pt"
                return tok.pad(encoded, return_tensors=rt)
            # As a last resort, return the raw list of dicts
            return encoded
        raise TypeError("Tokenizer object on LanguageModel is not usable for batch tokenization")

    @classmethod
    def from_local(cls, model_path: str, tokenizer_path: str):
        """Load model and tokenizer from local directories without network access.

        Parameters:
            model_path: Filesystem path to a directory containing a saved HF causal LM.
            tokenizer_path: Filesystem path to a directory containing a saved HF tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        return cls(model=model, tokenizer=tokenizer)
