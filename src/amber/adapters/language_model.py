from typing import Dict, List, Callable

from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class LanguageModel:
    model = None
    tokenizer = None
    name_to_layer: Dict[str, nn.Module] = {}
    idx_to_layer: Dict[int, nn.Module] = {}

    def __init__(
            self,
            model: AutoModelForCausalLM,
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
        layer.register_forward_hook(hook, **(hook_args or {}))

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
        layer.register_forward_pre_hook(hook, **(hook_args or {}))

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

    @classmethod
    def from_huggingface(
            cls,
            model_name: str,
            tokenizer_params: dict = None,
            model_params: dict = None,
    ) -> "LanguageModel":
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_params)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_params)
        return cls(model=model, tokenizer=tokenizer)  # TODO: fix the type hinting

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
