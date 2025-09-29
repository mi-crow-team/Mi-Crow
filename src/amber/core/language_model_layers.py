from typing import Dict, List, Callable

from torch import nn


class LanguageModelLayers:

    def __init__(
            self,
            model: nn.Module
    ):
        self._model = model
        self.name_to_layer: Dict[str, nn.Module] = {}
        self.idx_to_layer: Dict[int, nn.Module] = {}
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

        _recurse(self._model, self._model.__class__.__name__.lower(), [])

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

    def print_layer_names(self):
        for name, layer in self.name_to_layer.items():
            print(f"{name}: {getattr(layer, 'weight', None).shape if hasattr(layer, 'weight') else 'No weight'}")

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
