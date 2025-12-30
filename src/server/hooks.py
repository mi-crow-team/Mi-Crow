from __future__ import annotations

from typing import Dict

import torch

from mi_crow.hooks.controller import Controller
from mi_crow.hooks.hook import HookType


class NeuronMultiplierController(Controller):
    def __init__(
        self,
        layer_signature: str | int,
        weights: Dict[int, float],
        hook_type: HookType | str = HookType.PRE_FORWARD,
        hook_id: str | None = None,
    ):
        if not weights:
            raise ValueError("weights cannot be empty")
        normalized = {int(k): float(v) for k, v in weights.items()}
        super().__init__(hook_type=hook_type, hook_id=hook_id, layer_signature=layer_signature)
        self.weights = normalized

    def modify_activations(self, module, inputs: torch.Tensor, output: torch.Tensor | None):
        if not isinstance(inputs, torch.Tensor):
            return None
        result = inputs.clone()
        for idx, weight in self.weights.items():
            if idx < 0 or idx >= result.shape[-1]:
                continue
            result[..., idx] = result[..., idx] * weight
        return result
