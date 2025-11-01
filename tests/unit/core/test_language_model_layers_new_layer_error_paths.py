import pytest
import torch
from torch import nn
from dataclasses import dataclass, field
from typing import Dict, List, Any

from amber.core.language_model_layers import LanguageModelLayers


@dataclass
class MockContext:
    """Mock context for testing."""
    language_model: Any
    model: nn.Module | None = None
    _hook_registry: Dict[str | int, Dict[str, List[tuple[Any, Any]]]] = field(default_factory=dict)
    _hook_id_map: Dict[str, tuple[str | int, str, Any]] = field(default_factory=dict)


class ParentReturnsBad(nn.Module):
    def forward(self, x):
        return {"not": "a tensor"}


class Model(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.block = ParentReturnsBad()

    def forward(self, x):
        return self.block(x)


def _sig(model: nn.Module) -> str:
    return f"{model.__class__.__name__.lower()}_block"


class ChildRaises(nn.Module):
    def forward(self, x):
        raise ValueError("fail")


class Passthrough(nn.Module):
    def forward(self, x):
        return x


def test_register_new_layer_raises_when_parent_output_has_no_tensor():
    m = Model(d=3)
    context = MockContext(language_model=object(), model=m)
    layers = LanguageModelLayers(context=context)
    # child won't be called because parent output can't be converted to tensor input
    hook = layers.register_new_layer("pt", Passthrough(), after_layer_signature=_sig(m))
    try:
        with pytest.raises(RuntimeError):
            _ = m(torch.randn(2, 3))
    finally:
        hook.remove()


def test_register_new_layer_wraps_child_forward_exception():
    # use a valid parent producing tensors
    class GoodParent(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.lin = nn.Linear(d, d)
        def forward(self, x):
            return self.lin(x)

    class M2(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.block = GoodParent(d)
        def forward(self, x):
            return self.block(x)

    m2 = M2(4)
    context = MockContext(language_model=object(), model=m2)
    layers = LanguageModelLayers(context=context)
    hook = layers.register_new_layer("bad", ChildRaises(), after_layer_signature=_sig(m2))
    try:
        with pytest.raises(RuntimeError):
            _ = m2(torch.randn(2, 4))
    finally:
        hook.remove()
