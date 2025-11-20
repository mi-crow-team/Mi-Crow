import pytest
import torch
from torch import nn
from dataclasses import dataclass, field
from typing import Dict, List, Any

from amber.language_model.language_model_layers import LanguageModelLayers


class BlockA(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class BlockB(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class TinyNestedModel(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.a = BlockA(d)
        self.container = nn.Sequential(BlockB(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape [B, D]
        x = self.a(x)
        x = self.container(x)
        return x


@dataclass
class MockContext:
    """Mock context for testing."""
    language_model: Any
    model: nn.Module | None = None
    _hook_registry: Dict[str | int, Dict[str, List[tuple[Any, Any]]]] = field(default_factory=dict)
    _hook_id_map: Dict[str, tuple[str | int, str, Any]] = field(default_factory=dict)


def test_get_layer_names_and_errors():
    model = TinyNestedModel(4)
    context = MockContext(language_model=object(), model=model)
    layers = LanguageModelLayers(context=context)
    names = layers.get_layer_names()
    # Should contain flattened names for a and container_0
    assert any(name.endswith('_a') for name in names)
    assert any('container' in name for name in names)

    # Access by name works
    for name in names:
        _ = layers._get_layer_by_name(name)

    # Invalid name raises
    with pytest.raises(ValueError):
        layers._get_layer_by_name('does_not_exist')

    # Invalid index raises
    with pytest.raises(ValueError):
        layers._get_layer_by_index(10_000)


def test_register_forward_and_pre_forward_hooks_fire():
    torch.manual_seed(0)
    model = TinyNestedModel(3)
    context = MockContext(language_model=object(), model=model)
    layers = LanguageModelLayers(context=context)

    call_order: list[str] = []

    def pre_hook(_m, _in):
        call_order.append('pre')

    def fwd_hook(_m, _in, out):
        call_order.append('fwd')
        assert isinstance(out, torch.Tensor)

    # pick the first layer name to attach hooks
    name = layers.get_layer_names()[0]
    h1 = layers.register_pre_forward_hook_for_layer(name, pre_hook)
    h2 = layers.register_forward_hook_for_layer(name, fwd_hook)

    try:
        _ = model(torch.randn(2, 3))
        assert call_order == ['pre', 'fwd']
    finally:
        h1.remove()
        h2.remove()


class DoubleLayer(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * 2


