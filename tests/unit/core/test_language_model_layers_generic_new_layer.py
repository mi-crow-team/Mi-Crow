import pytest
import torch
from torch import nn

from amber.core.language_model_layers import LanguageModelLayers


class Block(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        y = self.proj(x.reshape(b * t, d))
        return y.view(b, t, d)


class TinyModel(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.block = Block(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ReturnsTensor(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x: torch.Tensor):
        return self.lin(x)


class ReturnsTuple(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x: torch.Tensor):
        y = self.lin(x)
        return (y, {"aux": 1})


class Obj:
    def __init__(self, t: torch.Tensor):
        self.last_hidden_state = t


class ReturnsObject(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x: torch.Tensor):
        return Obj(self.lin(x))


class ReturnsBad(nn.Module):
    def forward(self, x: torch.Tensor):
        return {"no": "tensor"}


def _sig(model: nn.Module) -> str:
    return f"{model.__class__.__name__.lower()}_block"


@pytest.mark.parametrize("child_cls", [ReturnsTensor, ReturnsTuple, ReturnsObject])
def test_register_new_layer_generic_child_output_selection(child_cls):
    torch.manual_seed(0)
    d = 7
    model = TinyModel(d)
    layers = LanguageModelLayers(lm=object(), model=model)

    child = child_cls(d)
    hook = layers.register_new_layer("child", child, after_layer_signature=_sig(model))
    try:
        x = torch.randn(2, 3, d)
        y = model(x)
        # shape must be preserved for generic children as well
        assert y.shape == x.shape
    finally:
        hook.remove()


def test_register_new_layer_generic_child_unsupported_type_raises():
    d = 5
    model = TinyModel(d)
    layers = LanguageModelLayers(lm=object(), model=model)

    bad = ReturnsBad()
    hook = layers.register_new_layer("bad", bad, after_layer_signature=_sig(model))
    try:
        with pytest.raises(RuntimeError):
            _ = model(torch.randn(1, 2, d))
    finally:
        hook.remove()
