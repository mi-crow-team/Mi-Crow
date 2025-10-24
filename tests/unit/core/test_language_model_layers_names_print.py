import io
import sys
import torch
from torch import nn

from amber.core.language_model_layers import LanguageModelLayers


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(8, 2)
        self.lin = nn.Linear(2, 2)

    def forward(self, x):
        return self.lin(self.emb(x))


def test_get_and_print_layer_names_covers_branches(capsys):
    m = M()
    layers = LanguageModelLayers(lm=object(), model=m)
    names = layers.get_layer_names()
    assert any("emb" in n for n in names) and any("lin" in n for n in names)

    # Capture print output
    layers.print_layer_names()
    out = capsys.readouterr().out
    assert "emb" in out and "lin" in out
