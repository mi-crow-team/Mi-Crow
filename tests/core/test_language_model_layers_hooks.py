import torch
from torch import nn

from amber.core.language_model import LanguageModel


class SmallLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(32, 4)
        self.lin = nn.Linear(4, 4)

    def forward(self, input_ids, attention_mask=None):
        x = self.emb(input_ids)
        return self.lin(x)


class Tok:
    def __call__(self, texts, **kwargs):
        padding = kwargs.get("padding", False)
        rt = kwargs.get("return_tensors", "pt")
        ids = [[1 + i for i, _ in enumerate(t)] for t in texts]
        T = max(len(x) for x in ids)
        if padding:
            ids = [row + [0] * (T - len(row)) for row in ids]
        attn = [[1] * len(t) + [0] * (T - len(t)) for t in texts]
        if rt == "pt":
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }
        raise ValueError


def test_pre_forward_hook_is_called_and_handle_removable():
    lm = LanguageModel(model=SmallLM(), tokenizer=Tok())

    # choose the linear layer by index and then by name to exercise both paths
    names = list(lm.layers.name_to_layer.keys())
    lin_name = [n for n in names if n.endswith("lin") or ".lin" in n][0]
    lin_index = list(lm.layers.name_to_layer.keys()).index(lin_name)

    seen = {"calls": 0, "last_shape": None}

    def pre_hook(_module, inputs):
        seen["calls"] += 1
        if inputs and isinstance(inputs[0], torch.Tensor):
            seen["last_shape"] = tuple(inputs[0].shape)

    # by index
    h1 = lm.layers.register_pre_forward_hook_for_layer(lin_index, pre_hook)
    # by name
    h2 = lm.layers.register_pre_forward_hook_for_layer(lin_name, pre_hook)

    out, enc = lm.forwards(["ab", "c"], autocast=False)
    assert isinstance(out, torch.Tensor)
    # both hooks should have fired once
    assert seen["calls"] == 2
    assert seen["last_shape"][0] == enc["input_ids"].shape[0]

    # cleanup handles should not raise
    h1.remove()
    h2.remove()
