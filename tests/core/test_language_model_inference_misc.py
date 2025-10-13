import torch
from torch import nn

from amber.core.language_model import LanguageModel


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(32, 4)

    def forward(self, input_ids, attention_mask=None):
        return self.emb(input_ids)


class Tok:
    def __call__(self, texts, **kwargs):
        padding = kwargs.get("padding", False)
        rt = kwargs.get("return_tensors", "pt")
        ids = [[1] * max(1, len(t)) for t in texts]
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


def test_inference_swallows_tracker_exception():
    lm = LanguageModel(model=Tiny(), tokenizer=Tok())

    class BadTracker:
        def set_current_texts(self, texts):
            raise RuntimeError("boom")

    lm.register_activation_text_tracker(BadTracker())
    # Should not raise despite tracker error
    out, enc = lm.forwards(["a", "bb"], autocast=False)
    assert isinstance(out, torch.Tensor)
