from typing import Sequence, Any

import torch
from torch import nn

from amber.core.language_model import LanguageModel
from amber.mechanistic.autoencoder.concepts.autoencoder_concepts import AutoencoderConcepts


class FakeTokenizer:
    def __init__(self, vocab: dict[str, int] | None = None, pad_id: int = 0):
        self.vocab = vocab or {}
        self.pad_id = pad_id

    def _encode_one(self, text: str) -> list[int]:
        ids = []
        for tok in text.split():
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab) + 1  # 0 is padding
            ids.append(self.vocab[tok])
        if not ids:
            ids = [self.pad_id]
        return ids

    def __call__(self, texts: Sequence[str], **kwargs: Any):
        padding = kwargs.get("padding", False)
        truncation = kwargs.get("truncation", False)
        max_length = kwargs.get("max_length")
        return_tensors = kwargs.get("return_tensors", "pt")

        encoded = [self._encode_one(t) for t in texts]
        if truncation and max_length is not None:
            encoded = [e[: max_length] for e in encoded]
        lengths = [len(e) for e in encoded]
        max_len = max(lengths) if padding else max(lengths)
        if padding:
            encoded = [e + [self.pad_id] * (max_len - len(e)) for e in encoded]
        input_ids = torch.tensor(encoded, dtype=torch.long)
        attention_mask = torch.tensor([[1] * l + [0] * (max_len - l) for l in lengths], dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        raise ValueError("Only return_tensors='pt' supported in FakeTokenizer")


class SyntheticLayer(nn.Module):
    """
    Deterministic layer producing activations based on token ids:
    For each token id x, emits vector [x, -x, 0.5*x] (float32).
    Output shape: [B, T, D] with D=3.
    """

    def __init__(self, d: int = 3):
        super().__init__()
        assert d == 3, "This synthetic layer is defined for D=3"
        self.d = d

    def forward(self, input_ids, attention_mask=None):
        x = input_ids.to(torch.float32)
        v0 = x
        v1 = -x
        v2 = 0.5 * x
        out = torch.stack([v0, v1, v2], dim=-1)
        return out


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        # Expose a named submodule so LanguageModelLayers can hook into it
        self.synthetic = SyntheticLayer()

    def forward(self, input_ids, attention_mask=None):
        # Simply return the synthetic layer's activations
        return self.synthetic(input_ids=input_ids, attention_mask=attention_mask)


def _find_layer_name(lm: LanguageModel, submod: nn.Module) -> str:
    for name, layer in lm.layers.name_to_layer.items():
        if layer is submod:
            return name
    raise AssertionError("Layer not found in flattened names")


def test_top_texts_tracker_positive_and_negative(tmp_path):
    # Build LM wrapper around deterministic TinyLM
    tok = FakeTokenizer()
    net = TinyLM()
    lm = LanguageModel(model=net, tokenizer=tok)

    # Determine the layer name for the synthetic submodule
    target_layer_name = _find_layer_name(lm, net.synthetic)

    # Attach via AutoencoderConcepts
    # AutoencoderConcepts expects a context
    from amber.mechanistic.autoencoder.autoencoder import Autoencoder
    sae = Autoencoder(n_latents=3, n_inputs=8)
    concepts = AutoencoderConcepts(sae.context)
    # Provide LM and layer context and enable tracking
    concepts.context.lm = lm
    concepts.context.lm_layer_signature = target_layer_name
    concepts.context.text_tracking_k = 2
    concepts.context.text_tracking_negative = False
    concepts.context.text_tracking_enabled = True

    # Positive tracking: higher token id -> higher score on neuron 0
    concepts.enable_text_tracking()

    texts1 = ["a", "b", "c"]  # ids will be 1,2,3 (in this call order)
    lm.forwards(texts1)

    # After first forward, top-2 for neuron 0 should be ['c', 'b']
    top0 = concepts.get_top_texts_for_neuron(0)
    assert [nt.text for nt in top0] == ["c", "b"], f"Unexpected top0 after first pass: {top0}"
    assert [round(nt.score, 4) for nt in top0] == [3.0, 2.0]

    # Second forward introduces higher-id tokens; tracker should update keeping k=2 best
    texts2 = ["d", "e"]  # ids 4,5
    lm.forwards(texts2)

    top0 = concepts.get_top_texts_for_neuron(0)
    assert [nt.text for nt in top0] == ["e", "d"], f"Top0 should update to highest ids only: {top0}"
    assert [round(nt.score, 4) for nt in top0] == [5.0, 4.0]

    # Now test negative tracking on neuron 1 (-id): most-negative = lowest value
    # Detach previous and re-enable with negative=True
    concepts.disable_text_tracking()
    concepts.context.text_tracking_k = 3
    concepts.context.text_tracking_negative = True
    concepts.context.text_tracking_enabled = True
    concepts.enable_text_tracking()

    # Re-run over a combined set
    lm.forwards(["a", "b", "c", "d", "e"])  # ids 1..5

    top1 = concepts.get_top_texts_for_neuron(1)
    # Most negative (lowest scores) for neuron1 are from highest ids: 'e', 'd', 'c'
    assert [nt.text for nt in top1] == ["e", "d", "c"], f"Unexpected order for negative tracking: {top1}"
    # Scores should be negative values -5, -4, -3
    assert [round(nt.score, 4) for nt in top1] == [-5.0, -4.0, -3.0]

    # Sanity for neuron 2 (0.5 * id), k=3
    # In negative mode, the tracker picks the smallest (most negative relative key),
    # which corresponds to the smallest ids for positive-valued neuron 2.
    top2 = concepts.get_top_texts_for_neuron(2)
    assert [nt.text for nt in top2] == ["a", "b", "c"]
    assert [round(nt.score, 4) for nt in top2] == [0.5, 1.0, 1.5]

    # Cleanup
    concepts.disable_text_tracking()
    # Ensure reset works without error
    concepts.reset_top_texts()
