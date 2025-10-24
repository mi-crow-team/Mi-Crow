import pytest
import torch
from torch import nn

from amber.core.language_model_layers import LanguageModelLayers
from amber.mechanistic.autoencoder.autoencoder import Autoencoder


class Block(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B, T, D] and return [B, T, D]
        assert x.dim() == 3
        b, t, d = x.shape
        y = self.proj(x.reshape(b * t, d))
        return y.view(b, t, d)


class TinyModel(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.block = Block(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BadAutoencoder(Autoencoder):
    """An Autoencoder that intentionally returns a wrong-width reconstruction.

    n_inputs matches the parent feature size so it passes the pre-check,
    but forward returns recon tensors with width D+1 to trigger the shape enforcement error.
    """

    def forward(self, x: torch.Tensor, detach: bool = False):  # type: ignore[override]
        b = x.shape[0]
        # wrong feature width
        recon = torch.zeros((b, x.shape[1] + 1), dtype=x.dtype, device=x.device)
        latents = torch.zeros((b, self.n_latents), dtype=x.dtype, device=x.device)
        recon_full = recon.clone()
        full_latents = latents.clone()
        if detach:
            recon = recon.detach()
            recon_full = recon_full.detach()
            latents = latents.detach()
            full_latents = full_latents.detach()
        return recon, latents, recon_full, full_latents


def _layer_signature_for_block(model: nn.Module) -> str:
    # LanguageModelLayers flattens using model class lowercased as prefix
    return f"{model.__class__.__name__.lower()}_block"


def test_register_new_layer_sae_reshapes_and_matches_output_shape():
    torch.manual_seed(0)
    d = 8
    model = TinyModel(d)
    layers = LanguageModelLayers(lm=object(), model=model)

    # Attach a standard Autoencoder with matching feature size
    sae = Autoencoder(n_latents=4, n_inputs=d)
    after_sig = _layer_signature_for_block(model)
    hook = layers.register_new_layer("sae", sae, after_layer_signature=after_sig)

    # Concepts linkage should be set on register
    assert sae.concepts.lm is layers._lm
    assert sae.concepts.lm_layer_signature == f"{after_sig}_sae"

    try:
        x = torch.randn(2, 3, d)
        y = model(x)
        assert y.shape == x.shape  # must match parent output shape exactly
    finally:
        # cleanup the hook
        hook.remove()


def test_register_new_layer_sae_feature_dim_mismatch_raises():
    d = 6
    model = TinyModel(d)
    layers = LanguageModelLayers(lm=object(), model=model)

    # Intentionally set n_inputs != D to trigger the feature-dim check
    sae = Autoencoder(n_latents=4, n_inputs=d + 1)

    with pytest.raises(RuntimeError) as ei:
        layers.register_new_layer("sae", sae, after_layer_signature=_layer_signature_for_block(model))
        # need to run a forward to exercise the hook
        _ = model(torch.randn(1, 2, d))
    assert "feature dim mismatch" in str(ei.value)


def test_register_new_layer_sae_reconstruction_shape_mismatch_raises():
    d = 5
    model = TinyModel(d)
    layers = LanguageModelLayers(lm=object(), model=model)

    # n_inputs matches D, but BadAutoencoder will return wrong-width reconstructions
    bad_sae = BadAutoencoder(n_latents=3, n_inputs=d)
    hook = layers.register_new_layer("sae", bad_sae, after_layer_signature=_layer_signature_for_block(model))

    try:
        with pytest.raises(RuntimeError) as ei:
            _ = model(torch.randn(2, 2, d))
        assert "reconstruction shape" in str(ei.value)
    finally:
        hook.remove()
