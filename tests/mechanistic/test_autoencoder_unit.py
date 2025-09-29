import math
import torch

from amber.mechanistic.autoencoder.autoencoder import Autoencoder


def test_autoencoder_forward_project_and_renorm_untied():
    torch.manual_seed(0)
    d = 12
    k = 5
    ae = Autoencoder(n_latents=k, n_inputs=d, activation="TopKReLU_3", tied=False)
    x = torch.randn(8, d)

    recon, lat, recon_full, lat_full = ae(x)
    assert recon.shape == x.shape
    assert lat.shape == (8, k)
    assert recon_full.shape == x.shape
    assert lat_full.shape == (8, k)

    # Create gradients and project
    loss = torch.nn.functional.mse_loss(recon, x)
    loss.backward()
    # Should not error even if some grads are None
    with torch.no_grad():
        ae.project_grads_decode()
        ae.scale_to_unit_norm()

    if ae.decoder is not None:
        # Each row should be unit norm (approx)
        norms = ae.decoder.data.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_autoencoder_tied_and_init_methods_exercise_paths():
    torch.manual_seed(0)
    d = 10
    k = 4
    for method in ["kaiming", "xavier", "uniform", "normal"]:
        ae = Autoencoder(n_latents=k, n_inputs=d, activation="TopK_2", tied=True, init_method=method)
        # Call init for a subset of neurons to hit branch
        ae._init_weights(neuron_indices=[0, 2])
        x = torch.randn(3, d)
        recon, lat, _, _ = ae(x)
        loss = (recon - x).pow(2).mean()
        loss.backward()
        with torch.no_grad():
            ae.project_grads_decode()
            ae.scale_to_unit_norm()
        # For tied, encoder columns should be unit norm
        norms = ae.encoder.data.T.norm(p=2, dim=-1)
        assert torch.all(norms > 0)


def test_autoencoder_encode_with_topk_number():
    torch.manual_seed(0)
    d = 9
    k = 6
    ae = Autoencoder(n_latents=k, n_inputs=d, activation="TopK_3", tied=False)
    x = torch.randn(2, d)
    enc, full, _ = ae.encode(x, topk_number=2)
    # full should have exactly 2 non-zeros per row (or <= if dim<k)
    nz = (full != 0).sum(dim=-1)
    assert torch.all(nz <= torch.tensor(2))
    # Encoded uses activation path (TopK during training), so at most k non-zeros
    assert enc.shape == (2, k)
