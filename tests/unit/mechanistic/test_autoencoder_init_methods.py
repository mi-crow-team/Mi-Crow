import torch
from amber.mechanistic.autoencoder.autoencoder import Autoencoder


def test_autoencoder_init_methods_cover_branches():
    # kaiming is default and already covered elsewhere; cover xavier, uniform, normal
    for method in ["xavier", "uniform", "normal"]:
        ae = Autoencoder(n_latents=4, n_inputs=4, init_method=method)
        # Ensure shapes are correct after init
        assert ae.encoder.shape == (4, 4)
        if not ae.context.tied:
            assert ae.decoder.shape == (4, 4)
        # Trigger partial re-init path for neuron_indices
        with torch.no_grad():
            ae._init_weights(neuron_indices=[0, 1])
        # Basic forward sanity
        x = torch.randn(2, 4)
        reconstructed, latents, reconstructed_full, full_latents = ae(x)
        assert reconstructed.shape == (2, 4)
        assert latents.shape == (2, 4)
        assert reconstructed_full.shape == (2, 4)
        assert full_latents.shape == (2, 4)
        # Cover utility methods
        ae.project_grads_decode()
        with torch.no_grad():
            ae.scale_to_unit_norm()
