from pathlib import Path

from amber.mechanistic.autoencoder.autoencoder import Autoencoder


def test_autoencoder_save_and_load(tmp_path):
    d = 7
    k = 3
    ae = Autoencoder(n_latents=k, n_inputs=d, activation="TopK_2", tied=False)
    # Save to a provided path
    ae.save("model", path=str(tmp_path))
    file_path = tmp_path / "model.pt"
    assert file_path.exists()

    # Load into a fresh model
    ae2 = Autoencoder(n_latents=k, n_inputs=d, activation="TopK_2", tied=False)
    ae2.load("model", path=str(tmp_path))
    # Spot-check that a known parameter matches
    for p1, p2 in zip(ae.parameters(), ae2.parameters()):
        assert p1.shape == p2.shape
