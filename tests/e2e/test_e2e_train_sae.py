"""
End-to-end test for SAE training workflow.

Based on examples/01_train_sae_model.ipynb - demonstrates the complete
workflow of training a Sparse Autoencoder on model activations.
"""
import pytest
import torch
from pathlib import Path
import tempfile
import shutil
from datasets import Dataset

from amber.core.language_model import LanguageModel
from amber.adapters.text_snippet_dataset import TextSnippetDataset
from amber.store import LocalStore
from amber.mechanistic.autoencoder.autoencoder import Autoencoder
from amber.mechanistic.autoencoder.train import SAETrainer, SAETrainingConfig


@pytest.fixture
def temp_dirs():
    """Create temporary directories for test artifacts."""
    temp_dir = tempfile.mkdtemp()
    store_dir = Path(temp_dir) / "store"
    cache_dir = Path(temp_dir) / "cache"
    store_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    
    yield {
        "temp_dir": temp_dir,
        "store_dir": store_dir,
        "cache_dir": cache_dir,
    }
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_e2e_train_sae_workflow(temp_dirs):
    """
    Test complete SAE training workflow:
    1. Load language model
    2. Create dataset
    3. Save activations
    4. Train SAE
    5. Save and load SAE model
    6. Verify reconstruction works
    """
    store_dir = temp_dirs["store_dir"]
    cache_dir = temp_dirs["cache_dir"]
    
    # Configuration
    MODEL_ID = "sshleifer/tiny-gpt2"
    LAYER_SIGNATURE = "gpt2lmheadmodel_transformer_h_0_attn_c_attn"
    RUN_ID = "test_sae_training"
    DEVICE = "cpu"
    
    # Step 1: Load language model
    print("\n📥 Loading language model...")
    lm = LanguageModel.from_huggingface(MODEL_ID)
    lm.model.to(DEVICE)
    
    assert lm.model is not None
    assert lm.model_id == "sshleifer_tiny-gpt2"
    print(f"✅ Model loaded: {lm.model_id}")
    
    # Step 2: Create small dataset
    print("\n📥 Creating dataset...")
    texts = [
        "The cat sat on the mat.",
        "Dogs are loyal animals.",
        "The sun shines brightly.",
        "Water flows in the river.",
        "Birds fly in the sky.",
        "Trees grow in the forest.",
        "Stars twinkle at night.",
        "Fish swim in the ocean.",
    ]
    hf_dataset = Dataset.from_dict({"text": texts})
    dataset = TextSnippetDataset(hf_dataset, cache_dir)
    
    assert len(dataset) == len(texts)
    print(f"✅ Created dataset with {len(dataset)} samples")
    
    # Step 3: Save activations
    print("\n💾 Saving activations...")
    store = LocalStore(store_dir)
    
    lm.activations.infer_and_save(
        dataset,
        layer_signature=LAYER_SIGNATURE,
        run_name=RUN_ID,
        store=store,
        batch_size=4,
        autocast=False,
    )
    
    # Verify activations were saved
    batches = store.list_run_batches(RUN_ID)
    assert len(batches) > 0, "No activations were saved"
    print(f"✅ Saved {len(batches)} batches of activations")
    
    # Step 4: Get activation dimensions and create SAE
    print("\n🏗️ Creating SAE model...")
    first_batch = store.get_run_batch(RUN_ID, 0)
    if isinstance(first_batch, dict):
        activations = first_batch["activations"]
    else:
        activations = first_batch[0]
    
    hidden_dim = activations.shape[-1]
    print(f"📏 Hidden dimension: {hidden_dim}")
    
    sae = Autoencoder(
        n_latents=hidden_dim * 2,  # 2x expansion
        n_inputs=hidden_dim,
        activation="TopK_4",
        tied=False,
        init_method="kaiming",
        device=DEVICE,
    )
    
    assert sae.context.n_latents == hidden_dim * 2
    assert sae.context.n_inputs == hidden_dim
    print(f"🧠 SAE architecture: {hidden_dim} → {sae.context.n_latents} → {hidden_dim}")
    
    # Step 5: Train SAE
    print("\n🏋️ Training SAE...")
    config = SAETrainingConfig(
        epochs=3,
        batch_size=4,
        lr=1e-3,
        l1_lambda=1e-4,
        device=DEVICE,
        max_batches_per_epoch=2,  # Small for testing
        project_decoder_grads=True,
        renorm_decoder_every=1,
        verbose=False,  # Reduce noise in tests
    )
    
    trainer = SAETrainer(sae, store, RUN_ID, config)
    history = trainer.train()
    
    # Verify training completed
    assert "loss" in history
    assert "recon_mse" in history
    assert "l1" in history
    assert len(history["loss"]) > 0
    print(f"✅ Training completed! Final loss: {history['loss'][-1]:.6f}")
    
    # Step 6: Save SAE model
    print("\n💾 Saving SAE model...")
    sae_path = store_dir / "test_sae_model.pt"
    metadata = {
        "hidden_dim": hidden_dim,
        "n_latents": sae.context.n_latents,
        "training_history": history,
    }
    
    sae.save(
        name="test_sae_model",
        path=store_dir,
        run_metadata=metadata,
    )
    
    assert sae_path.exists(), "SAE model was not saved"
    print(f"✅ SAE saved to: {sae_path}")
    
    # Step 7: Load SAE and verify it works
    print("\n📥 Loading SAE model...")
    loaded_sae, normalize, target_norm, mean = Autoencoder.load_model(sae_path)
    loaded_sae.to(DEVICE)
    
    assert loaded_sae.context.n_latents == sae.context.n_latents
    assert loaded_sae.context.n_inputs == sae.context.n_inputs
    print("✅ SAE loaded successfully")
    
    # Step 8: Test reconstruction
    print("\n🔬 Testing reconstruction...")
    # SAE expects 2D input: (batch_size, n_inputs)
    test_input = torch.randn(8, hidden_dim).to(DEVICE)
    
    with torch.inference_mode():
        result = loaded_sae(test_input)
    
    # SAE forward may return either tensor or tuple (reconstruction, latents, ...)
    if isinstance(result, tuple):
        reconstructed = result[0]
    else:
        reconstructed = result
    
    assert reconstructed.shape == test_input.shape
    assert reconstructed.shape == (8, hidden_dim)
    print("✅ Reconstruction works correctly")
    
    print("\n🎉 E2E SAE training workflow completed successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

