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
from amber.mechanistic.autoencoder.modules.topk_sae import TopKSae
from amber.mechanistic.autoencoder.sae_trainer import SaeTrainingConfig


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
    
    sae = TopKSae(
        n_latents=hidden_dim * 4,  # 4x expansion
        n_inputs=hidden_dim,
        k=4,
        device=DEVICE,
    )
    
    assert sae.context.n_latents == hidden_dim * 4
    assert sae.context.n_inputs == hidden_dim
    print(f"🧠 TopKSAE architecture: {hidden_dim} → {sae.context.n_latents} → {hidden_dim} (k={sae.k})")
    
    # Step 5: Train TopKSAE
    print("\n🏋️ Training TopKSAE...")
    config = SaeTrainingConfig(
        epochs=3,
        batch_size=4,
        lr=1e-3,
        l1_lambda=1e-4,
        device=DEVICE,
        max_batches_per_epoch=2,  # Small for testing
        verbose=False,  # Reduce noise in tests
    )
    
    history = sae.train(store, RUN_ID, config)
    
    # Verify training completed
    assert "loss" in history
    assert "recon_mse" in history
    assert "l1" in history
    assert len(history["loss"]) > 0
    print(f"✅ Training completed! Final loss: {history['loss'][-1]:.6f}")
    
    # Step 6: Save TopKSAE model
    print("\n💾 Saving TopKSAE model...")
    sae_path = store_dir / "topk_sae_model.pt"
    
    sae.save(
        name="topk_sae_model",
        path=store_dir,
    )
    
    assert sae_path.exists(), "TopKSAE model was not saved"
    print(f"✅ TopKSAE saved to: {sae_path}")
    
    # Step 7: Load TopKSAE and verify it works
    print("\n📥 Loading TopKSAE model...")
    loaded_sae = TopKSae.load(sae_path)
    loaded_sae.sae_engine.to(DEVICE)  # Move underlying engine to device
    
    assert loaded_sae.context.n_latents == sae.context.n_latents
    assert loaded_sae.context.n_inputs == sae.context.n_inputs
    assert loaded_sae.k == sae.k
    print("✅ TopKSAE loaded successfully")
    
    # Step 8: Test reconstruction
    print("\n🔬 Testing reconstruction...")
    # TopKSAE expects 2D input: (batch_size, n_inputs)
    test_input = torch.randn(8, hidden_dim).to(DEVICE)
    
    with torch.inference_mode():
        # TopKSAE forward returns reconstructed tensor
        reconstructed = loaded_sae.forward(test_input)
    
    assert reconstructed.shape == test_input.shape
    assert reconstructed.shape == (8, hidden_dim)
    print("✅ Reconstruction works correctly")
    
    print("\n🎉 E2E SAE training workflow completed successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

