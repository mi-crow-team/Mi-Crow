
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



from mi_crow.language_model.language_model import LanguageModel


from mi_crow.datasets import TextDataset


from mi_crow.store.local_store import LocalStore


from mi_crow.store.local_store import LocalStore



from mi_crow.mechanistic.sae.modules.topk_sae import TopKSae, TopKSaeTrainingConfig




@pytest.fixture


def temp_dirs():


    """Create temporary directories for test artifacts."""


    temp_dir = tempfile.mkdtemp()


    store_dir = Path(temp_dir) / "store"


    dataset_dir = Path(temp_dir) / "cache"


    store_dir.mkdir(parents=True)


    dataset_dir.mkdir(parents=True)



    yield {
        "temp_dir": temp_dir,
        "store_dir": store_dir,
        "dataset_dir": dataset_dir,
    }




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


    dataset_dir = temp_dirs["dataset_dir"]




    MODEL_ID = "sshleifer/tiny-gpt2"


    LAYER_SIGNATURE = "gpt2lmheadmodel_transformer_h_0_attn_c_attn"


    RUN_ID = "test_sae_training"


    DEVICE = "cpu"




    print("\n📥 Loading language model...")


    store = LocalStore(store_dir)


    lm = LanguageModel.from_huggingface(MODEL_ID, store=store)


    lm.model.to(DEVICE)



    assert lm.model is not None


    assert lm.model_id == "sshleifer_tiny-gpt2"


    print(f"✅ Model loaded: {lm.model_id}")




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


    dataset_store = LocalStore(base_path=dataset_dir)


    dataset = TextDataset(hf_dataset, store=dataset_store)



    assert len(dataset) == len(texts)


    print(f"✅ Created dataset with {len(dataset)} samples")




    print("\n💾 Saving activations...")


    store = LocalStore(store_dir)



    lm.activations.save_activations_dataset(
        dataset,
        layer_signature=LAYER_SIGNATURE,
        run_name=RUN_ID,
        batch_size=4,
        autocast=False,
    )




    batches = store.list_run_batches(RUN_ID)


    assert len(batches) > 0, "No activations were saved"


    print(f"✅ Saved {len(batches)} batches of activations")




    print("\n🏗️ Creating SAE model...")


    first_batch = store.get_run_batch(RUN_ID, 0)


    if isinstance(first_batch, dict):


        activations = first_batch["activations"]


    else:


        activations = first_batch[0]



    hidden_dim = activations.shape[-1]


    print(f"📏 Hidden dimension: {hidden_dim}")



    sae = TopKSae(
        n_latents=hidden_dim * 4,
        n_inputs=hidden_dim,
        device=DEVICE,
    )



    assert sae.context.n_latents == hidden_dim * 4


    assert sae.context.n_inputs == hidden_dim




    print("\n🏋️ Training TopKSAE...")


    config = TopKSaeTrainingConfig(
        k=4,
        epochs=3,
        batch_size=4,
        lr=1e-3,
        l1_lambda=1e-4,
        device=DEVICE,
        max_batches_per_epoch=2,
        verbose=False,
    )



    result = sae.train(store, RUN_ID, LAYER_SIGNATURE, config)




    assert "history" in result


    assert "training_run_id" in result


    history = result["history"]


    assert "loss" in history


    assert "recon_mse" in history


    assert "l1" in history


    assert len(history["loss"]) > 0


    print(f"✅ Training completed! Final loss: {history['loss'][-1]:.6f}")




    print("\n💾 Saving TopKSAE model...")


    sae_path = store_dir / "topk_sae_model.pt"



    sae.save(
        name="topk_sae_model",
        path=store_dir,
    )



    assert sae_path.exists(), "TopKSAE model was not saved"


    print(f"✅ TopKSAE saved to: {sae_path}")




    print("\n📥 Loading TopKSAE model...")


    loaded_sae = TopKSae.load(sae_path)


    loaded_sae.sae_engine.to(DEVICE)



    assert loaded_sae.context.n_latents == sae.context.n_latents


    assert loaded_sae.context.n_inputs == sae.context.n_inputs



    loaded_k = getattr(loaded_sae.sae_engine, 'top_k', None)


    sae_k = getattr(sae.sae_engine, 'top_k', None)


    assert loaded_k == sae_k == 4, f"Expected k=4, got loaded_k={loaded_k}, sae_k={sae_k}"


    print("✅ TopKSAE loaded successfully")




    print("\n🔬 Testing reconstruction...")



    test_input = torch.randn(8, hidden_dim).to(DEVICE)



    with torch.inference_mode():



        reconstructed = loaded_sae.forward(test_input)



    assert reconstructed.shape == test_input.shape


    assert reconstructed.shape == (8, hidden_dim)


    print("✅ Reconstruction works correctly")



    print("\n🎉 E2E SAE training workflow completed successfully!")




if __name__ == "__main__":


    pytest.main([__file__, "-v", "-s"])



