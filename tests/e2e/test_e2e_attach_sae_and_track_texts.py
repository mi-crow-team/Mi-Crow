"""
End-to-end test for SAE attachment and text tracking.

Based on examples/02_attach_sae_and_save_texts.ipynb - demonstrates
attaching an SAE to a language model and collecting top activating texts.
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
def trained_sae_setup():
    """Set up a trained SAE for testing."""
    temp_dir = tempfile.mkdtemp()
    store_dir = Path(temp_dir) / "store"
    store_dir.mkdir(parents=True)
    
    # Quick training setup
    MODEL_ID = "sshleifer/tiny-gpt2"
    LAYER_SIGNATURE = "gpt2lmheadmodel_transformer_h_0_attn_c_attn"
    RUN_ID = "test_attachment"
    DEVICE = "cpu"
    
    # Load model and create dataset
    lm = LanguageModel.from_huggingface(MODEL_ID)
    lm.model.to(DEVICE)
    
    texts = [
        "The family went to the park.",
        "Animals live in the forest.",
        "The child played with toys.",
        "Birds sing in the morning.",
    ]
    hf_dataset = Dataset.from_dict({"text": texts})
    dataset = TextSnippetDataset(hf_dataset, store_dir / "dataset_cache")
    
    # Save activations
    store = LocalStore(store_dir)
    lm.activations.infer_and_save(
        dataset,
        layer_signature=LAYER_SIGNATURE,
        run_name=RUN_ID,
        store=store,
        batch_size=2,
        autocast=False,
    )
    
    # Get dimensions and create SAE
    first_batch = store.get_run_batch(RUN_ID, 0)
    if isinstance(first_batch, dict):
        activations = first_batch["activations"]
    else:
        activations = first_batch[0]
    hidden_dim = activations.shape[-1]
    
    sae = Autoencoder(
        n_latents=hidden_dim * 2,
        n_inputs=hidden_dim,
        activation="TopK_4",
        tied=False,
        device=DEVICE,
    )
    
    # Quick training
    config = SAETrainingConfig(
        epochs=2,
        batch_size=2,
        lr=1e-3,
        l1_lambda=1e-4,
        device=DEVICE,
        max_batches_per_epoch=2,
        verbose=False,
    )
    trainer = SAETrainer(sae, store, RUN_ID, config)
    trainer.train()
    
    # Save SAE
    sae_path = store_dir / "test_sae.pt"
    sae.save(name="test_sae", path=store_dir)
    
    yield {
        "temp_dir": temp_dir,
        "store_dir": store_dir,
        "sae_path": sae_path,
        "model_id": MODEL_ID,
        "layer_signature": LAYER_SIGNATURE,
        "hidden_dim": hidden_dim,
        "n_latents": sae.context.n_latents,
    }
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_e2e_sae_attachment_and_text_tracking(trained_sae_setup):
    """
    Test complete SAE attachment and text tracking workflow:
    1. Load language model
    2. Load trained SAE
    3. Enable text tracking
    4. Run inference to collect top texts
    5. Verify texts are collected
    6. Export and verify results
    """
    setup = trained_sae_setup
    MODEL_ID = setup["model_id"]
    LAYER_SIGNATURE = setup["layer_signature"]
    SAE_PATH = setup["sae_path"]
    DEVICE = "cpu"
    
    # Step 1: Load language model
    print("\n📥 Loading language model...")
    model = LanguageModel.from_huggingface(MODEL_ID)
    model.model.to(DEVICE)
    
    assert model.model is not None
    print(f"✅ Model loaded: {model.model_id}")
    
    # Step 2: Load trained SAE
    print("\n📥 Loading trained SAE...")
    assert SAE_PATH.exists(), "SAE model not found"
    
    sae, normalize, target_norm, mean = Autoencoder.load_model(SAE_PATH)
    sae.to(DEVICE)
    
    assert sae.context.n_latents == setup["n_latents"]
    print(f"✅ SAE loaded: {setup['hidden_dim']} → {setup['n_latents']} → {setup['hidden_dim']}")
    
    # Step 3: Enable text tracking
    print("\n🔗 Enabling text tracking...")
    sae.context.lm = model
    sae.context.lm_layer_signature = LAYER_SIGNATURE
    
    TOP_K = 5
    sae.enable_text_tracking(k=TOP_K, negative=False)
    
    assert sae.context.text_tracking_enabled
    assert sae.context.text_tracking_k == TOP_K
    print(f"✅ Text tracking enabled: top-{TOP_K} texts per neuron")
    
    # Step 4: Run inference to collect top texts
    print("\n🔍 Running inference to collect top texts...")
    test_texts = [
        "The happy family played together in the park.",
        "Wild animals roam freely in the forest.",
        "The smart child solved the puzzle quickly.",
        "Beautiful birds sing melodious songs.",
        "The brave dog saved the little kitten.",
        "Flowers bloom beautifully in the garden.",
        "The friendly cat slept peacefully.",
        "Children laugh and play with friends.",
    ]
    
    # Run inference in batches
    batch_size = 4
    for i in range(0, len(test_texts), batch_size):
        batch_texts = test_texts[i:i + batch_size]
        model.forwards(batch_texts)
    
    print(f"✅ Processed {len(test_texts)} texts")
    
    # Step 5: Verify texts were collected
    print("\n📊 Verifying collected texts...")
    neurons_with_texts = 0
    total_texts_collected = 0
    
    for neuron_idx in range(setup["n_latents"]):
        top_texts = sae.concepts.get_top_texts_for_neuron(neuron_idx)
        if top_texts:
            neurons_with_texts += 1
            total_texts_collected += len(top_texts)
            
            # Verify structure of collected texts
            for nt in top_texts:
                assert hasattr(nt, "text")
                assert hasattr(nt, "score")
                assert isinstance(nt.text, str)
                assert isinstance(nt.score, (int, float))
                assert len(nt.text) > 0
    
    assert neurons_with_texts > 0, "No texts were collected for any neuron"
    print(f"✅ Collected texts for {neurons_with_texts}/{setup['n_latents']} neurons")
    print(f"📊 Total texts collected: {total_texts_collected}")
    
    # Step 6: Export and verify
    print("\n💾 Exporting top texts...")
    output_path = Path(setup["temp_dir"]) / "top_texts.json"
    sae.concepts.export_top_texts_to_json(str(output_path))
    
    assert output_path.exists(), "Top texts JSON was not created"
    
    # Verify file is valid JSON
    import json
    with open(output_path, "r") as f:
        exported_data = json.load(f)
    
    assert isinstance(exported_data, dict)
    assert len(exported_data) > 0
    print(f"✅ Exported top texts to: {output_path}")
    
    # Step 7: Verify specific neuron's top texts
    print("\n🔍 Examining collected texts...")
    for neuron_idx in range(min(3, setup["n_latents"])):
        top_texts = sae.concepts.get_top_texts_for_neuron(neuron_idx)
        if top_texts:
            print(f"\n🧠 Neuron {neuron_idx}: {len(top_texts)} texts")
            for j, nt in enumerate(top_texts[:2]):
                print(f"   {j+1}. '{nt.text[:50]}...' (score: {nt.score:.4f})")
            
            # Verify texts are sorted by score (descending)
            scores = [nt.score for nt in top_texts]
            assert scores == sorted(scores, reverse=True), "Texts not sorted by score"
    
    print("\n🎉 E2E SAE attachment and text tracking completed successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

