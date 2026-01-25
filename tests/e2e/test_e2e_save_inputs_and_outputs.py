
"""
End-to-end test for saving model inputs and outputs.

Based on examples/04_save_inputs_and_outputs.ipynb - demonstrates
capturing and saving model inputs (input_ids) and outputs (logits).
"""


import pytest


import torch


from pathlib import Path


import tempfile


import shutil


from datetime import datetime



from mi_crow.hooks import ModelInputDetector, ModelOutputDetector


from mi_crow.language_model.language_model import LanguageModel


from mi_crow.store.local_store import LocalStore




@pytest.fixture


def temp_dirs():


    """Create temporary directories for test artifacts."""


    temp_dir = tempfile.mkdtemp()


    store_dir = Path(temp_dir) / "store"


    store_dir.mkdir(parents=True)



    yield {
        "temp_dir": temp_dir,
        "store_dir": store_dir,
    }




    shutil.rmtree(temp_dir, ignore_errors=True)




def test_e2e_save_inputs_and_outputs(temp_dirs):


    """
    Test complete input/output saving workflow:
    1. Load language model
    2. Attach ModelInputDetector and ModelOutputDetector
    3. Run inference and capture inputs/outputs
    4. Inspect captured data
    5. Save to store
    6. Verify saved data
    """


    store_dir = temp_dirs["store_dir"]




    MODEL_ID = "sshleifer/tiny-gpt2"


    DEVICE = "cpu"



    TEST_TEXTS = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
    ]




    print("\n📥 Loading language model...")


    store = LocalStore(store_dir)


    lm = LanguageModel.from_huggingface(MODEL_ID, store=store)


    lm.model.to(DEVICE)



    assert lm.model is not None


    assert lm.model_id == "sshleifer_tiny-gpt2"


    print(f"✅ Model loaded: {lm.model_id}")


    print(f"📱 Device: {DEVICE}")


    print(f"📁 Store location: {lm.context.store.base_path}")




    print("\n🔧 Creating ModelInputDetector and ModelOutputDetector...")




    input_layer_signature = "model_inputs"


    output_layer_signature = "model_outputs"




    root_model = lm.model




    if input_layer_signature not in lm.layers.name_to_layer:


        lm.layers.name_to_layer[input_layer_signature] = root_model


        print(f"📝 Added '{input_layer_signature}' to layers registry")



    if output_layer_signature not in lm.layers.name_to_layer:


        lm.layers.name_to_layer[output_layer_signature] = root_model


        print(f"📝 Added '{output_layer_signature}' to layers registry")




    input_detector = ModelInputDetector(
        layer_signature=input_layer_signature,
        hook_id="model_input_detector",
        save_input_ids=True,
        save_attention_mask=False,
    )




    output_detector = ModelOutputDetector(
        layer_signature=output_layer_signature,
        hook_id="model_output_detector",
        save_output_logits=True,
        save_output_hidden_state=False,
    )




    input_hook_id = lm.layers.register_hook(input_layer_signature, input_detector)


    output_hook_id = lm.layers.register_hook(output_layer_signature, output_detector)



    assert input_detector.id == "model_input_detector"


    assert output_detector.id == "model_output_detector"


    print(f"✅ Detectors attached to model via layers system")


    print(f"🆔 Input detector ID: {input_detector.id}")


    print(f"🆔 Output detector ID: {output_detector.id}")


    print(f"💾 Will save: input_ids, output_logits")




    print("\n🚀 Running inference...")


    print(f"📝 Processing {len(TEST_TEXTS)} texts")




    input_detector.clear_captured()


    output_detector.clear_captured()




    output, encodings = lm.inference.infer_texts(
        TEST_TEXTS,
        run_name=None,
        batch_size=None,
        tok_kwargs={"max_length": 128, "padding": True, "truncation": True},
        autocast=False,
    )



    assert output is not None


    assert isinstance(encodings, dict)


    assert "input_ids" in encodings


    print("✅ Inference completed")


    print(f"📊 Output type: {type(output)}")


    print(f"📊 Encodings keys: {list(encodings.keys())}")




    print("\n🔍 Inspecting captured data...")




    input_ids = input_detector.get_captured_input_ids()


    assert input_ids is not None, "No input_ids captured"


    assert input_ids.shape[0] == len(TEST_TEXTS)


    assert input_ids.dtype == torch.int64


    print(f"✅ Captured input_ids (from ModelInputDetector):")


    print(f"   Shape: {input_ids.shape}")


    print(f"   Dtype: {input_ids.dtype}")


    print(f"   Sample (first 10 tokens of first text): {input_ids[0, :10].tolist()}")




    output_logits = output_detector.get_captured_output_logits()


    assert output_logits is not None, "No output_logits captured"


    assert output_logits.shape[0] == len(TEST_TEXTS)


    assert output_logits.shape[1] == input_ids.shape[1]


    assert output_logits.dtype == torch.float32


    print(f"✅ Captured output_logits (from ModelOutputDetector):")


    print(f"   Shape: {output_logits.shape}")


    print(f"   Dtype: {output_logits.dtype}")


    print(f"   Vocabulary size: {output_logits.shape[-1]}")




    assert "input_ids_shape" in input_detector.metadata


    assert "output_logits_shape" in output_detector.metadata




    print("\n🔤 Decoding captured data...")




    for i, text in enumerate(TEST_TEXTS):


        decoded = lm.tokenizer.decode(input_ids[i], skip_special_tokens=False)


        assert isinstance(decoded, str)


        assert len(decoded) > 0




    predicted_token_ids = output_logits.argmax(dim=-1)


    assert predicted_token_ids.shape == input_ids.shape


    for i in range(len(TEST_TEXTS)):


        last_token_id = predicted_token_ids[i, -1].item()


        predicted_token = lm.tokenizer.decode([last_token_id], skip_special_tokens=True)


        assert isinstance(predicted_token, str)



    print("✅ Decoding works correctly")




    print("\n💾 Saving detector outputs to store...")



    run_name = f"model_io_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


    batch_idx = 0




    saved_path = lm.save_detector_metadata(run_name, batch_idx)



    assert saved_path is not None


    print(f"✅ Detector outputs saved to store")


    print(f"📁 Run name: {run_name}")


    print(f"📁 Batch index: {batch_idx}")


    print(f"📁 Saved path: {saved_path}")




    print("\n🔍 Verifying saved data...")




    retrieved_metadata, retrieved_tensors = lm.store.get_detector_metadata(run_name, batch_idx)



    assert len(retrieved_metadata) >= 2, "Should have metadata for both layers"


    assert len(retrieved_tensors) >= 2, "Should have tensors for both layers"




    assert input_layer_signature in retrieved_tensors


    assert output_layer_signature in retrieved_tensors



    saved_input_ids = retrieved_tensors[input_layer_signature].get("input_ids")


    saved_output_logits = retrieved_tensors[output_layer_signature].get("output_logits")



    assert saved_input_ids is not None, "input_ids not found in saved data"


    assert saved_output_logits is not None, "output_logits not found in saved data"



    assert saved_input_ids.shape == input_ids.shape


    assert saved_output_logits.shape == output_logits.shape




    assert torch.equal(saved_input_ids, input_ids)


    assert torch.allclose(saved_output_logits, output_logits)



    print(f"✅ Loaded metadata for {len(retrieved_metadata)} layer(s)")


    print(f"✅ Loaded tensors for {len(retrieved_tensors)} layer(s)")


    print(f"✅ Verified saved data matches captured data")




    print("\n🧹 Cleaning up...")




    lm.layers.unregister_hook(input_hook_id)


    lm.layers.unregister_hook(output_hook_id)




    input_detector.clear_captured()


    output_detector.clear_captured()



    print("✅ Hooks removed and data cleared")



    print("\n🎉 E2E input/output saving workflow completed successfully!")




if __name__ == "__main__":


    pytest.main([__file__, "-v", "-s"])



