import pytest
import torch


def test_load_huggingface_model_tiny_gpt2():
    from amber.core.language_model import LanguageModel

    model_id = "sshleifer/tiny-gpt2"

    try:
        lm = LanguageModel.from_huggingface(
            model_id,
            tokenizer_params={"use_fast": True},
            model_params={"dtype": torch.float32},
        )
    except Exception as e:
        pytest.skip(f"Skipping HF load test due to environment/network issue: {e}")
        return

    # Basic assertions that loading worked and model has layers indexed
    assert lm is not None
    assert hasattr(lm, "model") and hasattr(lm, "tokenizer")
    names = lm.layers.get_layer_names()
    assert isinstance(names, list) and len(names) > 0
