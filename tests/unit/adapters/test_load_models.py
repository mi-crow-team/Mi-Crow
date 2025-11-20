import torch


def test_load_huggingface_model_tiny_gpt2(tmp_path):
    from amber.language_model.language_model import LanguageModel
    from amber.store.local_store import LocalStore

    model_id = "sshleifer/tiny-gpt2"
    store = LocalStore(tmp_path / "store")

    lm = LanguageModel.from_huggingface(
        model_id,
        store=store,
        tokenizer_params={"use_fast": True},
        model_params={"dtype": torch.float32},
    )

    # Basic assertions that loading worked and model has layers indexed
    assert lm is not None
    assert hasattr(lm, "model") and hasattr(lm, "tokenizer")
    names = lm.layers.get_layer_names()
    assert isinstance(names, list) and len(names) > 0
