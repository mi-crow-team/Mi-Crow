"""Additional tests to push coverage above 85% for language_model_activations.py."""
import pytest

from amber.core.language_model_activations import LanguageModelActivations


class _FakeContext:
    def __init__(self, model=None, language_model=None, store=None):
        self.model = model
        self.language_model = language_model
        self.store = store


def test_infer_and_save_all_layers_raises_when_model_is_none():
    """Test infer_and_save_all_layers raises ValueError when model is None (line 321)."""
    ctx = _FakeContext(model=None)
    acts = LanguageModelActivations(ctx)
    
    class MockDataset:
        def iter_batches(self, batch_size):
            yield ["text1", "text2"]
    
    dataset = MockDataset()
    
    with pytest.raises(ValueError, match="Model must be initialized"):
        acts.infer_and_save_all_layers(dataset, layer_signatures=["layer1"])



