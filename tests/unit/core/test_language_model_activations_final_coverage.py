"""Final tests to push coverage over 85% for language_model_activations.py."""
import pytest

from amber.language_model.language_model_activations import LanguageModelActivations


class _FakeContext:
    def __init__(self, model=None, language_model=None, store=None):
        self.model = model
        self.language_model = language_model
        self.store = store


def test_save_activations_dataset_raises_when_model_is_none():
    """Test save_activations_dataset raises ValueError when model is None (line 167)."""
    ctx = _FakeContext(model=None)
    acts = LanguageModelActivations(ctx)
    
    # Create a mock dataset
    class MockDataset:
        def iter_batches(self, batch_size):
            yield ["text1", "text2"]
    
    dataset = MockDataset()
    
    with pytest.raises(ValueError, match="Model must be initialized"):
        acts.save_activations_dataset(dataset, layer_signature="layer1")



