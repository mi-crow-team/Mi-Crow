"""Test advanced functionality in TopNeuronTexts."""

import pytest
import torch
from torch import nn
from unittest.mock import Mock
from amber.core.language_model import LanguageModel
from amber.mechanistic.autoencoder.concepts.top_neuron_texts import TopNeuronTexts
from amber.mechanistic.autoencoder.autoencoder import Autoencoder


def create_top_neuron_texts(lm, layer_signature, k=5, nth_tensor=1):
    """Helper to create TopNeuronTexts with proper context."""
    sae = Autoencoder(n_latents=8, n_inputs=8)
    context = sae.context
    context.lm = lm
    context.lm_layer_signature = layer_signature
    context.text_tracking_k = k
    return TopNeuronTexts(context, k=k, nth_tensor=nth_tensor)


class MockTokenizer:
    """Test tokenizer for TopNeuronTexts tests."""
    
    def __init__(self):
        self.pad_token = None
        self.pad_token_id = None
        self.eos_token = None
        self.eos_token_id = None

    def __call__(self, texts, **kwargs):
        # Simple tokenization for testing
        if isinstance(texts, str):
            texts = [texts]
        
        max_len = max(len(t) for t in texts) if texts else 1
        ids = []
        attn = []
        for t in texts:
            row = [ord(c) % 97 + 1 for c in t] if t else [1]
            pad = max_len - len(row)
            ids.append(row + [0] * pad)
            attn.append([1] * len(row) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn)}

    def encode(self, text, add_special_tokens=False):
        """Mock encode method."""
        return [ord(c) % 97 + 1 for c in text] if text else [1]

    def decode(self, token_ids):
        """Mock decode method."""
        return "".join(chr(97 + (tid - 1) % 26) for tid in token_ids if tid > 0)

    def add_special_tokens(self, spec):
        pass

    def __len__(self):
        return 100


class MockModel(nn.Module):
    """Test model for TopNeuronTexts tests."""
    
    def __init__(self, vocab_size: int = 100, d_model: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, d_model)
        
        # Create a config with proper string attributes
        class SimpleConfig:
            def __init__(self):
                self.pad_token_id = None
                self.name_or_path = "MockModel"
        
        self.config = SimpleConfig()

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        return self.linear(x)

    def resize_token_embeddings(self, new_size):
        pass


def test_top_neuron_texts_initialization_error():
    """Test TopNeuronTexts initialization with invalid k value."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    # Test with invalid k value
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "mockmodel_linear"
    
    # Create a context for TopNeuronTexts
    from amber.mechanistic.autoencoder.autoencoder_context import AutoencoderContext
    from amber.mechanistic.autoencoder.autoencoder import Autoencoder
    
    sae = Autoencoder(n_latents=4, n_inputs=8)
    context = sae.context
    context.lm = lm
    context.lm_layer_signature = valid_layer
    context.text_tracking_k = 0  # Invalid k
    
    with pytest.raises(ValueError, match="k must be positive"):
        TopNeuronTexts(context, k=5)  # k from context.text_tracking_k will be used and is 0
    
    context.text_tracking_k = -1  # Invalid k
    with pytest.raises(ValueError, match="k must be positive"):
        TopNeuronTexts(context, k=5)  # k from context.text_tracking_k will be used and is -1


def test_token_decode_with_none_tokenizer():
    """Test token decoding when tokenizer is None."""
    model = MockModel()
    tokenizer = None
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "mockmodel_linear"
    
    tracker = create_top_neuron_texts(lm, valid_layer, k=3)
    
    # Test decode with None tokenizer
    result = tracker._decode_token("test", 0)
    assert result == "<token_0>"
    
    result = tracker._decode_token("test", 5)
    assert result == "<token_5>"


def test_token_decode_with_tokenization_error():
    """Test token decoding when tokenization fails."""
    class ErrorTokenizer:
        def encode(self, text, add_special_tokens=False):
            raise Exception("Tokenization failed")
        
        def decode(self, token_ids):
            raise Exception("Decode failed")
    
    model = MockModel()
    tokenizer = ErrorTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "mockmodel_linear"
    tracker = create_top_neuron_texts(lm, valid_layer, k=3)
    
    # Test decode with tokenization error
    result = tracker._decode_token("test", 0)
    assert result == "<token_0_decode_error>"


def test_token_decode_with_out_of_range_index():
    """Test token decoding with out of range token index."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "mockmodel_linear"
    tracker = create_top_neuron_texts(lm, valid_layer, k=3)
    
    # Test decode with out of range index
    result = tracker._decode_token("ab", 5)  # Index 5 is out of range for "ab"
    # Error message format may vary (out_of_range or decode_error)
    assert "<token_5" in result and ("out_of_range" in result or "decode_error" in result)


def test_activations_hook_with_tuple_output():
    """Test activations hook with tuple output."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    # Create a model that returns tuple output
    class TupleOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 8)
            self.linear = nn.Linear(8, 8)
        
        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            output = self.linear(x)
            return (output, {"metadata": "test"})
    
    tuple_model = TupleOutputModel()
    lm_tuple = LanguageModel(model=tuple_model, tokenizer=tokenizer)
    
    # Find a valid layer name
    layer_names = lm_tuple.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "tupleoutputmodel_linear"
    tracker = create_top_neuron_texts(lm_tuple, valid_layer, k=3, nth_tensor=0)
    
    # Test with tuple output
    x = torch.randint(0, 50, (2, 3))
    output = tuple_model(x)
    
    # Manually call the hook
    tracker.process_activations(tuple_model, (), output)
    
    # Should have captured activations
    assert tracker.last_activations is not None


def test_activations_hook_with_insufficient_tensors():
    """Test activations hook with insufficient tensors in tuple."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    # Create a model that returns tuple with insufficient tensors
    class InsufficientTupleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 8)
            self.linear = nn.Linear(8, 8)
        
        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            output = self.linear(x)
            return (output,)  # Only one tensor, but we need nth_tensor=1
    
    insufficient_model = InsufficientTupleModel()
    lm_insufficient = LanguageModel(model=insufficient_model, tokenizer=tokenizer)
    
    # Find a valid layer name
    layer_names = lm_insufficient.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "insufficienttuplemodel_linear"
    tracker = create_top_neuron_texts(lm_insufficient, valid_layer, k=3, nth_tensor=1)
    
    # Test with insufficient tensors
    x = torch.randint(0, 50, (2, 3))
    output = insufficient_model(x)
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="Expected at least 2 tensors in output, got 1"):
        tracker.process_activations(insufficient_model, (), output)


def test_activations_hook_with_last_hidden_state():
    """Test activations hook with object having last_hidden_state attribute."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    # Create a model that returns object with last_hidden_state
    class ObjectWithHiddenStateModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 8)
            self.linear = nn.Linear(8, 8)
        
        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            output = self.linear(x)
            
            class OutputObject:
                def __init__(self, hidden_state):
                    self.last_hidden_state = hidden_state
            
            return OutputObject(output)
    
    object_model = ObjectWithHiddenStateModel()
    lm_object = LanguageModel(model=object_model, tokenizer=tokenizer)
    
    # Find a valid layer name
    layer_names = lm_object.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "objectwithhiddenstatemodel_linear"
    tracker = create_top_neuron_texts(lm_object, valid_layer, k=3)
    
    # Test with object output
    x = torch.randint(0, 50, (2, 3))
    output = object_model(x)
    
    # Manually call the hook
    tracker.process_activations(object_model, (), output)
    
    # Should have captured activations
    assert tracker.last_activations is not None


def test_activations_hook_with_none_output():
    """Test activations hook with None output."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "mockmodel_linear"
    tracker = create_top_neuron_texts(lm, valid_layer, k=3)
    
    # Test with None output
    tracker.process_activations(model, (), None)
    
    # Should not have captured activations
    assert tracker.last_activations is None


def test_activations_hook_with_exception_in_reduce():
    """Test activations hook when _reduce_over_tokens raises exception."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "mockmodel_linear"
    tracker = create_top_neuron_texts(lm, valid_layer, k=3)
    
    # Mock _reduce_over_tokens to raise exception
    def mock_reduce(activations):
        raise Exception("Reduce failed")
    
    tracker._reduce_over_tokens = mock_reduce
    
    # Test with exception in reduce
    x = torch.randint(0, 50, (2, 3))
    output = model(x)
    
    # Should handle exception gracefully
    tracker.process_activations(model, (), output)
    
    # Should still have activations stored
    assert tracker.last_activations is not None


def test_activations_hook_with_1d_scores():
    """Test activations hook with 1D scores tensor."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "mockmodel_linear"
    tracker = create_top_neuron_texts(lm, valid_layer, k=3)
    
    # Mock _reduce_over_tokens to return 1D tensors
    def mock_reduce(activations):
        scores = torch.randn(3)  # 1D tensor
        indices = torch.randint(0, 10, (3,))
        return scores, indices
    
    tracker._reduce_over_tokens = mock_reduce
    
    # Test with 1D scores
    x = torch.randint(0, 50, (2, 3))
    output = model(x)
    
    # Should handle 1D scores by adding batch dimension
    tracker.process_activations(model, (), output)
    
    # Should have captured activations
    assert tracker.last_activations is not None


def test_register_activation_text_tracker_exception():
    """Test registration with exception handling."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    # Mock register_activation_text_tracker to raise exception
    original_register = lm.register_activation_text_tracker
    def mock_register(tracker):
        raise Exception("Registration failed")
    
    lm.register_activation_text_tracker = mock_register
    
    # Should handle registration exception gracefully
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "mockmodel_linear"
    tracker = create_top_neuron_texts(lm, valid_layer, k=3)
    
    # Should still be created despite registration failure
    assert tracker is not None
    assert tracker.context.text_tracking_k == 3
    
    # Restore original method
    lm.register_activation_text_tracker = original_register
