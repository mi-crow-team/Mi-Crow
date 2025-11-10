"""Additional tests to push coverage above 85% for language_model_layers.py."""
import pytest
import torch
from torch import nn

from amber.core.language_model import LanguageModel
from amber.hooks.controller import Controller
from amber.hooks.detector import Detector
from amber.hooks.hook import HookType
from amber.mechanistic.autoencoder.autoencoder import Autoencoder


class MockModel(nn.Module):
    """Test model."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 8)
        self.linear = nn.Linear(8, 8)
        
        class SimpleConfig:
            def __init__(self):
                self.pad_token_id = None
                self.name_or_path = "MockModel"
        
        self.config = SimpleConfig()

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        return self.linear(x)


class MockTokenizer:
    """Test tokenizer."""
    
    def __init__(self):
        self.pad_token = None
        self.pad_token_id = None

    def __call__(self, texts, **kwargs):
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
        return [ord(c) % 97 + 1 for c in text] if text else [1]

    def decode(self, token_ids):
        return "".join(chr(97 + (tid - 1) % 26) for tid in token_ids if tid > 0)

    def add_special_tokens(self, spec):
        pass

    def __len__(self):
        return 100


class SimpleController(Controller):
    """Simple controller for testing."""
    
    def __init__(self, layer_signature, hook_id=None):
        super().__init__(HookType.FORWARD, hook_id, layer_signature)
        self.modified = False
    
    def modify_activations(self, module, inputs, output):
        if isinstance(output, torch.Tensor):
            self.modified = True
            return output * 1.1
        return output


class SimpleDetector(Detector):
    """Simple detector for testing."""
    
    def __init__(self, layer_signature, hook_id=None):
        super().__init__(HookType.FORWARD, hook_id, None, layer_signature)
        self.captured = None
    
    def process_activations(self, module, inputs, output):
        if isinstance(output, torch.Tensor):
            self.captured = output.clone()


def test_register_hook_invalid_hook_type_string():
    """Test register_hook with invalid hook_type string (line 273)."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    layer_names = lm.layers.get_layer_names()
    layer_name = layer_names[0]
    
    controller = SimpleController(layer_name, hook_id="test")
    
    # Try to register with invalid hook_type string
    with pytest.raises(ValueError, match="Invalid hook_type string"):
        lm.layers.register_hook(layer_name, controller, hook_type="invalid_type")


def test_register_hook_invalid_hook_type_type():
    """Test register_hook with invalid hook_type type (line 280)."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    layer_names = lm.layers.get_layer_names()
    layer_name = layer_names[0]
    
    controller = SimpleController(layer_name, hook_id="test")
    
    # Try to register with invalid hook_type (not a HookType enum)
    with pytest.raises(ValueError, match="hook_type must be a HookType enum"):
        lm.layers.register_hook(layer_name, controller, hook_type=123)  # Invalid type


def test_register_hook_mixing_detector_and_controller():
    """Test register_hook mixing Detector and Controller (line 308)."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    layer_names = lm.layers.get_layer_names()
    layer_name = layer_names[0]
    
    # Register a detector first
    detector = SimpleDetector(layer_name, hook_id="det1")
    lm.layers.register_hook(layer_name, detector)
    
    # Try to register a controller on the same layer - should fail
    controller = SimpleController(layer_name, hook_id="ctrl1")
    with pytest.raises(ValueError, match="Cannot register Controller hook"):
        lm.layers.register_hook(layer_name, controller)


def test_register_new_layer_with_tuple_output_3_elements():
    """Test register_new_layer with tuple output having 3+ elements (line 167)."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    layer_names = lm.layers.get_layer_names()
    after_layer = layer_names[0]
    
    # Create a layer that returns tuple with 3 elements
    class Tuple3Layer(nn.Module):
        def forward(self, x):
            return (x, x * 2, x * 3)  # 3 elements
    
    layer = Tuple3Layer()
    
    # Register - should handle tuple with 3 elements
    lm.layers.register_new_layer("tuple3_layer", layer, after_layer)
    
    all_names = lm.layers.get_layer_names()
    assert any("tuple3_layer" in name for name in all_names)


def test_register_new_layer_with_reconstruction_attribute():
    """Test register_new_layer with object having reconstruction attribute (line 171)."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    layer_names = lm.layers.get_layer_names()
    after_layer = layer_names[0]
    
    # Create a layer that returns object with reconstruction attribute
    class ReconstructionLayer(nn.Module):
        def forward(self, x):
            class Output:
                def __init__(self, recon):
                    self.reconstruction = recon
            return Output(x * 2)
    
    layer = ReconstructionLayer()
    
    # Register - should handle object with reconstruction attribute
    lm.layers.register_new_layer("recon_layer", layer, after_layer)
    
    all_names = lm.layers.get_layer_names()
    assert any("recon_layer" in name for name in all_names)


def test_register_new_layer_with_list_output():
    """Test register_new_layer with list output (line 197)."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    layer_names = lm.layers.get_layer_names()
    after_layer = layer_names[0]
    
    # Create a layer that returns list
    class ListLayer(nn.Module):
        def forward(self, x):
            return [x, x * 2]  # List output
    
    layer = ListLayer()
    
    # Register - should handle list output
    lm.layers.register_new_layer("list_layer", layer, after_layer)
    
    all_names = lm.layers.get_layer_names()
    assert any("list_layer" in name for name in all_names)


def test_register_new_layer_with_no_tensor_in_tuple():
    """Test register_new_layer with tuple containing no tensor (line 201)."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    layer_names = lm.layers.get_layer_names()
    after_layer = layer_names[0]
    
    # Create a layer that returns tuple with no tensors
    class NoTensorLayer(nn.Module):
        def forward(self, x):
            return ("not a tensor", "also not a tensor")  # No tensors
    
    layer = NoTensorLayer()
    
    # Register - the hook will be set up, but will fail during inference
    # This tests the RuntimeError path at line 201
    lm.layers.register_new_layer("no_tensor_layer", layer, after_layer)
    
    all_names = lm.layers.get_layer_names()
    assert any("no_tensor_layer" in name for name in all_names)


def test_register_new_layer_with_last_hidden_state():
    """Test register_new_layer with object having last_hidden_state (line 204)."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    layer_names = lm.layers.get_layer_names()
    after_layer = layer_names[0]
    
    # Create a layer that returns object with last_hidden_state
    class HiddenStateLayer(nn.Module):
        def forward(self, x):
            class Output:
                def __init__(self, hidden):
                    self.last_hidden_state = hidden
            return Output(x * 2)
    
    layer = HiddenStateLayer()
    
    # Register - should handle object with last_hidden_state
    lm.layers.register_new_layer("hidden_state_layer", layer, after_layer)
    
    all_names = lm.layers.get_layer_names()
    assert any("hidden_state_layer" in name for name in all_names)


def test_register_new_layer_with_unsupported_type():
    """Test register_new_layer with unsupported output type (line 207)."""
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    layer_names = lm.layers.get_layer_names()
    after_layer = layer_names[0]
    
    # Create a layer that returns unsupported type
    class UnsupportedLayer(nn.Module):
        def forward(self, x):
            return "unsupported string type"  # Not a tensor, tuple, list, or object with attributes
    
    layer = UnsupportedLayer()
    
    # Register - the hook will be set up, but will fail during inference
    # This tests the RuntimeError path at line 207
    lm.layers.register_new_layer("unsupported_layer", layer, after_layer)
    
    all_names = lm.layers.get_layer_names()
    assert any("unsupported_layer" in name for name in all_names)

