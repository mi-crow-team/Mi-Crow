"""Additional tests to improve coverage for layers.py."""
import torch
from torch import nn
from pathlib import Path
import tempfile

from amber.language_model.language_model import LanguageModel
from amber.hooks.controller import Controller
from amber.hooks.detector import Detector
from amber.hooks.hook import HookType, HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT
from amber.store.local_store import LocalStore


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
        # Controller constructor: hook_type, hook_id, layer_signature
        super().__init__(HookType.FORWARD, hook_id, layer_signature)
        self.modified = False
    
    def modify_activations(self, module, inputs: torch.Tensor | None, output: torch.Tensor | None) -> torch.Tensor | None:
        if isinstance(output, torch.Tensor):
            self.modified = True
            return output * 1.1  # Simple scaling
        return output


class SimpleDetector(Detector):
    """Simple detector for testing."""
    
    def __init__(self, layer_signature, hook_id=None):
        # Detector constructor: hook_type, hook_id, store, layer_signature
        super().__init__(HookType.FORWARD, hook_id, None, layer_signature)
        self.captured = None
    
    def process_activations(self, module, input: HOOK_FUNCTION_INPUT, output: HOOK_FUNCTION_OUTPUT) -> None:
        """Required abstract method implementation."""
        if isinstance(output, torch.Tensor):
            self.captured = output.clone()


def test_get_controllers():
    """Test get_controllers method (line 459->461)."""
    model = MockModel()
    tokenizer = MockTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / "store")
    lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    # Get actual layer name from model
    layer_names = lm.layers.get_layer_names()
    assert len(layer_names) > 0
    layer_name = layer_names[0]
    
    # Register controllers
    controller1 = SimpleController(layer_name, hook_id="ctrl1")
    controller2 = SimpleController(layer_name, hook_id="ctrl2")
    
    lm.layers.register_hook(layer_name, controller1)
    lm.layers.register_hook(layer_name, controller2)
    
    # Get controllers
    controllers = lm.layers.get_controllers()
    
    # Should return all controllers
    assert len(controllers) >= 2
    assert controller1 in controllers
    assert controller2 in controllers
    assert all(isinstance(c, Controller) for c in controllers)


def test_get_detectors():
    """Test get_detectors method (line 463->465)."""
    model = MockModel()
    tokenizer = MockTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / "store")
    lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    # Get actual layer name from model
    layer_names = lm.layers.get_layer_names()
    assert len(layer_names) > 0
    layer_name = layer_names[0]
    
    # Register detectors
    detector1 = SimpleDetector(layer_name, hook_id="det1")
    detector2 = SimpleDetector(layer_name, hook_id="det2")
    
    lm.layers.register_hook(layer_name, detector1)
    lm.layers.register_hook(layer_name, detector2)
    
    # Get detectors
    detectors = lm.layers.get_detectors()
    
    # Should return all detectors
    assert len(detectors) >= 2
    assert detector1 in detectors
    assert detector2 in detectors
    assert all(isinstance(d, Detector) for d in detectors)


def test_enable_hook():
    """Test enable_hook method (lines 417->431)."""
    model = MockModel()
    tokenizer = MockTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / "store")
    lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    # Get actual layer name from model
    layer_names = lm.layers.get_layer_names()
    assert len(layer_names) > 0
    layer_name = layer_names[0]
    
    # Register a controller and disable it
    controller = SimpleController(layer_name, hook_id="test_ctrl")
    hook_id = lm.layers.register_hook(layer_name, controller)
    controller.disable()
    
    assert not controller.enabled
    
    # Enable via enable_hook
    result = lm.layers.enable_hook(hook_id)
    
    assert result is True
    assert controller.enabled
    
    # Test with non-existent hook_id
    result = lm.layers.enable_hook("non_existent")
    assert result is False


def test_disable_hook():
    """Test disable_hook method (lines 433->447)."""
    model = MockModel()
    tokenizer = MockTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / "store")
    lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    # Get actual layer name from model
    layer_names = lm.layers.get_layer_names()
    assert len(layer_names) > 0
    layer_name = layer_names[0]
    
    # Register a controller
    controller = SimpleController(layer_name, hook_id="test_ctrl")
    hook_id = lm.layers.register_hook(layer_name, controller)
    
    assert controller.enabled
    
    # Disable via disable_hook
    result = lm.layers.disable_hook(hook_id)
    
    assert result is True
    assert not controller.enabled
    
    # Test with non-existent hook_id
    result = lm.layers.disable_hook("non_existent")
    assert result is False


def test_enable_all_hooks():
    """Test enable_all_hooks method (lines 449->452)."""
    model = MockModel()
    tokenizer = MockTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / "store")
    lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    # Get actual layer name from model
    layer_names = lm.layers.get_layer_names()
    assert len(layer_names) > 0
    layer_name = layer_names[0]
    
    # Register multiple hooks and disable them
    controller1 = SimpleController(layer_name, hook_id="ctrl1")
    controller2 = SimpleController(layer_name, hook_id="ctrl2")
    
    lm.layers.register_hook(layer_name, controller1)
    lm.layers.register_hook(layer_name, controller2)
    
    controller1.disable()
    controller2.disable()
    
    assert not controller1.enabled
    assert not controller2.enabled
    
    # Enable all
    lm.layers.enable_all_hooks()
    
    assert controller1.enabled
    assert controller2.enabled


def test_disable_all_hooks():
    """Test disable_all_hooks method (lines 454->457)."""
    model = MockModel()
    tokenizer = MockTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / "store")
    lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    # Get actual layer name from model
    layer_names = lm.layers.get_layer_names()
    assert len(layer_names) > 0
    layer_name = layer_names[0]
    
    # Register multiple hooks
    controller1 = SimpleController(layer_name, hook_id="ctrl1")
    controller2 = SimpleController(layer_name, hook_id="ctrl2")
    
    lm.layers.register_hook(layer_name, controller1)
    lm.layers.register_hook(layer_name, controller2)
    
    assert controller1.enabled
    assert controller2.enabled
    
    # Disable all
    lm.layers.disable_all_hooks()
    
    assert not controller1.enabled
    assert not controller2.enabled

