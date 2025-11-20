"""Additional tests for LanguageModelLayers to improve coverage."""
import pytest
import torch
from torch import nn

from amber.language_model.language_model import LanguageModel
from amber.store.local_store import LocalStore
from amber.hooks.controller import Controller
from amber.hooks.hook import HookType
from amber.hooks.implementations.activation_saver import LayerActivationDetector
from amber.mechanistic.sae.modules.topk_sae import TopKSae


class MockTokenizer:
    """Mock tokenizer for testing."""
    
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
    
    def encode(self, text, **kwargs):
        return [ord(c) % 97 + 1 for c in text] if text else [1]
    
    def decode(self, token_ids, **kwargs):
        return "".join(chr(97 + (tid - 1) % 26) for tid in token_ids if tid > 0)
    
    def __len__(self):
        return 100


class MockModel(nn.Module):
    """Mock model with nested structure for testing."""
    
    def __init__(self, vocab_size: int = 100, d_model: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layer1 = nn.Linear(d_model, d_model)
        self.layer2 = nn.Linear(d_model, d_model)
        
        class SimpleConfig:
            def __init__(self):
                self.pad_token_id = None
                self.name_or_path = "MockModel"
        
        self.config = SimpleConfig()
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class TestLanguageModelLayersRegistration:
    """Test hook registration edge cases."""
    
    @pytest.fixture
    def setup_lm(self, tmp_path):
        """Set up LanguageModel for testing."""
        model = MockModel()
        tokenizer = MockTokenizer()
        store = LocalStore(tmp_path)
        return LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    def test_register_hook_with_string_hook_type(self, setup_lm):
        """Test registering hook with string hook_type."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        detector = LayerActivationDetector(layer_signature=layer_name)
        
        hook_id = lm.layers.register_hook(layer_name, detector, hook_type="forward")
        
        assert hook_id == detector.id
    
    def test_register_hook_with_invalid_string_hook_type(self, setup_lm):
        """Test registering hook with invalid string hook_type raises error."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        detector = LayerActivationDetector(layer_signature=layer_name)
        
        with pytest.raises(ValueError, match="Invalid hook_type string"):
            lm.layers.register_hook(layer_name, detector, hook_type="invalid_type")
    
    def test_register_hook_with_duplicate_id_raises_error(self, setup_lm):
        """Test registering hook with duplicate ID raises error."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        detector1 = LayerActivationDetector(layer_signature=layer_name, hook_id="duplicate_id")
        detector2 = LayerActivationDetector(layer_signature=layer_name, hook_id="duplicate_id")
        
        lm.layers.register_hook(layer_name, detector1)
        
        with pytest.raises(ValueError, match="Hook with ID 'duplicate_id' is already registered"):
            lm.layers.register_hook(layer_name, detector2)
    
    def test_register_hook_mixing_detector_and_controller_raises_error(self, setup_lm):
        """Test that mixing Detector and Controller on same layer raises error."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        # Register a detector first
        detector = LayerActivationDetector(layer_signature=layer_name)
        lm.layers.register_hook(layer_name, detector)
        
        # Try to register a controller on the same layer
        topk_sae = TopKSae(n_latents=8, n_inputs=8, k=4, device='cpu')
        
        with pytest.raises(ValueError, match="Cannot register Controller hook"):
            lm.layers.register_hook(layer_name, topk_sae)
    
    def test_register_hook_mixing_controller_and_detector_raises_error(self, setup_lm):
        """Test that mixing Controller and Detector on same layer raises error."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        # Register a controller first
        topk_sae = TopKSae(n_latents=8, n_inputs=8, k=4, device='cpu')
        lm.layers.register_hook(layer_name, topk_sae)
        
        # Try to register a detector on the same layer
        detector = LayerActivationDetector(layer_signature=layer_name)
        
        with pytest.raises(ValueError, match="Cannot register Detector hook"):
            lm.layers.register_hook(layer_name, detector)
    
    def test_register_hook_with_explicit_hook_type(self, setup_lm):
        """Test registering hook with explicit hook_type parameter."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        detector = LayerActivationDetector(layer_signature=layer_name)
        
        hook_id = lm.layers.register_hook(layer_name, detector, hook_type=HookType.FORWARD)
        
        assert hook_id == detector.id


class TestLanguageModelLayersUnregister:
    """Test hook unregistration."""
    
    @pytest.fixture
    def setup_lm(self, tmp_path):
        """Set up LanguageModel for testing."""
        model = MockModel()
        tokenizer = MockTokenizer()
        store = LocalStore(tmp_path)
        return LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    def test_unregister_hook_by_id(self, setup_lm):
        """Test unregistering hook by ID."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        detector = LayerActivationDetector(layer_signature=layer_name)
        hook_id = lm.layers.register_hook(layer_name, detector)
        
        result = lm.layers.unregister_hook(hook_id)
        
        assert result is True
        assert hook_id not in lm.context._hook_id_map
    
    def test_unregister_hook_by_instance(self, setup_lm):
        """Test unregistering hook by Hook instance."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        detector = LayerActivationDetector(layer_signature=layer_name)
        hook_id = lm.layers.register_hook(layer_name, detector)
        
        result = lm.layers.unregister_hook(detector)
        
        assert result is True
        assert hook_id not in lm.context._hook_id_map
    
    def test_unregister_nonexistent_hook(self, setup_lm):
        """Test unregistering nonexistent hook returns False."""
        lm = setup_lm
        
        result = lm.layers.unregister_hook("nonexistent_id")
        
        assert result is False


class TestLanguageModelLayersGetHooks:
    """Test get_hooks method."""
    
    @pytest.fixture
    def setup_lm(self, tmp_path):
        """Set up LanguageModel for testing."""
        model = MockModel()
        tokenizer = MockTokenizer()
        store = LocalStore(tmp_path)
        return LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    def test_get_hooks_all(self, setup_lm):
        """Test getting all hooks."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        if len(layer_names) >= 2:
            layer1 = layer_names[0]
            layer2 = layer_names[1]
            
            detector1 = LayerActivationDetector(layer_signature=layer1)
            detector2 = LayerActivationDetector(layer_signature=layer2)
            
            lm.layers.register_hook(layer1, detector1)
            lm.layers.register_hook(layer2, detector2)
            
            all_hooks = lm.layers.get_hooks()
            
            assert len(all_hooks) >= 2
            assert detector1 in all_hooks
            assert detector2 in all_hooks
    
    def test_get_hooks_by_layer(self, setup_lm):
        """Test getting hooks filtered by layer."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        detector = LayerActivationDetector(layer_signature=layer_name)
        lm.layers.register_hook(layer_name, detector)
        
        hooks = lm.layers.get_hooks(layer_signature=layer_name)
        
        assert len(hooks) >= 1
        assert detector in hooks
    
    def test_get_hooks_by_type(self, setup_lm):
        """Test getting hooks filtered by type."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        detector = LayerActivationDetector(layer_signature=layer_name)
        lm.layers.register_hook(layer_name, detector)
        
        hooks = lm.layers.get_hooks(hook_type=HookType.FORWARD)
        
        assert len(hooks) >= 1
        assert detector in hooks
    
    def test_get_hooks_by_layer_and_type(self, setup_lm):
        """Test getting hooks filtered by both layer and type."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        detector = LayerActivationDetector(layer_signature=layer_name)
        lm.layers.register_hook(layer_name, detector)
        
        hooks = lm.layers.get_hooks(layer_signature=layer_name, hook_type=HookType.FORWARD)
        
        assert len(hooks) >= 1
        assert detector in hooks
    
    def test_get_hooks_with_string_type(self, setup_lm):
        """Test getting hooks with string hook_type."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        detector = LayerActivationDetector(layer_signature=layer_name)
        lm.layers.register_hook(layer_name, detector)
        
        hooks = lm.layers.get_hooks(hook_type="forward")
        
        assert len(hooks) >= 1
        assert detector in hooks


class TestLanguageModelLayersGetControllers:
    """Test get_controllers method."""
    
    @pytest.fixture
    def setup_lm(self, tmp_path):
        """Set up LanguageModel for testing."""
        model = MockModel()
        tokenizer = MockTokenizer()
        store = LocalStore(tmp_path)
        return LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    def test_get_controllers_all(self, setup_lm):
        """Test getting all controllers."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        if len(layer_names) >= 2:
            layer1 = layer_names[0]
            layer2 = layer_names[1]
            
            topk_sae1 = TopKSae(n_latents=8, n_inputs=8, k=4, device='cpu')
            topk_sae2 = TopKSae(n_latents=8, n_inputs=8, k=4, device='cpu')
            
            lm.layers.register_hook(layer1, topk_sae1)
            lm.layers.register_hook(layer2, topk_sae2)
            
            controllers = lm.layers.get_controllers()
            
            assert len(controllers) >= 2
            assert topk_sae1 in controllers
            assert topk_sae2 in controllers
    
    def test_get_controllers_filters_controllers(self, setup_lm):
        """Test that get_controllers only returns Controller instances."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        if len(layer_names) >= 2:
            layer1 = layer_names[0]
            layer2 = layer_names[1]
            
            # Add a controller
            topk_sae = TopKSae(n_latents=8, n_inputs=8, k=4, device='cpu')
            lm.layers.register_hook(layer1, topk_sae)
            
            # Add a detector (should not be in controllers)
            detector = LayerActivationDetector(layer_signature=layer2)
            lm.layers.register_hook(layer2, detector)
            
            controllers = lm.layers.get_controllers()
            
            # Should only contain controllers, not detectors
            assert topk_sae in controllers
            assert detector not in controllers
            assert all(isinstance(c, Controller) for c in controllers)


class TestLanguageModelLayersEdgeCases:
    """Test edge cases in LanguageModelLayers."""
    
    @pytest.fixture
    def setup_lm(self, tmp_path):
        """Set up LanguageModel for testing."""
        model = MockModel()
        tokenizer = MockTokenizer()
        store = LocalStore(tmp_path)
        return LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    def test_get_layer_by_name_not_found_raises_error(self, setup_lm):
        """Test that getting layer by nonexistent name raises error."""
        lm = setup_lm
        
        with pytest.raises(ValueError, match="Layer name 'nonexistent_layer' not found"):
            lm.layers._get_layer_by_name("nonexistent_layer")
    
    def test_get_layer_by_index_not_found_raises_error(self, setup_lm):
        """Test that getting layer by nonexistent index raises error."""
        lm = setup_lm
        
        # Get a large index that doesn't exist
        max_idx = max(lm.layers.idx_to_layer.keys()) if lm.layers.idx_to_layer else -1
        
        with pytest.raises(ValueError, match=f"Layer index '{max_idx + 1}' not found"):
            lm.layers._get_layer_by_index(max_idx + 1)
    
    def test_flatten_layer_names_rebuilds_maps(self, setup_lm):
        """Test that _flatten_layer_names rebuilds the maps."""
        lm = setup_lm
        
        # Clear maps
        lm.layers.name_to_layer.clear()
        lm.layers.idx_to_layer.clear()
        
        # Rebuild
        lm.layers._flatten_layer_names()
        
        assert len(lm.layers.name_to_layer) > 0
        assert len(lm.layers.idx_to_layer) > 0

