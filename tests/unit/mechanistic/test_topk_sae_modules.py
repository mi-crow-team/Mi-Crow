"""Tests for TopKSAE in modules directory (inherits from Sae)."""

import torch
import pytest

from amber.mechanistic.sae.modules.topk_sae import TopKSae
from amber.language_model.language_model import LanguageModel


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


class MockModel(torch.nn.Module):
    """Test model."""
    
    def __init__(self, vocab_size: int = 100, d_model: int = 8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.linear = torch.nn.Linear(d_model, d_model)
        
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


@pytest.fixture
def mock_lm(tmp_path):
    """Create a mock language model."""
    from amber.store.local_store import LocalStore
    model = MockModel()
    tokenizer = MockTokenizer()
    store = LocalStore(tmp_path / 'store')
    return LanguageModel(model=model, tokenizer=tokenizer, store=store)


def test_topk_sae_modules_basic():
    """Test basic TopKSAE functionality."""
    
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4)
    
    assert topk_sae.k == 4
    assert topk_sae.context.n_latents == 8
    assert topk_sae.context.n_inputs == 16
    assert topk_sae.sae_engine is not None


def test_topk_sae_modules_encode_decode():
    """Test TopKSAE encode and decode."""
    
    torch.manual_seed(42)
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4)
    
    x = torch.randn(5, 16)
    
    # Test encode
    encoded = topk_sae.encode(x)
    assert encoded.shape == (5, 8)
    
    # Test decode
    reconstructed = topk_sae.decode(encoded)
    assert reconstructed.shape == (5, 16)
    
    # Test forward
    recon = topk_sae.forward(x)
    assert recon.shape == (5, 16)


def test_topk_sae_modules_save_load(tmp_path, mock_lm):
    """Test TopKSAE save and load."""
    
    torch.manual_seed(42)
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4, device='cpu')
    
    # Set some context metadata
    topk_sae.context.lm_layer_signature = "test_layer"
    topk_sae.context.model_id = "test_model"
    
    # Set concept manipulation parameters
    topk_sae.concepts.multiplication.data = torch.ones(8) * 2.0
    topk_sae.concepts.bias.data = torch.zeros(8)
    
    # Save
    save_path = tmp_path / "test_model.pt"
    topk_sae.save("test_model", tmp_path)
    assert save_path.exists()
    
    # Load
    loaded = TopKSae.load(save_path)
    
    assert loaded.k == 4
    assert loaded.context.n_latents == 8
    assert loaded.context.n_inputs == 16
    assert loaded.context.lm_layer_signature == "test_layer"
    assert loaded.context.model_id == "test_model"
    
    # Check concepts state
    assert torch.allclose(loaded.concepts.multiplication.data, torch.ones(8) * 2.0)
    assert torch.allclose(loaded.concepts.bias.data, torch.zeros(8))


def test_topk_sae_modules_save_load_with_top_texts(tmp_path, mock_lm):
    """Test TopKSAE save and load - top texts are not saved/loaded (export separately)."""
    
    torch.manual_seed(42)
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4, device='cpu')
    
    # Set context for text tracking
    layer_names = mock_lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "mockmodel_linear"
    
    topk_sae.context.lm = mock_lm
    topk_sae.context.lm_layer_signature = valid_layer
    topk_sae.context.text_tracking_k = 5
    topk_sae.context.text_tracking_enabled = True
    
    # Enable text tracking and populate with data
    topk_sae.concepts.enable_text_tracking()
    
    # Manually populate concepts with test data
    import heapq
    topk_sae.concepts._ensure_heaps(8)
    topk_sae.concepts._top_texts_heaps[0] = []
    topk_sae.concepts._top_texts_heaps[1] = []
    
    # Add some test entries
    heapq.heappush(topk_sae.concepts._top_texts_heaps[0], (-5.0, (5.0, "text1", 0)))
    heapq.heappush(topk_sae.concepts._top_texts_heaps[0], (-3.0, (3.0, "text2", 1)))
    heapq.heappush(topk_sae.concepts._top_texts_heaps[1], (-4.0, (4.0, "text1", 0)))
    
    # Save
    save_path = tmp_path / "test_model.pt"
    topk_sae.save("test_model", tmp_path)
    assert save_path.exists()
    
    # Load with language model
    loaded = TopKSae.load(save_path)
    loaded.context.lm = mock_lm  # Set LM for text tracking
    
    # Top texts are NOT saved/loaded with the model (export separately if needed)
    # The loaded model should have empty heaps until text tracking is enabled and inference is run
    assert loaded.concepts._top_texts_heaps is None or len(loaded.concepts._top_texts_heaps) == 0
    
    # Verify model structure is correct
    assert loaded.k == 4
    assert loaded.context.n_latents == 8
    assert loaded.context.n_inputs == 16


def test_topk_sae_modules_save_load_without_top_texts(tmp_path):
    """Test TopKSAE save and load without top texts."""
    
    torch.manual_seed(42)
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4, device='cpu')
    
    # Save without text tracking
    save_path = tmp_path / "test_model.pt"
    topk_sae.save("test_model", tmp_path)
    assert save_path.exists()
    
    # Load
    loaded = TopKSae.load(save_path)
    
    # Should not have top texts heaps
    assert loaded.concepts._top_texts_heaps is None


def test_topk_sae_modules_modify_activations():
    """Test TopKSAE modify_activations."""
    
    torch.manual_seed(42)
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4, device='cpu')
    topk_sae.sae_engine.eval()  # Set eval mode on the underlying engine
    
    # Create test tensor
    x = torch.randn(2, 3, 16)
    
    # Get hook function
    
    class DummyModule:
        pass
    
    module = DummyModule()
    output = x
    
    # Call modify_activations directly
    modified = topk_sae.modify_activations(module, (), output)
    
    # Should return tensor of same shape
    assert modified.shape == x.shape
    
    # Should be different from original (reconstruction)
    assert not torch.allclose(modified, x, atol=1e-5)


def test_topk_sae_modules_modify_activations_with_concepts():
    """Test TopKSAE modify_activations with concept manipulation."""
    
    torch.manual_seed(42)
    topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4, device='cpu')
    topk_sae.sae_engine.eval()  # Set eval mode on the underlying engine
    
    # Set concept manipulation parameters
    topk_sae.concepts.multiplication.data = torch.ones(8) * 2.0  # Double all latents
    topk_sae.concepts.bias.data = torch.zeros(8)
    
    # Create test tensor
    x = torch.randn(2, 3, 16)
    
    # Get hook function
    
    class DummyModule:
        pass
    
    module = DummyModule()
    output = x
    
    # Call modify_activations directly
    modified = topk_sae.modify_activations(module, (), output)
    
    # Should return tensor of same shape
    assert modified.shape == x.shape
    
    # Should be different from original
    assert not torch.allclose(modified, x, atol=1e-5)


def test_topk_sae_modules_load_invalid_format(tmp_path):
    """Test TopKSAE load with invalid format raises error."""
    
    # Create payload in old format (without amber_metadata wrapper)
    old_payload = {
        "sae_engine_state": None,
        "concepts_state": {
            'multiplication': torch.ones(8) * 1.5,
            'bias': torch.ones(8) * 0.5,
        },
        "n_latents": 8,
        "n_inputs": 16,
        "k": 4,
    }
    
    save_path = tmp_path / "old_model.pt"
    torch.save(old_payload, save_path)
    
    # Should raise ValueError for invalid format
    with pytest.raises(ValueError, match="Invalid TopKSAE save format"):
        TopKSae.load(save_path)

