"""Tests that verify actual behavior of TopKSae, not just compilation."""
import pytest
import torch
from pathlib import Path
import tempfile

from amber.mechanistic.sae.modules.topk_sae import TopKSae
from amber.language_model.language_model import LanguageModel
from amber.store.local_store import LocalStore


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


class TestTopKSaeEncodeSparsity:
    """Test encode sparsity verification."""
    
    def test_encode_returns_at_most_k_non_zero_values(self):
        """Verify encode returns at most k non-zero values per sample (TopK sparsity)."""
        torch.manual_seed(42)
        topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4)
        topk_sae.sae_engine.eval()
        
        x = torch.randn(5, 16)
        encoded = topk_sae.encode(x)
        
        # Each row should have at most k non-zero values (TopK sparsity)
        assert encoded.shape == (5, 8)
        for i in range(5):
            non_zero = (encoded[i] != 0).sum().item()
            assert non_zero <= 4, f"Row {i} has {non_zero} non-zero values, expected at most 4"
            # Most rows should have k non-zero, but some may have fewer due to implementation details
            assert non_zero > 0, f"Row {i} should have at least some non-zero values"

    def test_encode_sparsity_is_topk(self):
        """Verify encode returns sparse activations (TopK sparsity)."""
        torch.manual_seed(42)
        topk_sae = TopKSae(n_latents=8, n_inputs=16, k=3)
        topk_sae.sae_engine.eval()
        
        x = torch.randn(3, 16)
        encoded = topk_sae.encode(x)
        
        # Each row should have at most k non-zero values (TopK sparsity)
        for i in range(3):
            non_zero = (encoded[i] != 0).sum().item()
            assert non_zero <= 3, f"Row {i} has {non_zero} non-zero values, expected at most 3"
            assert non_zero > 0, f"Row {i} should have at least some non-zero values"
            
            # Verify sparsity: most values should be zero
            total_values = encoded[i].numel()
            sparsity = (non_zero / total_values) * 100
            assert sparsity <= (3 / 8) * 100, f"Row {i} should be sparse (at most 3/8 non-zero)"


class TestTopKSaeModifyActivations:
    """Test modify_activations behavior."""
    
    def test_modify_activations_uses_pre_codes_for_text_tracking(self):
        """Verify modify_activations passes pre_codes (full activations) to text tracking."""
        torch.manual_seed(42)
        topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4)
        topk_sae.sae_engine.eval()
        
        # Set up language model and enable text tracking
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / 'store')
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        topk_sae.context.lm = lm
        topk_sae.context.lm_layer_signature = "test_layer"
        topk_sae.context.text_tracking_enabled = True
        topk_sae.context.text_tracking_k = 5
        topk_sae._text_tracking_enabled = True
        
        # Enable text tracking
        tracker = lm._ensure_input_tracker()
        tracker.enable()
        topk_sae.concepts.enable_text_tracking()
        
        # Set texts
        texts = ["test text"]
        tracker.set_current_texts(texts)
        
        # Create test tensor
        x = torch.randn(1, 3, 16)  # [B, T, D]
        
        # Mock update_top_texts_from_latents to capture what's passed
        captured_latents = []
        original_update = topk_sae.concepts.update_top_texts_from_latents
        
        def mock_update(latents, texts, original_shape):
            captured_latents.append(latents.clone())
            return original_update(latents, texts, original_shape)
        
        topk_sae.concepts.update_top_texts_from_latents = mock_update
        
        # Call modify_activations
        class DummyModule:
            pass
        module = DummyModule()
        modified = topk_sae.modify_activations(module, (), x)
        
        # Verify pre_codes (full activations) were passed, not sparse codes
        assert len(captured_latents) > 0
        passed_latents = captured_latents[0]
        
        # Get pre_codes and codes for comparison
        x_flat = x.reshape(-1, 16)
        pre_codes, codes = topk_sae.sae_engine.encode(x_flat)
        
        # Verify passed latents match pre_codes (full activations)
        assert torch.allclose(passed_latents, pre_codes.detach().cpu())
        
        # Verify passed latents are NOT sparse codes (should have more non-zeros)
        non_zero_pre = (pre_codes != 0).sum().item()
        non_zero_codes = (codes != 0).sum().item()
        assert non_zero_pre > non_zero_codes, "pre_codes should have more non-zeros than sparse codes"

    def test_modify_activations_3d_reshaping(self):
        """Verify 3D inputs are correctly reshaped and restored."""
        torch.manual_seed(42)
        topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4)
        topk_sae.sae_engine.eval()
        
        x = torch.randn(2, 3, 16)  # [B, T, D]
        original_shape = x.shape
        
        class DummyModule:
            pass
        module = DummyModule()
        
        modified = topk_sae.modify_activations(module, (), x)
        
        # Should return tensor of same shape
        assert modified.shape == original_shape
        assert modified.shape == (2, 3, 16)

    def test_modify_activations_return_format_matches_input_tuple(self):
        """Verify return format matches input format for tuple output."""
        torch.manual_seed(42)
        topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4)
        topk_sae.sae_engine.eval()
        
        # Create tuple output
        x = torch.randn(2, 16)
        output_tuple = (x, torch.tensor([1, 2]))
        
        class DummyModule:
            pass
        module = DummyModule()
        
        modified = topk_sae.modify_activations(module, (), output_tuple)
        
        # Should return tuple with first element modified
        assert isinstance(modified, tuple)
        assert len(modified) == 2
        assert isinstance(modified[0], torch.Tensor)
        assert modified[0].shape == x.shape
        assert isinstance(modified[1], torch.Tensor)  # Second element unchanged

    def test_modify_activations_return_format_matches_input_list(self):
        """Verify return format matches input format for list output."""
        torch.manual_seed(42)
        topk_sae = TopKSae(n_latents=8, n_inputs=16, k=4)
        topk_sae.sae_engine.eval()
        
        # Create list output
        x = torch.randn(2, 16)
        output_list = [x, torch.tensor([1, 2])]
        
        class DummyModule:
            pass
        module = DummyModule()
        
        modified = topk_sae.modify_activations(module, (), output_list)
        
        # Should return list with first element modified
        assert isinstance(modified, list)
        assert len(modified) == 2
        assert isinstance(modified[0], torch.Tensor)
        assert modified[0].shape == x.shape

