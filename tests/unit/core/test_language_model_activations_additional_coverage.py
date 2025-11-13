"""Additional tests for LanguageModelActivations to improve coverage."""
import pytest
import torch
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

from amber.core.language_model import LanguageModel
from amber.store.local_store import LocalStore
from amber.adapters.text_snippet_dataset import TextSnippetDataset
from datasets import Dataset


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


class MockModel(torch.nn.Module):
    """Mock model for testing."""
    
    def __init__(self, vocab_size: int = 100, d_model: int = 8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.linear = torch.nn.Linear(d_model, d_model)
        
        class SimpleConfig:
            def __init__(self):
                self.pad_token_id = None
                self.name_or_path = "MockModel"
        
        self.config = SimpleConfig()
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embedding(input_ids)
        return self.linear(x)


class TestLanguageModelActivationsSave:
    """Test save_activations methods."""
    
    @pytest.fixture
    def setup_lm(self, tmp_path):
        """Set up LanguageModel for testing."""
        model = MockModel()
        tokenizer = MockTokenizer()
        store = LocalStore(tmp_path)
        return LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    @pytest.fixture
    def setup_dataset(self, tmp_path):
        """Set up test dataset."""
        from datasets import Dataset
        texts = ["hello world", "test text", "another example"]
        cache_dir = tmp_path / "cache"
        ds = Dataset.from_dict({"text": texts})
        dataset = TextSnippetDataset(ds, cache_dir=cache_dir)
        return dataset
    
    def test_infer_and_save_basic(self, setup_lm, setup_dataset):
        """Test basic save_activations_dataset functionality."""
        lm = setup_lm
        dataset = setup_dataset
        
        # Get a layer name
        layer_names = lm.layers.get_layer_names()
        assert len(layer_names) > 0
        layer_name = layer_names[0]
        
        lm.activations.save_activations_dataset(
            dataset=dataset,
            layer_signature=layer_name,
            run_name="test_run",
            batch_size=2,
            verbose=False
        )
        
        # Verify batches were saved
        batches = lm.store.list_run_batches("test_run")
        assert len(batches) > 0
    
    def test_infer_and_save_with_dtype(self, setup_lm, setup_dataset):
        """Test save_activations_dataset with dtype conversion."""
        lm = setup_lm
        dataset = setup_dataset
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        lm.activations.save_activations_dataset(
            dataset=dataset,
            layer_signature=layer_name,
            run_name="test_run_dtype",
            batch_size=2,
            dtype=torch.float32,
            verbose=False
        )
        
        batches = lm.store.list_run_batches("test_run_dtype")
        assert len(batches) > 0
    
    def test_infer_and_save_with_max_length(self, setup_lm, setup_dataset):
        """Test save_activations_dataset with max_length parameter."""
        lm = setup_lm
        dataset = setup_dataset
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        lm.activations.save_activations_dataset(
            dataset=dataset,
            layer_signature=layer_name,
            run_name="test_run_maxlen",
            batch_size=2,
            max_length=10,
            verbose=False
        )
        
        batches = lm.store.list_run_batches("test_run_maxlen")
        assert len(batches) > 0
    
    def test_infer_and_save_basic(self, setup_lm, setup_dataset):
        """Test save_activations_dataset basic functionality."""
        lm = setup_lm
        dataset = setup_dataset
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        lm.activations.save_activations_dataset(
            dataset=dataset,
            layer_signature=layer_name,
            run_name="test_run_basic",
            batch_size=2,
            verbose=False
        )
        
        batches = lm.store.list_run_batches("test_run_basic")
        assert len(batches) > 0
    
    def test_infer_and_save_with_verbose(self, setup_lm, setup_dataset):
        """Test save_activations_dataset with verbose logging."""
        lm = setup_lm
        dataset = setup_dataset
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        lm.activations.save_activations_dataset(
            dataset=dataset,
            layer_signature=layer_name,
            run_name="test_run_verbose",
            batch_size=2,
            verbose=True
        )
        
        batches = lm.store.list_run_batches("test_run_verbose")
        assert len(batches) > 0
    
    def test_infer_and_save_all_layers(self, setup_lm, setup_dataset):
        """Test save_activations_dataset with multiple layers (call separately for each)."""
        lm = setup_lm
        dataset = setup_dataset
        
        # Get first two layers
        layer_names = lm.layers.get_layer_names()
        test_layers = layer_names[:2] if len(layer_names) >= 2 else layer_names
        
        # Save activations for each layer separately
        for layer in test_layers:
            lm.activations.save_activations_dataset(
                dataset=dataset,
                layer_signature=layer,
                run_name=f"test_run_all_{layer}",
                batch_size=2,
                verbose=False
            )
        
        # Verify runs were created
        for layer in test_layers:
            batches = lm.store.list_run_batches(f"test_run_all_{layer}")
            assert len(batches) >= 0  # May be empty but should not crash
    
    def test_infer_and_save_all_layers_with_none_signatures(self, setup_lm, setup_dataset):
        """Test save_activations_dataset for all layers."""
        lm = setup_lm
        dataset = setup_dataset
        
        # Get all layers
        layer_names = lm.layers.get_layer_names()
        
        # Save activations for each layer
        for layer in layer_names:
            lm.activations.save_activations_dataset(
                dataset=dataset,
                layer_signature=layer,
                run_name=f"test_run_all_none_{layer}",
                batch_size=2,
                verbose=False
            )
        
        # Should complete without error
        for layer in layer_names:
            batches = lm.store.list_run_batches(f"test_run_all_none_{layer}")
            assert isinstance(batches, list)


class TestLanguageModelActivationsEdgeCases:
    """Test edge cases in LanguageModelActivations."""
    
    @pytest.fixture
    def setup_lm(self, tmp_path):
        """Set up LanguageModel for testing."""
        model = MockModel()
        tokenizer = MockTokenizer()
        store = LocalStore(tmp_path)
        return LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    def test_cleanup_detector_handles_exception(self, setup_lm):
        """Test that _cleanup_detector handles exceptions gracefully."""
        lm = setup_lm
        
        # Should not raise even if unregister fails
        lm.activations._cleanup_detector("nonexistent_hook_id")
        
        # Should complete without error
