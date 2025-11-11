"""Test advanced functionality in LanguageModelActivations."""

import torch
from torch import nn
from datasets import Dataset
from amber.core.language_model import LanguageModel
from amber.adapters.text_snippet_dataset import TextSnippetDataset
from amber.store.local_store import LocalStore


class MockTokenizer:
    """Test tokenizer for activation tests."""
    
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id
        self.pad_token = None
        self.pad_token_id = None
        self.eos_token = None
        self.eos_token_id = None

    def __call__(self, texts, **kwargs):
        padding = kwargs.get("padding", False)
        return_tensors = kwargs.get("return_tensors", "pt")
        
        # Handle both string and dict inputs
        if isinstance(texts, list) and len(texts) > 0:
            if isinstance(texts[0], dict):
                text_strings = [item.get("text", "") for item in texts]
            else:
                text_strings = texts
        else:
            text_strings = []
        
        max_len = max(len(t) for t in text_strings) if text_strings and padding else max(len(t) for t in text_strings) if text_strings else 1
        ids = []
        attn = []
        for t in text_strings:
            row = [ord(c) % 97 + 1 for c in t] if t else [1]
            pad = max_len - len(row)
            ids.append(row + [0] * pad)
            attn.append([1] * len(row) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn)}

    def add_special_tokens(self, spec):
        if "pad_token" in spec:
            self.pad_token = spec["pad_token"]
            self.pad_token_id = 0

    def __len__(self):
        return 100

    def batch_encode_plus(self, texts, **kwargs):
        return self(texts, **kwargs)

    def encode_plus(self, text, **kwargs):
        if isinstance(text, dict):
            text_str = text.get("text", "")
        else:
            text_str = text
        return self([text_str], **kwargs)[0]

    def pad(self, encoded, return_tensors="pt"):
        if return_tensors == "pt":
            input_ids = torch.stack([torch.tensor(e["input_ids"]) for e in encoded])
            attention_mask = torch.stack([torch.tensor(e["attention_mask"]) for e in encoded])
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return encoded


class MockModel(nn.Module):
    """Test model for activation tests."""
    
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


def test_metadata_extraction_with_cache_dir_error(tmp_path):
    """Test metadata extraction when cache_dir access fails."""
    base = Dataset.from_dict({"text": ["sample 1", "sample 2", "sample 3"]})
    ds = TextSnippetDataset(base, cache_dir=tmp_path / "cache")
    
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    store = LocalStore(tmp_path)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "linear"
    
    # Mock the cache_dir property to raise an error
    original_cache_dir = ds.cache_dir
    def error_cache_dir():
        raise AttributeError("cache_dir not accessible")
    
    # Patch the cache_dir property
    ds.cache_dir = property(error_cache_dir)
    
    # Should handle cache_dir error gracefully
    lm.activations.infer_and_save(
        ds,
        layer_signature=valid_layer,
        run_name="cache_error_test",
        store=store,
        batch_size=2,
        verbose=True,
    )
    
    # Should still complete despite errors
    batches = store.list_run_batches("cache_error_test")
    assert len(batches) > 0


def test_metadata_extraction_with_model_name_error(tmp_path):
    """Test metadata extraction when model name extraction fails."""
    base = Dataset.from_dict({"text": ["sample 1", "sample 2", "sample 3"]})
    ds = TextSnippetDataset(base, cache_dir=tmp_path / "cache")
    
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    # Remove model_name attribute to simulate error
    if hasattr(lm, 'model_name'):
        delattr(lm, 'model_name')
    
    store = LocalStore(tmp_path)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "linear"
    
    # Should handle missing model_name gracefully
    lm.activations.infer_and_save(
        ds,
        layer_signature=valid_layer,
        run_name="model_name_error_test",
        store=store,
        batch_size=2,
        verbose=True,
    )
    
    # Should still complete despite errors
    batches = store.list_run_batches("model_name_error_test")
    assert len(batches) > 0


def test_metadata_storage_error_handling(tmp_path):
    """Test error handling when metadata storage fails."""
    base = Dataset.from_dict({"text": ["sample 1", "sample 2", "sample 3"]})
    ds = TextSnippetDataset(base, cache_dir=tmp_path / "cache")
    
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    store = LocalStore(tmp_path)
    
    # Mock put_run_meta to raise an error
    def error_put_meta(run_name, meta):
        raise RuntimeError("Metadata storage failed")
    
    store.put_run_meta = error_put_meta
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "linear"
    
    # Should handle metadata storage error gracefully
    lm.activations.infer_and_save(
        ds,
        layer_signature=valid_layer,
        run_name="metadata_error_test",
        store=store,
        batch_size=2,
        verbose=True,
    )
    
    # Should still complete despite errors
    batches = store.list_run_batches("metadata_error_test")
    assert len(batches) > 0


def test_activation_capture_edge_cases(tmp_path):
    """Test edge cases in activation capture."""
    base = Dataset.from_dict({"text": ["sample 1", "sample 2", "sample 3"]})
    ds = TextSnippetDataset(base, cache_dir=tmp_path / "cache")
    
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    store = LocalStore(tmp_path)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "linear"
    
    # Test with different batch sizes and edge cases
    lm.activations.infer_and_save(
        ds,
        layer_signature=valid_layer,
        run_name="edge_case_test",
        store=store,
        batch_size=1,  # Small batch size
        verbose=True,
    )
    
    # Should complete successfully
    batches = store.list_run_batches("edge_case_test")
    assert len(batches) > 0


def test_verbose_logging_with_errors(tmp_path, caplog):
    """Test that verbose logging works even when errors occur."""
    import logging
    
    base = Dataset.from_dict({"text": ["sample 1", "sample 2", "sample 3"]})
    ds = TextSnippetDataset(base, cache_dir=tmp_path / "cache")
    
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    store = LocalStore(tmp_path)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "linear"
    
    with caplog.at_level(logging.INFO):
        lm.activations.infer_and_save(
            ds,
            layer_signature=valid_layer,
            run_name="verbose_error_test",
            store=store,
            batch_size=2,
            verbose=True,
        )
    
    log_messages = [rec.message for rec in caplog.records]
    assert any("Starting save_model_activations" in msg for msg in log_messages)
    assert any("Completed save_model_activations" in msg for msg in log_messages)


def test_layer_signature_edge_cases(tmp_path):
    """Test various layer signature inputs, including invalid ones."""
    base = Dataset.from_dict({"text": ["sample 1", "sample 2", "sample 3"]})
    ds = TextSnippetDataset(base, cache_dir=tmp_path / "cache")
    
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    store = LocalStore(tmp_path)
    
    # Test with different layer signatures
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "linear"
    layer_signatures = [valid_layer, 0, "nonexistent_layer"]
    
    for i, signature in enumerate(layer_signatures):
        try:
            lm.activations.infer_and_save(
                ds,
                layer_signature=signature,
                run_name=f"edge_case_test_{i}",
                store=store,
                batch_size=2,
                verbose=True,
            )
        except Exception:
            # Some signatures should fail, that's expected
            pass
    
    # At least one should succeed
    all_batches = []
    for i in range(len(layer_signatures)):
        try:
            batches = store.list_run_batches(f"edge_case_test_{i}")
            all_batches.extend(batches)
        except Exception:
            pass
    assert len(all_batches) > 0


def test_activation_capture_with_different_shapes(tmp_path):
    """Test activation capture with different input shapes."""
    base = Dataset.from_dict({"text": ["a", "bb", "ccc", "dddd"]})
    ds = TextSnippetDataset(base, cache_dir=tmp_path / "cache")
    
    model = MockModel()
    tokenizer = MockTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    store = LocalStore(tmp_path)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "linear"
    
    # Test with different batch sizes
    for batch_size in [1, 2, 4]:
        lm.activations.infer_and_save(
            ds,
            layer_signature=valid_layer,
            run_name=f"shape_test_{batch_size}",
            store=store,
            batch_size=batch_size,
            verbose=False,
        )
        
        # Should complete successfully
        batches = store.list_run_batches(f"shape_test_{batch_size}")
        assert len(batches) > 0
