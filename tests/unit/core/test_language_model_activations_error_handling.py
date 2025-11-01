import pytest
import torch
from torch import nn
from datasets import Dataset
from unittest.mock import Mock, patch

from amber.core.language_model import LanguageModel
from amber.adapters.text_snippet_dataset import TextSnippetDataset
from amber.store import LocalStore


class ErrorProneTokenizer:
    """Tokenizer that can raise various errors for testing."""
    
    def __init__(self, error_type=None):
        self.error_type = error_type
        self.pad_token = None
        self.pad_token_id = None
        self.eos_token = None
        self.eos_token_id = None
    
    def __call__(self, texts, **kwargs):
        if self.error_type == "tokenization_error":
            raise ValueError("Tokenization failed")
        elif self.error_type == "return_tensors_error":
            # Always raise error for this error type
            raise ValueError("Only return_tensors='pt' supported")
        
        # Normal tokenization - handle both strings and dicts
        padding = kwargs.get("padding", False)
        
        # Extract text strings from input
        if isinstance(texts, list) and len(texts) > 0:
            if isinstance(texts[0], dict):
                # Handle dict format from dataset
                text_strings = [item.get("text", "") for item in texts]
            else:
                # Handle string format
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
        """Mock batch_encode_plus method."""
        return self(texts, **kwargs)
    
    def encode_plus(self, text, **kwargs):
        """Mock encode_plus method."""
        # Handle both string and dict inputs
        if isinstance(text, dict):
            text_str = text.get("text", "")
        else:
            text_str = text
        return self([text_str], **kwargs)[0]
    
    def pad(self, encoded, return_tensors="pt"):
        """Mock pad method."""
        if return_tensors == "pt":
            input_ids = torch.stack([torch.tensor(e["input_ids"]) for e in encoded])
            attention_mask = torch.stack([torch.tensor(e["attention_mask"]) for e in encoded])
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return encoded


class ErrorProneModel(nn.Module):
    """Model that can raise various errors for testing."""
    
    def __init__(self, error_type=None):
        super().__init__()
        self.emb = nn.Embedding(100, 4)
        self.lin = nn.Linear(4, 4)
        self.error_type = error_type
        
        # Create a config with proper string attributes
        class SimpleConfig:
            def __init__(self):
                self.pad_token_id = None
                self.name_or_path = "ErrorProneModel"
        
        self.config = SimpleConfig()
    
    def forward(self, input_ids, attention_mask=None):
        if self.error_type == "forward_error":
            raise RuntimeError("Forward pass failed")
        elif self.error_type == "device_error":
            # Try to access device that doesn't exist
            return self.emb(input_ids).to("nonexistent_device")
        
        x = self.emb(input_ids)
        return self.lin(x)
    
    def resize_token_embeddings(self, new_size):
        """Mock resize_token_embeddings method."""
        pass


class ErrorProneDataset:
    """Dataset that can raise various errors for testing."""
    
    def __init__(self, error_type=None):
        self.error_type = error_type
        self._length = 3
    
    def __len__(self):
        if self.error_type == "length_error":
            raise RuntimeError("Length calculation failed")
        return self._length
    
    def __getitem__(self, idx):
        if self.error_type == "getitem_error":
            raise RuntimeError("Item access failed")
        return {"text": f"sample {idx}"}
    
    @property
    def cache_dir(self):
        if self.error_type == "cache_dir_error":
            raise AttributeError("cache_dir not accessible")
        return "/tmp/test_cache"
    
    def iter_batches(self, batch_size):
        """Iterate over batches of data."""
        if self.error_type == "iter_batches_error":
            raise RuntimeError("Batch iteration failed")
        
        # Simple batching implementation
        for i in range(0, self._length, batch_size):
            batch = []
            for j in range(i, min(i + batch_size, self._length)):
                batch.append(self[j])
            yield batch


def test_metadata_extraction_error_handling(tmp_path):
    """Test error handling in metadata extraction."""
    # Test with dataset that raises errors
    error_dataset = ErrorProneDataset("length_error")
    
    model = ErrorProneModel()
    tokenizer = ErrorProneTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "lin"
    
    store = LocalStore(tmp_path)
    
    # Should handle dataset length error gracefully
    lm.activations.infer_and_save(
        error_dataset,
        layer_signature=valid_layer,
        run_name="error_test",
        store=store,
        batch_size=2,
        verbose=True,
    )
    
    # Should still complete despite errors
    batches = store.list_run_batches("error_test")
    assert len(batches) > 0


def test_cache_dir_error_handling(tmp_path):
    """Test error handling when cache_dir is not accessible."""
    error_dataset = ErrorProneDataset("cache_dir_error")
    
    model = ErrorProneModel()
    tokenizer = ErrorProneTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "lin"
    
    store = LocalStore(tmp_path)
    
    # Should handle cache_dir error gracefully
    lm.activations.infer_and_save(
        error_dataset,
        layer_signature=valid_layer,
        run_name="cache_error_test",
        store=store,
        batch_size=2,
        verbose=True,
    )
    
    # Should still complete despite errors
    batches = store.list_run_batches("cache_error_test")
    assert len(batches) > 0


def test_model_name_extraction_error_handling(tmp_path):
    """Test error handling when model name extraction fails."""
    base = Dataset.from_dict({"text": ["sample 1", "sample 2", "sample 3"]})
    ds = TextSnippetDataset(base, cache_dir=tmp_path)
    
    # Create model without model_name attribute
    model = ErrorProneModel()
    tokenizer = ErrorProneTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    # Remove model_name attribute if it exists
    if hasattr(lm, 'model_name'):
        delattr(lm, 'model_name')
    
    store = LocalStore(tmp_path)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "lin"
    
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


def test_store_metadata_error_handling(tmp_path):
    """Test error handling when store metadata operations fail."""
    base = Dataset.from_dict({"text": ["sample 1", "sample 2", "sample 3"]})
    ds = TextSnippetDataset(base, cache_dir=tmp_path)
    
    model = ErrorProneModel()
    tokenizer = ErrorProneTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    # Mock store to raise error on put_run_meta
    store = LocalStore(tmp_path)
    original_put_meta = store.put_run_meta
    
    def error_put_meta(run_name, meta):
        raise RuntimeError("Metadata storage failed")
    
    store.put_run_meta = error_put_meta
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "lin"
    
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
    ds = TextSnippetDataset(base, cache_dir=tmp_path)
    
    model = ErrorProneModel()
    tokenizer = ErrorProneTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    store = LocalStore(tmp_path)
    
    # Test with different layer signatures
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "lin"
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


def test_device_handling_errors(tmp_path):
    """Test error handling in device operations."""
    base = Dataset.from_dict({"text": ["sample 1", "sample 2", "sample 3"]})
    ds = TextSnippetDataset(base, cache_dir=tmp_path)
    
    # Test with model that has device issues
    model = ErrorProneModel("device_error")
    tokenizer = ErrorProneTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    store = LocalStore(tmp_path)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "lin"
    
    # Should handle device errors gracefully
    with pytest.raises(Exception):  # This should raise due to device error
        lm.activations.infer_and_save(
            ds,
            layer_signature=valid_layer,
            run_name="device_error_test",
            store=store,
            batch_size=2,
            verbose=True,
        )


def test_tokenization_error_handling(tmp_path):
    """Test error handling in tokenization."""
    base = Dataset.from_dict({"text": ["sample 1", "sample 2", "sample 3"]})
    ds = TextSnippetDataset(base, cache_dir=tmp_path)
    
    model = ErrorProneModel()
    tokenizer = ErrorProneTokenizer("tokenization_error")
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    store = LocalStore(tmp_path)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "lin"
    
    # Should handle tokenization errors gracefully
    with pytest.raises(Exception):  # This should raise due to tokenization error
        lm.activations.infer_and_save(
            ds,
            layer_signature=valid_layer,
            run_name="tokenization_error_test",
            store=store,
            batch_size=2,
            verbose=True,
        )


def test_return_tensors_error_handling(tmp_path):
    """Test error handling with unsupported return_tensors."""
    base = Dataset.from_dict({"text": ["sample 1", "sample 2", "sample 3"]})
    ds = TextSnippetDataset(base, cache_dir=tmp_path)
    
    model = ErrorProneModel()
    tokenizer = ErrorProneTokenizer("return_tensors_error")
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    store = LocalStore(tmp_path)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "lin"
    
    # Should handle return_tensors errors gracefully
    with pytest.raises(Exception):  # This should raise due to return_tensors error
        lm.activations.infer_and_save(
            ds,
            layer_signature=valid_layer,
            run_name="return_tensors_error_test",
            store=store,
            batch_size=2,
            verbose=True,
        )


def test_verbose_logging_with_errors(tmp_path, caplog):
    """Test that verbose logging works even when errors occur."""
    import logging
    
    base = Dataset.from_dict({"text": ["sample 1", "sample 2", "sample 3"]})
    ds = TextSnippetDataset(base, cache_dir=tmp_path)
    
    model = ErrorProneModel()
    tokenizer = ErrorProneTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer)
    
    store = LocalStore(tmp_path)
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "lin"
    
    with caplog.at_level(logging.INFO):
        lm.activations.infer_and_save(
            ds,
            layer_signature=valid_layer,
            run_name="verbose_error_test",
            store=store,
            batch_size=2,
            verbose=True,
        )
    
    # Should have logged the start message
    log_messages = [rec.message for rec in caplog.records]
    assert any("Starting save_model_activations" in msg for msg in log_messages)
