import pytest
import torch
from torch import nn
from datasets import Dataset

from amber.language_model.language_model import LanguageModel
from amber.store.local_store import LocalStore


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
    def dataset_dir(self):
        if self.error_type == "dataset_dir_error":
            raise AttributeError("dataset_dir not accessible")
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
    lm = LanguageModel(model=model, tokenizer=tokenizer, store=LocalStore(tmp_path / "store"))
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "lin"
    
    # Should handle dataset length error gracefully
    lm.activations.save_activations_dataset(
        error_dataset,
        layer_signature=valid_layer,
        run_name="error_test",
        batch_size=2,
        verbose=True,
    )
    
    # Should still complete despite errors
    batches = lm.store.list_run_batches("error_test")
    assert len(batches) > 0


def test_dataset_dir_error_handling(tmp_path):
    """Test error handling when dataset_dir is not accessible."""
    error_dataset = ErrorProneDataset("dataset_dir_error")
    
    model = ErrorProneModel()
    tokenizer = ErrorProneTokenizer()
    lm = LanguageModel(model=model, tokenizer=tokenizer, store=LocalStore(tmp_path / "store"))
    
    # Find a valid layer name
    layer_names = lm.layers.get_layer_names()
    valid_layer = layer_names[0] if layer_names else "lin"
    
    # Should handle dataset_dir error gracefully
    lm.activations.save_activations_dataset(
        error_dataset,
        layer_signature=valid_layer,
        run_name="cache_error_test",
        batch_size=2,
        verbose=True,
    )
    
    # Should still complete despite errors
    batches = lm.store.list_run_batches("cache_error_test")
    assert len(batches) > 0


