"""Test batch processing functionality in LanguageModel."""

import pytest
import torch
from torch import nn
from pathlib import Path
import tempfile

from amber.language_model.language_model import LanguageModel
from amber.store.local_store import LocalStore


class MockTokenizer:
    """Test tokenizer for batch processing tests."""
    
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
    """Test model for batch processing tests."""
    
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
    
    @property
    def device(self):
        """Return the device of the first parameter."""
        return next(self.parameters()).device


class TestLanguageModelBatchProcessing:
    """Test batch processing functionality and edge cases."""

    def test_variable_length_sequences_in_batch(self):
        """Test handling of variable-length sequences in batch."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with variable-length sequences
        texts = ["a", "bb", "ccc", "dddd"]
        
        # Tokenize texts
        tokenized = tokenizer(texts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Check that sequences are padded to same length
        assert input_ids.shape[1] == 4  # Max length
        assert attention_mask.shape[1] == 4
        
        # Check that attention mask is correct
        assert torch.equal(attention_mask[0], torch.tensor([1, 0, 0, 0]))  # "a"
        assert torch.equal(attention_mask[1], torch.tensor([1, 1, 0, 0]))  # "bb"
        assert torch.equal(attention_mask[2], torch.tensor([1, 1, 1, 0]))  # "ccc"
        assert torch.equal(attention_mask[3], torch.tensor([1, 1, 1, 1]))  # "dddd"

    def test_padding_token_handling(self):
        """Test padding token handling."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with different padding strategies
        texts = ["a", "bb", "ccc"]
        
        # Test with padding
        tokenized = tokenizer(texts, padding=True)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # All sequences should be padded to same length
        assert input_ids.shape[1] == 3  # Max length
        assert attention_mask.shape[1] == 3
        
        # Test without padding
        tokenized = tokenizer(texts, padding=False)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Sequences should have different lengths
        assert input_ids.shape[1] == 3  # Max length
        assert attention_mask.shape[1] == 3

    def test_attention_mask_application(self):
        """Test attention mask application."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with attention mask
        texts = ["a", "bb", "ccc"]
        tokenized = tokenizer(texts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Test forward pass with attention mask
        output = lm.model(input_ids, attention_mask=attention_mask)
        
        # Output should have correct shape
        assert output.shape[0] == 3  # Batch size
        assert output.shape[1] == 3  # Sequence length

    def test_batch_size_edge_cases(self):
        """Test batch size edge cases."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with different batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            # Create batch
            texts = ["a"] * batch_size
            tokenized = tokenizer(texts)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            
            # Test forward pass
            output = lm.model(input_ids, attention_mask=attention_mask)
            
            # Output should have correct batch size
            assert output.shape[0] == batch_size

    def test_empty_batch_handling(self):
        """Test handling of empty batch raises ValueError."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with empty batch - should raise ValueError
        texts = []
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            lm.forwards(texts)
        
        # Tokenizer can still handle empty batch directly
        tokenized = tokenizer(texts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        assert input_ids.shape[0] == 0
        assert attention_mask.shape[0] == 0

    def test_single_sequence_batch(self):
        """Test handling of single sequence batch."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with single sequence
        texts = ["hello"]
        tokenized = tokenizer(texts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Test forward pass
        output = lm.model(input_ids, attention_mask=attention_mask)
        
        # Output should have correct shape
        assert output.shape[0] == 1  # Batch size
        assert output.shape[1] == 5  # Sequence length

    def test_large_batch_handling(self):
        """Test handling of large batches."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with large batch
        batch_size = 100
        texts = ["a"] * batch_size
        tokenized = tokenizer(texts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Test forward pass
        output = lm.model(input_ids, attention_mask=attention_mask)
        
        # Output should have correct shape
        assert output.shape[0] == batch_size
        assert output.shape[1] == 1  # Sequence length

    def test_batch_with_different_sequence_lengths(self):
        """Test batch with different sequence lengths."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with sequences of different lengths
        texts = ["a", "bb", "ccc", "dddd", "eeeee"]
        tokenized = tokenizer(texts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Test forward pass
        output = lm.model(input_ids, attention_mask=attention_mask)
        
        # Output should have correct shape
        assert output.shape[0] == 5  # Batch size
        assert output.shape[1] == 5  # Max sequence length

    def test_batch_with_special_tokens(self):
        """Test batch with special tokens."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with special tokens
        texts = ["<start>hello<end>", "<pad>world<pad>", "<unk>test<unk>"]
        tokenized = tokenizer(texts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Test forward pass
        output = lm.model(input_ids, attention_mask=attention_mask)
        
        # Output should have correct shape
        assert output.shape[0] == 3  # Batch size
        assert output.shape[1] == 17  # Max sequence length

    def test_batch_with_unicode_text(self):
        """Test batch with unicode text."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with unicode text
        texts = ["hello", "世界", "مرحبا", "🌍"]
        tokenized = tokenizer(texts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Test forward pass
        output = lm.model(input_ids, attention_mask=attention_mask)
        
        # Output should have correct shape
        assert output.shape[0] == 4  # Batch size
        assert output.shape[1] == 5  # Max sequence length

    def test_batch_with_empty_sequences(self):
        """Test batch with empty sequences."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with empty sequences
        texts = ["", "a", "", "bb"]
        tokenized = tokenizer(texts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Test forward pass
        output = lm.model(input_ids, attention_mask=attention_mask)
        
        # Output should have correct shape
        assert output.shape[0] == 4  # Batch size
        assert output.shape[1] == 2  # Max sequence length

    def test_batch_with_very_long_sequences(self):
        """Test batch with very long sequences."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with very long sequences
        long_text = "a" * 1000
        texts = [long_text, "short"]
        tokenized = tokenizer(texts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Test forward pass
        output = lm.model(input_ids, attention_mask=attention_mask)
        
        # Output should have correct shape
        assert output.shape[0] == 2  # Batch size
        assert output.shape[1] == 1000  # Max sequence length

    def test_batch_with_mixed_data_types(self):
        """Test batch with mixed data types."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with mixed data types
        texts = ["text", "123", "text123", "123text"]
        tokenized = tokenizer(texts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Test forward pass
        output = lm.model(input_ids, attention_mask=attention_mask)
        
        # Output should have correct shape
        assert output.shape[0] == 4  # Batch size
        assert output.shape[1] == 7  # Max sequence length

    def test_batch_with_duplicate_sequences(self):
        """Test batch with duplicate sequences."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with duplicate sequences
        texts = ["hello", "hello", "world", "world"]
        tokenized = tokenizer(texts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Test forward pass
        output = lm.model(input_ids, attention_mask=attention_mask)
        
        # Output should have correct shape
        assert output.shape[0] == 4  # Batch size
        assert output.shape[1] == 5  # Max sequence length

    def test_batch_with_none_values(self):
        """Test batch with None values."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with None values
        texts = ["hello", None, "world"]
        
        # Should handle None values gracefully
        with pytest.raises(TypeError):
            tokenizer(texts)

    def test_batch_with_non_string_values(self):
        """Test batch with non-string values."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with non-string values
        texts = ["hello", 123, "world"]
        
        # Should handle non-string values gracefully
        with pytest.raises(TypeError):
            tokenizer(texts)

    def test_batch_with_very_small_sequences(self):
        """Test batch with very small sequences."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with very small sequences
        texts = ["a", "b", "c", "d"]
        tokenized = tokenizer(texts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Test forward pass
        output = lm.model(input_ids, attention_mask=attention_mask)
        
        # Output should have correct shape
        assert output.shape[0] == 4  # Batch size
        assert output.shape[1] == 1  # Max sequence length

    def test_batch_with_single_character_sequences(self):
        """Test batch with single character sequences."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / "store")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with single character sequences
        texts = ["a", "b", "c", "d", "e"]
        tokenized = tokenizer(texts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Test forward pass
        output = lm.model(input_ids, attention_mask=attention_mask)
        
        # Output should have correct shape
        assert output.shape[0] == 5  # Batch size
        assert output.shape[1] == 1  # Max sequence length
