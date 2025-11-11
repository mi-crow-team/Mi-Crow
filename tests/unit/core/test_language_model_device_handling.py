from amber.store.local_store import LocalStore
from pathlib import Path
import tempfile
"""Test device handling functionality in LanguageModel."""

import pytest
import torch
from torch import nn
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
from amber.store.local_store import LocalStore

from amber.core.language_model import LanguageModel
import tempfile
from pathlib import Path
from amber.store.local_store import LocalStore


class MockTokenizer:
    """Test tokenizer for device handling tests."""
    
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
    """Test model for device handling tests."""
    
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


class TestLanguageModelDeviceHandling:
    """Test device handling functionality and edge cases."""

    def test_model_on_cpu_device(self):
        """Test model on CPU device."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / 'store')
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Model should be on CPU
        assert lm.model.device.type == "cpu"
        
        # Test forward pass
        input_ids = torch.tensor([[1, 2, 3]])
        output = lm.model(input_ids)
        assert output.device.type == "cpu"

    def test_model_on_cuda_device(self):
        """Test model on CUDA device if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / 'store')
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Move model to CUDA
        lm.context.model = lm.context.model.cuda()
        
        # Model should be on CUDA
        assert lm.model.device.type == "cuda"
        
        # Test forward pass
        input_ids = torch.tensor([[1, 2, 3]]).cuda()
        output = lm.model(input_ids)
        assert output.device.type == "cuda"

    def test_model_on_mps_device(self):
        """Test model on MPS device if available."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")
        
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / 'store')
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Move model to MPS
        lm.context.model = lm.context.model.to("mps")
        
        # Model should be on MPS
        assert lm.model.device.type == "mps"
        
        # Test forward pass
        input_ids = torch.tensor([[1, 2, 3]]).to("mps")
        output = lm.model(input_ids)
        assert output.device.type == "mps"

    def test_device_transfer_for_activations(self):
        """Test device transfer for activations."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / 'store')
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with different devices
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
        
        for device in devices:
            # Move model to device
            lm.context.model = lm.context.model.to(device)
            
            # Test activation extraction
            input_ids = torch.tensor([[1, 2, 3]]).to(device)
            output = lm.model(input_ids)
            
            # Output should be on correct device
            assert output.device.type == device

    def test_dtype_conversion(self):
        """Test dtype conversion functionality."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / 'store')
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with different dtypes
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        
        for dtype in dtypes:
            # Convert model to dtype
            lm.context.model = lm.context.model.to(dtype)
            
            # Test forward pass
            input_ids = torch.tensor([[1, 2, 3]])
            output = lm.model(input_ids)
            
            # Output should be on correct dtype
            assert output.dtype == dtype

    def test_mixed_device_tensor_handling(self):
        """Test handling of tensors on different devices."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / 'store')
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with model on CPU and input on different device
        if torch.cuda.is_available():
            # Model on CPU, input on CUDA
            lm.context.model = lm.context.model.cpu()
            input_ids = torch.tensor([[1, 2, 3]]).cuda()
            
            # Should handle device mismatch gracefully
            with pytest.raises(RuntimeError):
                lm.model(input_ids)

    def test_device_consistency_across_operations(self):
        """Test device consistency across operations."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / 'store')
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with different devices
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
        
        for device in devices:
            # Move model to device
            lm.context.model = lm.context.model.to(device)
            
            # Test multiple operations
            input_ids = torch.tensor([[1, 2, 3]]).to(device)
            
            # Forward pass
            output1 = lm.model(input_ids)
            assert output1.device.type == device
            
            # Another forward pass
            output2 = lm.model(input_ids)
            assert output2.device.type == device
            
            # Operations should be consistent
            assert output1.device == output2.device

    def test_device_handling_with_attention_mask(self):
        """Test device handling with attention mask."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / 'store')
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with different devices
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
        
        for device in devices:
            # Move model to device
            lm.context.model = lm.context.model.to(device)
            
            # Test with attention mask
            input_ids = torch.tensor([[1, 2, 3]]).to(device)
            attention_mask = torch.tensor([[1, 1, 1]]).to(device)
            
            output = lm.model(input_ids, attention_mask=attention_mask)
            
            # Output should be on correct device
            assert output.device.type == device

    def test_device_handling_with_different_batch_sizes(self):
        """Test device handling with different batch sizes."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / 'store')
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with different batch sizes
        batch_sizes = [1, 2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            # Test on CPU
            lm.context.model = lm.context.model.cpu()
            input_ids = torch.randint(0, 100, (batch_size, 3))
            output = lm.model(input_ids)
            
            # Output should be on CPU
            assert output.device.type == "cpu"
            assert output.shape[0] == batch_size

    def test_device_handling_with_different_sequence_lengths(self):
        """Test device handling with different sequence lengths."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / 'store')
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with different sequence lengths
        sequence_lengths = [1, 2, 4, 8, 16, 32]
        
        for seq_len in sequence_lengths:
            # Test on CPU
            lm.context.model = lm.context.model.cpu()
            input_ids = torch.randint(0, 100, (2, seq_len))
            output = lm.model(input_ids)
            
            # Output should be on CPU
            assert output.device.type == "cpu"
            assert output.shape[1] == seq_len

    def test_device_handling_with_gradient_computation(self):
        """Test device handling with gradient computation."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / 'store')
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with different devices
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
        
        for device in devices:
            # Move model to device
            lm.context.model = lm.context.model.to(device)
            
            # Test with gradient computation
            input_ids = torch.tensor([[1, 2, 3]]).to(device)
            output = lm.model(input_ids)
            
            # Compute loss and backward pass
            loss = output.sum()
            loss.backward()
            
            # Gradients should be on correct device
            for param in lm.model.parameters():
                if param.grad is not None:
                    assert param.grad.device.type == device

    def test_device_handling_with_model_parameters(self):
        """Test device handling with model parameters."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / 'store')
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with different devices
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
        
        for device in devices:
            # Move model to device
            lm.context.model = lm.context.model.to(device)
            
            # All parameters should be on correct device
            for param in lm.model.parameters():
                assert param.device.type == device
            
            # All buffers should be on correct device
            for buffer in lm.model.buffers():
                assert buffer.device.type == device

    def test_device_handling_with_model_state_dict(self):
        """Test device handling with model state dict."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / 'store')
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with different devices
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
        
        for device in devices:
            # Move model to device
            lm.context.model = lm.context.model.to(device)
            
            # Get state dict
            state_dict = lm.model.state_dict()
            
            # All tensors in state dict should be on correct device
            for key, tensor in state_dict.items():
                assert tensor.device.type == device

    def test_device_handling_with_model_eval_mode(self):
        """Test device handling with model in eval mode."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / 'store')
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with different devices
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
        
        for device in devices:
            # Move model to device
            lm.context.model = lm.context.model.to(device)
            
            # Set to eval mode
            lm.model.eval()
            
            # Test forward pass
            input_ids = torch.tensor([[1, 2, 3]]).to(device)
            output = lm.model(input_ids)
            
            # Output should be on correct device
            assert output.device.type == device
            
            # Model should be in eval mode
            assert not lm.model.training

    def test_device_handling_with_model_train_mode(self):
        """Test device handling with model in train mode."""
        model = MockModel()
        tokenizer = MockTokenizer()
        temp_dir = tempfile.mkdtemp()
        store = LocalStore(Path(temp_dir) / 'store')
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        # Test with different devices
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
        
        for device in devices:
            # Move model to device
            lm.context.model = lm.context.model.to(device)
            
            # Set to train mode
            lm.model.train()
            
            # Test forward pass
            input_ids = torch.tensor([[1, 2, 3]]).to(device)
            output = lm.model(input_ids)
            
            # Output should be on correct device
            assert output.device.type == device
            
            # Model should be in train mode
            assert lm.model.training
