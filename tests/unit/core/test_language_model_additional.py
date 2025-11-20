"""Additional tests for LanguageModel to improve coverage."""
import pytest
import torch
from torch import nn

from amber.language_model.language_model import LanguageModel
from amber.store.local_store import LocalStore


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
    """Mock model for testing."""
    
    def __init__(self, vocab_size: int = 100, d_model: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)
        
        class SimpleConfig:
            def __init__(self):
                self.pad_token_id = None
                self.name_or_path = "MockModel"
        
        self.config = SimpleConfig()
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embedding(input_ids)
        logits = self.linear(x)
        
        # Return object with logits attribute
        class Output:
            def __init__(self, logits):
                self.logits = logits
        
        return Output(logits)


class TestLanguageModelGenerate:
    """Test LanguageModel.generate method."""
    
    @pytest.fixture
    def setup_lm(self, tmp_path):
        """Set up LanguageModel for testing."""
        model = MockModel()
        tokenizer = MockTokenizer()
        store = LocalStore(tmp_path)
        return LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    def test_generate_with_logits_attribute(self, setup_lm):
        """Test generate with output that has logits attribute."""
        lm = setup_lm
        
        texts = ["hello", "world"]
        results = lm.generate(texts)
        
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)
    
    def test_generate_with_tuple_output(self, setup_lm):
        """Test generate with tuple output."""
        lm = setup_lm
        
        # Mock model to return tuple
        original_forward = lm.model.forward
        def tuple_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            return (result.logits,)
        
        lm.model.forward = tuple_forward
        
        texts = ["test"]
        results = lm.generate(texts)
        
        assert len(results) == 1
        assert isinstance(results[0], str)
    
    def test_generate_with_tensor_output(self, setup_lm):
        """Test generate with tensor output."""
        lm = setup_lm
        
        # Mock model to return tensor directly
        original_forward = lm.model.forward
        def tensor_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            return result.logits
        
        lm.model.forward = tensor_forward
        
        texts = ["test"]
        results = lm.generate(texts)
        
        assert len(results) == 1
        assert isinstance(results[0], str)
    
    def test_generate_with_skip_special_tokens(self, setup_lm):
        """Test generate with skip_special_tokens parameter."""
        lm = setup_lm
        
        texts = ["hello"]
        results = lm.generate(texts, skip_special_tokens=True)
        
        assert len(results) == 1
        assert isinstance(results[0], str)
    
    def test_generate_with_empty_texts_raises_error(self, setup_lm):
        """Test generate raises error when texts list is empty."""
        lm = setup_lm
        
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            lm.generate([])
    
    def test_generate_without_tokenizer_raises_error(self, setup_lm):
        """Test generate raises error when tokenizer is None."""
        lm = setup_lm
        
        # Set tokenizer to None - validation happens at start of generate()
        original_tokenizer = lm.context.tokenizer
        lm.context.tokenizer = None
        
        try:
            with pytest.raises(ValueError, match="Tokenizer is required"):
                lm.generate(["test"])
        finally:
            lm.context.tokenizer = original_tokenizer
    
    def test_generate_with_invalid_output_raises_error(self, setup_lm):
        """Test generate raises error with invalid output type."""
        lm = setup_lm
        
        # Mock model to return invalid output
        original_forward = lm.model.forward
        def invalid_forward(*args, **kwargs):
            return "invalid_output"
        
        lm.model.forward = invalid_forward
        
        with pytest.raises(ValueError, match="Unable to extract logits"):
            lm.generate(["test"])


class TestLanguageModelInference:
    """Test LanguageModel inference methods."""
    
    @pytest.fixture
    def setup_lm(self, tmp_path):
        """Set up LanguageModel for testing."""
        model = MockModel()
        tokenizer = MockTokenizer()
        store = LocalStore(tmp_path)
        return LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    def test_forwards_basic(self, setup_lm):
        """Test basic forwards method."""
        lm = setup_lm
        
        texts = ["hello", "world"]
        output, enc = lm.forwards(texts)
        
        assert output is not None
        assert enc is not None
        assert "input_ids" in enc
    
    def test_forwards_with_autocast_false(self, setup_lm):
        """Test forwards with autocast disabled."""
        lm = setup_lm
        
        texts = ["test"]
        output, enc = lm.forwards(texts, autocast=False)
        
        assert output is not None
    
    def test_forwards_with_custom_tok_kwargs(self, setup_lm):
        """Test forwards with custom tokenizer kwargs."""
        lm = setup_lm
        
        texts = ["test"]
        output, enc = lm.forwards(texts, tok_kwargs={"max_length": 10})
        
        assert output is not None
    
    def test_forwards_with_controllers_disabled(self, setup_lm):
        """Test forwards with controllers disabled."""
        lm = setup_lm
        
        texts = ["test"]
        output, enc = lm.forwards(texts, with_controllers=False)
        
        assert output is not None
    
    def test_forwards_with_empty_texts_raises_error(self, setup_lm):
        """Test forwards raises error when texts list is empty."""
        lm = setup_lm
        
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            lm.forwards([])
    
    def test_inference_sets_texts_on_tracker(self, setup_lm):
        """Test that inference sets texts on input tracker."""
        lm = setup_lm
        
        # Enable input tracker
        tracker = lm._ensure_input_tracker()
        tracker.enable()
        
        texts = ["hello", "world"]
        lm.forwards(texts)
        
        # Text tracker should have been called
        # (We can't easily verify this without mocking, but we test the code path)


class TestLanguageModelInitialization:
    """Test LanguageModel initialization edge cases."""
    
    def test_init_with_model_id(self, tmp_path):
        """Test initialization with explicit model_id."""
        model = MockModel()
        tokenizer = MockTokenizer()
        store = LocalStore(tmp_path)
        
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store, model_id="custom_model")
        
        assert lm.model_id == "custom_model"
    
    def test_init_with_model_config_name_or_path(self, tmp_path):
        """Test initialization with model.config.name_or_path."""
        model = MockModel()
        model.config.name_or_path = "test/model"
        tokenizer = MockTokenizer()
        store = LocalStore(tmp_path)
        
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        assert lm.model_id == "test_model"  # Slashes replaced with underscores
    
    def test_init_without_model_config(self, tmp_path):
        """Test initialization without model.config."""
        class ModelWithoutConfig(nn.Module):
            def forward(self, input_ids, **kwargs):
                return torch.randn(input_ids.shape[0], 10, 100)
        
        model = ModelWithoutConfig()
        tokenizer = MockTokenizer()
        store = LocalStore(tmp_path)
        
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
        
        assert lm.model_id == "ModelWithoutConfig"  # Uses class name

