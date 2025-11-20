"""Test split_to_tokens method in LanguageModelTokenizer."""

import torch
from torch import nn

<<<<<<< Updated upstream
=======
from amber.language_model.language_model import LanguageModel
>>>>>>> Stashed changes
import tempfile
from pathlib import Path
from amber.store.local_store import LocalStore


class TinyConfig:
    def __init__(self):
        self.pad_token_id = None


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 16, d_model: int = 4):
        super().__init__()
        self.config = TinyConfig()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        return self.embed(input_ids)


class TokenizeMethodTokenizer:
    """Tokenizer with tokenize() method (typical HuggingFace style)."""
    
    def __init__(self):
        self.vocab = {
            "hello": 1,
            "world": 2,
            "how": 3,
            "are": 4,
            "you": 5,
            "<bos>": 6,
            "<eos>": 7,
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def tokenize(self, text: str, add_special_tokens: bool = False):
        """Tokenize text into token strings."""
        # Simple whitespace tokenization for testing
        tokens = text.lower().split()
        if add_special_tokens:
            return ["<bos>"] + tokens + ["<eos>"]
        return tokens


class EncodeConvertTokenizer:
    """Tokenizer with encode() and convert_ids_to_tokens() methods."""
    
    def __init__(self):
        self.vocab = {
            "hello": 1,
            "world": 2,
            "how": 3,
            "are": 4,
            "you": 5,
            "<bos>": 6,
            "<eos>": 7,
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str, add_special_tokens: bool = False):
        """Encode text to token IDs."""
        tokens = text.lower().split()
        ids = [self.vocab.get(t, 0) for t in tokens]
        if add_special_tokens:
            return [self.vocab["<bos>"]] + ids + [self.vocab["<eos>"]]
        return ids
    
    def convert_ids_to_tokens(self, token_ids):
        """Convert token IDs to token strings."""
        return [self.id_to_token.get(tid, f"<unk_{tid}>") for tid in token_ids]


class EncodePlusConvertTokenizer:
    """Tokenizer with encode_plus() and convert_ids_to_tokens() methods."""
    
    def __init__(self):
        self.vocab = {
            "hello": 1,
            "world": 2,
            "how": 3,
            "are": 4,
            "you": 5,
            "<bos>": 6,
            "<eos>": 7,
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def encode_plus(self, text: str, add_special_tokens: bool = False):
        """Encode text to a dictionary with input_ids."""
        tokens = text.lower().split()
        ids = [self.vocab.get(t, 0) for t in tokens]
        if add_special_tokens:
            ids = [self.vocab["<bos>"]] + ids + [self.vocab["<eos>"]]
        return {"input_ids": ids}
    
    def convert_ids_to_tokens(self, token_ids):
        """Convert token IDs to token strings."""
        return [self.id_to_token.get(tid, f"<unk_{tid}>") for tid in token_ids]


class WhitespaceFallbackTokenizer:
    """Tokenizer that has none of the above methods - should fall back to whitespace split."""
    pass


class FailingTokenizer:
    """Tokenizer where all methods fail - should fall back to whitespace split."""
    
    def tokenize(self, text: str, add_special_tokens: bool = False):
        raise ValueError("Tokenization failed")
    
    def encode(self, text: str, add_special_tokens: bool = False):
        raise ValueError("Encoding failed")
    
    def encode_plus(self, text: str, add_special_tokens: bool = False):
        raise ValueError("Encode plus failed")
    
    def convert_ids_to_tokens(self, token_ids):
        raise ValueError("Convert failed")


def test_split_to_tokens_single_string_with_tokenize_method():
    """Test split_to_tokens with single string using tokenize() method."""
    model = TinyLM()
    tok = TokenizeMethodTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)  # type: ignore[arg-type]
    
    result = lm.lm_tokenizer.split_to_tokens("hello world")
    assert result == ["hello", "world"]
    assert isinstance(result, list)
    assert all(isinstance(t, str) for t in result)


def test_split_to_tokens_single_string_with_special_tokens():
    """Test split_to_tokens with special tokens enabled."""
    model = TinyLM()
    tok = TokenizeMethodTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)  # type: ignore[arg-type]
    
    result = lm.lm_tokenizer.split_to_tokens("hello world", add_special_tokens=True)
    assert result == ["<bos>", "hello", "world", "<eos>"]


def test_split_to_tokens_sequence_strings():
    """Test split_to_tokens with sequence of strings."""
    model = TinyLM()
    tok = TokenizeMethodTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)  # type: ignore[arg-type]
    
    result = lm.lm_tokenizer.split_to_tokens(["hello world", "how are you"])
    assert len(result) == 2
    assert result[0] == ["hello", "world"]
    assert result[1] == ["how", "are", "you"]
    assert isinstance(result, list)
    assert all(isinstance(t, list) for t in result)


def test_split_to_tokens_sequence_with_special_tokens():
    """Test split_to_tokens with sequence and special tokens."""
    model = TinyLM()
    tok = TokenizeMethodTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)  # type: ignore[arg-type]
    
    result = lm.lm_tokenizer.split_to_tokens(["hello", "world"], add_special_tokens=True)
    assert len(result) == 2
    assert result[0] == ["<bos>", "hello", "<eos>"]
    assert result[1] == ["<bos>", "world", "<eos>"]


def test_split_to_tokens_with_encode_convert_fallback():
    """Test split_to_tokens using encode() + convert_ids_to_tokens() fallback."""
    model = TinyLM()
    tok = EncodeConvertTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)  # type: ignore[arg-type]
    
    result = lm.lm_tokenizer.split_to_tokens("hello world")
    assert result == ["hello", "world"]


def test_split_to_tokens_with_encode_plus_convert_fallback():
    """Test split_to_tokens using encode_plus() + convert_ids_to_tokens() fallback."""
    model = TinyLM()
    tok = EncodePlusConvertTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)  # type: ignore[arg-type]
    
    result = lm.lm_tokenizer.split_to_tokens("hello world")
    assert result == ["hello", "world"]


def test_split_to_tokens_with_whitespace_fallback():
    """Test split_to_tokens falling back to whitespace split."""
    model = TinyLM()
    tok = WhitespaceFallbackTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)  # type: ignore[arg-type]
    
    result = lm.lm_tokenizer.split_to_tokens("hello world")
    assert result == ["hello", "world"]


def test_split_to_tokens_with_failing_tokenizer():
    """Test split_to_tokens when all tokenizer methods fail - should fall back to whitespace."""
    model = TinyLM()
    tok = FailingTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)  # type: ignore[arg-type]
    
    result = lm.lm_tokenizer.split_to_tokens("hello world")
    assert result == ["hello", "world"]


def test_split_to_tokens_empty_string():
    """Test split_to_tokens with empty string."""
    model = TinyLM()
    tok = TokenizeMethodTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)  # type: ignore[arg-type]
    
    result = lm.lm_tokenizer.split_to_tokens("")
    assert result == []


def test_split_to_tokens_empty_strings_sequence():
    """Test split_to_tokens with sequence containing empty strings."""
    model = TinyLM()
    tok = TokenizeMethodTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)  # type: ignore[arg-type]
    
    result = lm.lm_tokenizer.split_to_tokens(["", "hello"])
    assert len(result) == 2
    assert result[0] == []
    assert result[1] == ["hello"]


def test_split_to_tokens_single_word():
    """Test split_to_tokens with single word."""
    model = TinyLM()
    tok = TokenizeMethodTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)  # type: ignore[arg-type]
    
    result = lm.lm_tokenizer.split_to_tokens("hello")
    assert result == ["hello"]


def test_split_to_tokens_whitespace_only():
    """Test split_to_tokens with whitespace-only string."""
    model = TinyLM()
    tok = WhitespaceFallbackTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)  # type: ignore[arg-type]
    
    result = lm.lm_tokenizer.split_to_tokens("   ")
    # Should return empty list after split and filter
    assert result == []


def test_split_to_tokens_encode_convert_with_special_tokens():
    """Test encode+convert fallback with special tokens."""
    model = TinyLM()
    tok = EncodeConvertTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)  # type: ignore[arg-type]
    
    result = lm.lm_tokenizer.split_to_tokens("hello world", add_special_tokens=True)
    assert result == ["<bos>", "hello", "world", "<eos>"]


def test_split_to_tokens_encode_plus_convert_with_special_tokens():
    """Test encode_plus+convert fallback with special tokens."""
    model = TinyLM()
    tok = EncodePlusConvertTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)  # type: ignore[arg-type]
    
    result = lm.lm_tokenizer.split_to_tokens("hello world", add_special_tokens=True)
    assert result == ["<bos>", "hello", "world", "<eos>"]


def test_split_to_tokens_sequence_mixed_lengths():
    """Test split_to_tokens with sequence of strings of different lengths."""
    model = TinyLM()
    tok = TokenizeMethodTokenizer()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=tok, store=store)  # type: ignore[arg-type]
    
    result = lm.lm_tokenizer.split_to_tokens(["hello", "hello world", "how are you"])
    assert len(result) == 3
    assert result[0] == ["hello"]
    assert result[1] == ["hello", "world"]
    assert result[2] == ["how", "are", "you"]


def test_split_to_tokens_with_none_tokenizer():
    """Test split_to_tokens when tokenizer from LanguageModelContext is None."""
    model = TinyLM()
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / 'store')
    lm = LanguageModel(model=model, tokenizer=None, store=store)  # type: ignore[arg-type]
    
    # Should fall back to whitespace split
    result = lm.lm_tokenizer.split_to_tokens("hello world")
    assert result == ["hello", "world"]
    
    result = lm.lm_tokenizer.split_to_tokens(["hello world", "how are you"])
    assert len(result) == 2
    assert result[0] == ["hello", "world"]
    assert result[1] == ["how", "are", "you"]

