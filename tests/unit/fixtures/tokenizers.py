"""Mock tokenizers for testing."""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Union
from transformers import PreTrainedTokenizerBase


class MockTokenizer(PreTrainedTokenizerBase):
    """Mock tokenizer for testing."""

    _SPECIAL_TOKEN_NAMES = {"pad_token", "eos_token", "bos_token", "unk_token"}
    _SPECIAL_ID_NAMES = {"pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"}

    def __setattr__(self, key, value):
        if key in self._SPECIAL_TOKEN_NAMES:
            object.__setattr__(self, f"_{key}", value)
            if hasattr(self, "_special_tokens_map"):
                self._special_tokens_map[key] = value
            return
        if key in self._SPECIAL_ID_NAMES:
            object.__setattr__(self, key, value)
            return
        if key == "all_special_ids":
            object.__setattr__(self, "_all_special_ids", value)
            return
        super().__setattr__(key, value)

    def __init__(
        self,
        vocab_size: int = 1000,
        pad_token: str = "<pad>",
        eos_token: str = "<eos>",
        bos_token: str = "<bos>",
        unk_token: str = "<unk>",
    ):
        # Initialize parent without calling super().__init__() to avoid token validation
        # We'll set attributes directly
        self.vocab_size = vocab_size
        
        # Create simple vocabulary
        self._vocab = {
            pad_token: 0,
            unk_token: 1,
            eos_token: 2,
            bos_token: 3,
        }
        # Add some common words
        for i, word in enumerate(["the", "a", "an", "is", "are", "was", "were"], start=4):
            self._vocab[word] = i
        
        # Reverse vocab for decoding
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
        
        # Set special tokens as strings first (required by PreTrainedTokenizerBase)
        object.__setattr__(self, 'pad_token', pad_token)
        object.__setattr__(self, 'eos_token', eos_token)
        object.__setattr__(self, 'bos_token', bos_token)
        object.__setattr__(self, 'unk_token', unk_token)
        # Also set private versions some HF utilities expect
        object.__setattr__(self, '_pad_token', pad_token)
        object.__setattr__(self, '_eos_token', eos_token)
        object.__setattr__(self, '_bos_token', bos_token)
        object.__setattr__(self, '_unk_token', unk_token)
        
        # Add special token IDs
        object.__setattr__(self, 'pad_token_id', 0)
        object.__setattr__(self, 'eos_token_id', 2)
        object.__setattr__(self, 'bos_token_id', 3)
        object.__setattr__(self, 'unk_token_id', 1)
        
        # Set special tokens map for compatibility
        object.__setattr__(self, '_special_tokens_map', {
            'pad_token': pad_token,
            'eos_token': eos_token,
            'bos_token': bos_token,
            'unk_token': unk_token,
        })
        
        # Add other required attributes for transformers compatibility
        object.__setattr__(self, 'split_special_tokens', False)
        object.__setattr__(self, 'verbose', False)
        object.__setattr__(self, '_in_target_context_manager', False)
        
        # Define special token property helpers so assignments behave predictably
        def _make_token_property(attr_name):
            private_name = f"_{attr_name}"

            def getter(self):
                return getattr(self, private_name, None)

            def setter(self, value):
                object.__setattr__(self, private_name, value)
                if hasattr(self, "_special_tokens_map") and attr_name in self._special_tokens_map:
                    self._special_tokens_map[attr_name] = value

            return property(getter, setter)

        type(self).pad_token = _make_token_property("pad_token")
        type(self).eos_token = _make_token_property("eos_token")
        type(self).bos_token = _make_token_property("bos_token")
        type(self).unk_token = _make_token_property("unk_token")

        # Add __call__ method for direct tokenizer calls
        def __call__(self, text, **kwargs):
            return self.encode(text, **kwargs)
        object.__setattr__(self, '__call__', __call__)
        
        # Override all_special_tokens property
        @property
        def all_special_tokens(self):
            return [pad_token, eos_token, bos_token, unk_token]
        
        # Use type to add property to instance
        import types
        prop = property(lambda self: [pad_token, eos_token, bos_token, unk_token])
        type(self).all_special_tokens = prop
        
        # Initialize all_special_ids with default values
        object.__setattr__(self, '_all_special_ids', [0, 1, 2, 3])
        
        # Add property for all_special_ids
        def _all_special_ids_getter(self):
            return getattr(self, '_all_special_ids', [0, 1, 2, 3])
        
        def _all_special_ids_setter(self, value):
            object.__setattr__(self, '_all_special_ids', value)
        
        type(self).all_special_ids = property(_all_special_ids_getter, _all_special_ids_setter)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on spaces."""
        return text.lower().split()
    
    def tokenize(self, text: str, add_special_tokens: bool = False, **kwargs) -> List[str]:
        """Tokenize text (for compatibility with transformers)."""
        tokens = self._tokenize(text)
        if add_special_tokens:
            tokens = [self._bos_token] + tokens + [self._eos_token]
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        return self._vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token."""
        return self._ids_to_tokens.get(index, self._unk_token)
    
    def convert_ids_to_tokens(self, ids, skip_special_tokens: bool = False):
        """Convert token IDs to tokens (for compatibility with transformers)."""
        single_input = False
        if isinstance(ids, int):
            ids = [ids]
            single_input = True
        tokens = [self._convert_id_to_token(id) for id in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in [self._pad_token, self._eos_token, self._bos_token, self._unk_token]]
        if single_input:
            return tokens[0] if tokens else None
        return tokens
    
    def convert_tokens_to_ids(self, tokens):
        """Convert tokens to IDs (for compatibility with transformers)."""
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self._convert_token_to_id(t) for t in tokens]

    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary."""
        return self._vocab.copy()

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self._vocab)

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Union[List[int], List[List[int]], Any]:
        """Encode text to token IDs."""
        if isinstance(text, str):
            tokens = self._tokenize(text)
            ids = [self._convert_token_to_id(t) for t in tokens]
            if add_special_tokens:
                ids = [self.bos_token_id] + ids + [self.eos_token_id]
            
            if return_tensors == "pt":
                import torch
                return torch.tensor([ids])
            return ids
        else:
            # Batch encoding
            results = []
            for t in text:
                tokens = self._tokenize(t)
                ids = [self._convert_token_to_id(t) for t in tokens]
                if add_special_tokens:
                    ids = [self.bos_token_id] + ids + [self.eos_token_id]
                results.append(ids)
            
            if return_tensors == "pt":
                import torch
                # Pad to same length
                max_len = max(len(r) for r in results)
                padded = []
                for r in results:
                    padded.append(r + [self.pad_token_id] * (max_len - len(r)))
                return torch.tensor(padded)
            return results

    def decode(
        self,
        token_ids: Union[int, List[int], Any],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """Decode token IDs to text."""
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        
        tokens = [self._convert_id_to_token(id) for id in token_ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in [self._pad_token, self._eos_token, self._bos_token, self._unk_token]]
        return " ".join(tokens)

    def batch_encode_plus(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
        padding: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Batch encode texts."""
        if isinstance(text, str):
            text = [text]
        
        input_ids = []
        attention_mask = []
        
        for t in text:
            tokens = self._tokenize(t)
            ids = [self._convert_token_to_id(t) for t in tokens]
            if add_special_tokens:
                ids = [self.bos_token_id] + ids + [self.eos_token_id]
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
        
        if padding:
            max_len = max(len(ids) for ids in input_ids)
            for i in range(len(input_ids)):
                pad_len = max_len - len(input_ids[i])
                input_ids[i].extend([self.pad_token_id] * pad_len)
                attention_mask[i].extend([0] * pad_len)
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        if return_tensors == "pt":
            import torch
            result["input_ids"] = torch.tensor(result["input_ids"])
            result["attention_mask"] = torch.tensor(result["attention_mask"])
        
        return result


def create_mock_tokenizer(
    vocab_size: int = 1000,
    pad_token: str = "<pad>",
    eos_token: str = "<eos>",
    bos_token: str = "<bos>",
    unk_token: str = "<unk>",
) -> MockTokenizer:
    """
    Create a mock tokenizer for testing.
    
    Args:
        vocab_size: Vocabulary size
        pad_token: Padding token
        eos_token: End of sequence token
        bos_token: Beginning of sequence token
        unk_token: Unknown token
        
    Returns:
        Mock tokenizer instance
    """
    return MockTokenizer(
        vocab_size=vocab_size,
        pad_token=pad_token,
        eos_token=eos_token,
        bos_token=bos_token,
        unk_token=unk_token,
    )

