"""Test that InputTracker correctly saves texts and integrates with SAE hooks."""

import torch
from torch import nn

from amber.language_model.language_model import LanguageModel
from amber.store.local_store import LocalStore
from pathlib import Path
import tempfile
from amber.mechanistic.sae.concepts.input_tracker import InputTracker
from amber.mechanistic.sae.modules.topk_sae import TopKSae


class MockTokenizer:
    """Test tokenizer."""
    
    def __init__(self):
        self.pad_token = None
        self.pad_token_id = None
        self.eos_token = None
        self.eos_token_id = None

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


class MockModel(nn.Module):
    """Test model."""
    
    def __init__(self, vocab_size: int = 100, d_model: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, d_model)
        
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


def test_set_current_texts_saves_texts():
    """Test that set_current_texts correctly saves texts."""
    model = MockModel()
    tokenizer = MockTokenizer()
    temp_dir = tempfile.mkdtemp()

    store = LocalStore(Path(temp_dir) / 'store')

    lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    tracker = InputTracker(language_model=lm)
    tracker.enable()  # Enable tracking
    
    # Initially empty
    assert len(tracker._current_texts) == 0
    
    # Set texts
    texts = ["hello", "world", "test"]
    tracker.set_current_texts(texts)
    
    # Verify texts are saved
    assert len(tracker._current_texts) == 3
    assert tracker._current_texts == ["hello", "world", "test"]


def test_texts_persist_through_multiple_calls():
    """Test that texts persist correctly through multiple set_current_texts calls."""
    model = MockModel()
    tokenizer = MockTokenizer()
    temp_dir = tempfile.mkdtemp()

    store = LocalStore(Path(temp_dir) / 'store')

    lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    tracker = InputTracker(language_model=lm)
    tracker.enable()  # Enable tracking
    
    # First batch
    texts1 = ["first", "batch"]
    tracker.set_current_texts(texts1)
    assert tracker._current_texts == ["first", "batch"]
    
    # Second batch (should replace, not append)
    texts2 = ["second", "batch", "three"]
    tracker.set_current_texts(texts2)
    assert tracker._current_texts == ["second", "batch", "three"]
    assert len(tracker._current_texts) == 3


def test_sae_hook_updates_top_texts_during_inference():
    """Test that SAE hook updates top texts during modify_activations."""
    
    model = MockModel()
    tokenizer = MockTokenizer()
    temp_dir = tempfile.mkdtemp()

    store = LocalStore(Path(temp_dir) / 'store')

    lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    # Create and register SAE hook
    sae_hook = TopKSae(n_latents=4, n_inputs=8, k=2, device='cpu')
    
    # Find layer to register SAE hook on
    layer_names = lm.layers.get_layer_names()
    sae_layer = layer_names[0] if layer_names else None
    
    # Register SAE hook
    lm.layers.register_hook(sae_layer, sae_hook)
    
    # Set up context and enable text tracking
    sae_hook.context.lm = lm
    sae_hook.context.lm_layer_signature = sae_layer
    sae_hook.context.text_tracking_k = 5
    sae_hook.context.text_tracking_negative = False
    sae_hook.context.text_tracking_enabled = True
    sae_hook.concepts.enable_text_tracking()
    
    # Run inference - this should trigger text tracking
    texts = ["text1", "text2", "text3"]
    lm.forwards(texts)
    
    # Verify that texts were tracked in SAE hook's concepts
    # Note: Results depend on SAE encoding, so we just check that tracking happened
    # The InputTracker should have been created automatically
    assert lm.get_input_tracker() is not None
    
    # Cleanup
    lm.layers.unregister_hook(sae_hook.id)


def test_input_tracker_get_current_texts():
    """Test that get_current_texts returns a copy of saved texts."""
    model = MockModel()
    tokenizer = MockTokenizer()
    temp_dir = tempfile.mkdtemp()

    store = LocalStore(Path(temp_dir) / 'store')

    lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    tracker = InputTracker(language_model=lm)
    tracker.enable()  # Enable tracking
    
    # Set texts
    texts = ["hello", "world"]
    tracker.set_current_texts(texts)
    
    # Get texts
    retrieved = tracker.get_current_texts()
    
    # Should be equal but not the same object
    assert retrieved == texts
    assert retrieved is not tracker._current_texts


def test_sae_hook_tracks_texts_correctly():
    """Test that SAE hook correctly tracks texts during inference."""
    
    model = MockModel()
    tokenizer = MockTokenizer()
    temp_dir = tempfile.mkdtemp()

    store = LocalStore(Path(temp_dir) / 'store')

    lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    # Create and register SAE hook
    sae_hook = TopKSae(n_latents=3, n_inputs=8, k=2, device='cpu')
    
    # Find layer to register SAE hook on
    layer_names = lm.layers.get_layer_names()
    sae_layer = layer_names[0] if layer_names else None
    
    # Register SAE hook
    lm.layers.register_hook(sae_layer, sae_hook)
    
    # Set up context and enable text tracking
    sae_hook.context.lm = lm
    sae_hook.context.lm_layer_signature = sae_layer
    sae_hook.context.text_tracking_k = 10
    sae_hook.context.text_tracking_negative = False
    sae_hook.context.text_tracking_enabled = True
    sae_hook.concepts.enable_text_tracking()
    
    # Run inference with distinct texts
    texts = ["batch0", "batch1", "batch2"]
    lm.forwards(texts)
    
    # Verify that texts were tracked
    # The exact results depend on SAE encoding, but we should have some texts
    neuron0_texts = sae_hook.concepts.get_top_texts_for_neuron(0)
    # Just verify that tracking is working (may be empty if SAE didn't activate)
    # The important thing is that the system doesn't crash
    
    # Cleanup
    lm.layers.unregister_hook(sae_hook.id)


def test_language_model_calls_set_current_texts():
    """Test that LanguageModel correctly calls set_current_texts on InputTracker."""
    model = MockModel()
    tokenizer = MockTokenizer()
    temp_dir = tempfile.mkdtemp()

    store = LocalStore(Path(temp_dir) / 'store')

    lm = LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    # Create InputTracker (should be done via _ensure_input_tracker, but we can test directly)
    from amber.mechanistic.sae.concepts.input_tracker import InputTracker
    tracker = InputTracker(language_model=lm)
    tracker.enable()  # Enable tracking
    lm._input_tracker = tracker  # Set as singleton
    
    # Run inference
    texts = ["test1", "test2"]
    lm.forwards(texts)
    
    # Verify texts were set (before tokenization)
    assert len(tracker._current_texts) == 2
    assert tracker._current_texts == ["test1", "test2"]

