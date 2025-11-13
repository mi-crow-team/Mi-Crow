"""Tests for new activation saving methods."""
import pytest
import torch
from pathlib import Path
from datasets import Dataset

from amber.core.language_model import LanguageModel
from amber.store.local_store import LocalStore
from amber.adapters.text_snippet_dataset import TextSnippetDataset


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


class TestSaveActivationsPerSequence:
    """Test save_activations_per_sequence method."""
    
    @pytest.fixture
    def setup_lm(self, tmp_path):
        """Set up LanguageModel for testing."""
        model = MockModel()
        tokenizer = MockTokenizer()
        store = LocalStore(tmp_path)
        return LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    def test_save_activations_per_sequence_basic(self, setup_lm):
        """Test basic save_activations_per_sequence functionality."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        assert len(layer_names) > 0
        layer_name = layer_names[0]
        
        texts = ["hello", "world", "test"]
        run_name = lm.activations.save_activations_per_sequence(
            texts=texts,
            layer_signature=layer_name,
            run_name="test_seq",
            verbose=False
        )
        
        assert run_name == "test_seq"
        batches = lm.store.list_run_batches("test_seq")
        assert len(batches) == 3  # One batch per sequence
        assert batches == [0, 1, 2]
    
    def test_save_activations_per_sequence_auto_run_name(self, setup_lm):
        """Test save_activations_per_sequence with auto-generated run name."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        texts = ["a", "b"]
        run_name = lm.activations.save_activations_per_sequence(
            texts=texts,
            layer_signature=layer_name,
            verbose=False
        )
        
        assert run_name.startswith("run_")
        batches = lm.store.list_run_batches(run_name)
        assert len(batches) == 2
    
    def test_save_activations_per_sequence_with_options(self, setup_lm):
        """Test save_activations_per_sequence with various options."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        texts = ["test"]
        run_name = lm.activations.save_activations_per_sequence(
            texts=texts,
            layer_signature=layer_name,
            run_name="test_opts",
            dtype=torch.float32,
            max_length=10,
            save_inputs=False,
            verbose=True
        )
        
        batches = lm.store.list_run_batches(run_name)
        assert len(batches) == 1
        batch = lm.store.get_run_batch(run_name, 0)
        assert "activations" in batch
        # When save_inputs=False, inputs may not be in payload
        # but activations should be there


class TestSaveActivationsPerSequenceAllLayers:
    """Test save_activations_per_sequence_all_layers method."""
    
    @pytest.fixture
    def setup_lm(self, tmp_path):
        """Set up LanguageModel for testing."""
        model = MockModel()
        tokenizer = MockTokenizer()
        store = LocalStore(tmp_path)
        return LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    def test_save_activations_per_sequence_all_layers_basic(self, setup_lm):
        """Test basic save_activations_per_sequence_all_layers functionality."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        test_layers = layer_names[:2] if len(layer_names) >= 2 else layer_names
        
        texts = ["hello", "world"]
        run_name = lm.activations.save_activations_per_sequence_all_layers(
            texts=texts,
            layer_signatures=test_layers,
            run_name="test_all_seq",
            verbose=False
        )
        
        assert run_name == "test_all_seq"
        batches = lm.store.list_run_batches("test_all_seq")
        assert len(batches) == 2  # One batch per sequence
        
        # Check that activations from multiple layers are saved
        batch = lm.store.get_run_batch("test_all_seq", 0)
        activation_keys = [k for k in batch.keys() if k.startswith("activations_")]
        assert len(activation_keys) == len(test_layers)
    
    def test_save_activations_per_sequence_all_layers_none_signatures(self, setup_lm):
        """Test save_activations_per_sequence_all_layers with None layer_signatures."""
        lm = setup_lm
        
        texts = ["test"]
        run_name = lm.activations.save_activations_per_sequence_all_layers(
            texts=texts,
            layer_signatures=None,  # Should use all layers
            run_name="test_all_none",
            verbose=False
        )
        
        batches = lm.store.list_run_batches("test_all_none")
        assert len(batches) == 1


class TestSaveActivationsDatasetAllLayers:
    """Test save_activations_dataset_all_layers method."""
    
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
        texts = ["hello world", "test text", "another example"]
        cache_dir = tmp_path / "cache"
        ds = Dataset.from_dict({"text": texts})
        dataset = TextSnippetDataset(ds, cache_dir=cache_dir)
        return dataset
    
    def test_save_activations_dataset_all_layers_basic(self, setup_lm, setup_dataset):
        """Test basic save_activations_dataset_all_layers functionality."""
        lm = setup_lm
        dataset = setup_dataset
        
        layer_names = lm.layers.get_layer_names()
        test_layers = layer_names[:2] if len(layer_names) >= 2 else layer_names
        
        run_name = lm.activations.save_activations_dataset_all_layers(
            dataset=dataset,
            layer_signatures=test_layers,
            run_name="test_all_dataset",
            batch_size=2,
            verbose=False
        )
        
        assert run_name == "test_all_dataset"
        batches = lm.store.list_run_batches("test_all_dataset")
        assert len(batches) > 0


class TestHelperMethods:
    """Test private helper methods."""
    
    @pytest.fixture
    def setup_lm(self, tmp_path):
        """Set up LanguageModel for testing."""
        model = MockModel()
        tokenizer = MockTokenizer()
        store = LocalStore(tmp_path)
        return LanguageModel(model=model, tokenizer=tokenizer, store=store)
    
    def test_prepare_run_metadata_with_none_layer_signatures(self, setup_lm):
        """Test _prepare_run_metadata with None layer_signatures."""
        lm = setup_lm
        
        run_name, meta = lm.activations._prepare_run_metadata(
            layer_signatures=None,
            dataset=None,
            run_name="test",
            options={"key": "value"}
        )
        
        assert run_name == "test"
        assert "layer_signatures" not in meta or meta.get("layer_signatures") == []
    
    def test_prepare_run_metadata_with_dataset_error(self, setup_lm, tmp_path):
        """Test _prepare_run_metadata handles dataset errors gracefully."""
        lm = setup_lm
        
        # Create a dataset-like object that will raise error
        class BadDataset:
            def __len__(self):
                raise RuntimeError("Cannot get length")
        
        bad_ds = BadDataset()
        run_name, meta = lm.activations._prepare_run_metadata(
            layer_signatures="layer0",
            dataset=bad_ds,  # type: ignore
            run_name="test",
            options={}
        )
        
        assert run_name == "test"
        assert meta.get("dataset", {}).get("length") == -1
    
    def test_process_activation_tensor_with_reshape(self, setup_lm):
        """Test _process_activation_tensor with reshape."""
        lm = setup_lm
        
        # Create 2D activation [B*T, D] and 2D input_ids [B, T]
        act = torch.ones(6, 8)  # 2*3, 8
        inp_ids = torch.ones(2, 3, dtype=torch.long)  # B=2, T=3
        
        processed = lm.activations._process_activation_tensor(
            act, inp_ids, None, "cpu"
        )
        
        # Should be reshaped to [B, T, D] = [2, 3, 8]
        assert processed.shape == (2, 3, 8)
    
    def test_process_activation_tensor_without_reshape(self, setup_lm):
        """Test _process_activation_tensor without reshape."""
        lm = setup_lm
        
        # Create 3D activation that doesn't need reshaping
        act = torch.ones(2, 3, 8)  # Already [B, T, D]
        inp_ids = torch.ones(2, 3, dtype=torch.long)
        
        processed = lm.activations._process_activation_tensor(
            act, inp_ids, None, "cpu"
        )
        
        # Should remain [2, 3, 8]
        assert processed.shape == (2, 3, 8)
    
    def test_process_activation_tensor_with_dtype(self, setup_lm):
        """Test _process_activation_tensor with dtype conversion."""
        lm = setup_lm
        
        act = torch.ones(2, 3, 8, dtype=torch.float32)
        processed = lm.activations._process_activation_tensor(
            act, None, torch.float16, "cpu"
        )
        
        assert processed.dtype == torch.float16
    
    def test_process_batch_activations_single_detector(self, setup_lm):
        """Test _process_batch_activations with single detector."""
        from amber.hooks.activation_saver import LayerActivationDetector
        
        lm = setup_lm
        layer_name = lm.layers.get_layer_names()[0]
        
        # Create a detector and set its tensor metadata
        detector = LayerActivationDetector(layer_signature=layer_name)
        tensor = torch.ones(2, 3, 8)
        detector._tensor_metadata['activations'] = tensor
        detector._tensor_batches['activations'] = [tensor]
        
        payload = {}
        inp_ids = torch.ones(2, 3, dtype=torch.long)
        
        result = lm.activations._process_batch_activations(
            detector, layer_name, payload, inp_ids, None, "cpu", True
        )
        
        assert "activations" in result
        assert result["activations"].shape == (2, 3, 8)
    
    def test_process_batch_activations_multiple_detectors(self, setup_lm):
        """Test _process_batch_activations with multiple detectors."""
        from amber.hooks.activation_saver import LayerActivationDetector
        
        lm = setup_lm
        layer_names = lm.layers.get_layer_names()[:2]
        
        detectors = {}
        for layer_name in layer_names:
            detector = LayerActivationDetector(layer_signature=layer_name)
            tensor = torch.ones(2, 3, 8)
            detector._tensor_metadata['activations'] = tensor
            detector._tensor_batches['activations'] = [tensor]
            detectors[layer_name] = detector
        
        payload = {}
        inp_ids = torch.ones(2, 3, dtype=torch.long)
        
        result = lm.activations._process_batch_activations(
            detectors, layer_names, payload, inp_ids, None, "cpu", True
        )
        
        # Should have activations for each layer
        for layer_name in layer_names:
            safe_name = str(layer_name).replace("/", "_")
            assert f"activations_{safe_name}" in result
    
    def test_process_batch_activations_with_none_activations(self, setup_lm):
        """Test _process_batch_activations when detector returns None."""
        from amber.hooks.activation_saver import LayerActivationDetector
        
        lm = setup_lm
        layer_name = lm.layers.get_layer_names()[0]
        
        # Create detector without activations
        detector = LayerActivationDetector(layer_signature=layer_name)
        # Don't set activations, so get_captured() returns None
        
        payload = {}
        inp_ids = torch.ones(2, 3, dtype=torch.long)
        
        result = lm.activations._process_batch_activations(
            detector, layer_name, payload, inp_ids, None, "cpu", True
        )
        
        # Should not have activations key when None
        assert "activations" not in result
    
    def test_process_activation_tensor_exception_paths(self, setup_lm):
        """Test _process_activation_tensor exception handling."""
        lm = setup_lm
        
        # Test reshape exception path
        act = torch.ones(6, 8)
        # Create inp_ids that will cause reshape to fail
        inp_ids = torch.ones(2, 2, dtype=torch.long)  # B*T=4, but act has 6 elements
        
        # Should handle exception gracefully
        processed = lm.activations._process_activation_tensor(
            act, inp_ids, None, "cpu"
        )
        assert processed.shape == (6, 8)  # Should remain unchanged
        
        # Test dtype conversion exception path
        act = torch.ones(2, 3, 8)
        # Use a dtype that might cause issues
        processed = lm.activations._process_activation_tensor(
            act, None, torch.float16, "cpu"
        )
        assert processed.dtype == torch.float16
    
    def test_prepare_run_metadata_with_none_options(self, setup_lm):
        """Test _prepare_run_metadata with None options."""
        lm = setup_lm
        
        run_name, meta = lm.activations._prepare_run_metadata(
            layer_signatures="layer0",
            dataset=None,
            run_name="test",
            options=None  # None options
        )
        
        assert run_name == "test"
        assert "options" in meta
    
    def test_save_run_metadata_error_handling(self, setup_lm):
        """Test _save_run_metadata error handling."""
        from unittest.mock import Mock
        
        lm = setup_lm
        
        # Create a store that raises exception
        class FailingStore:
            def put_run_meta(self, key, value):
                raise RuntimeError("Store failed")
        
        failing_store = FailingStore()
        
        # Should not raise, just log warning
        lm.activations._save_run_metadata(
            failing_store, "test_run", {"key": "value"}, verbose=True
        )
    
    def test_save_activations_per_sequence_empty_texts(self, setup_lm):
        """Test save_activations_per_sequence with empty texts."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        texts = ["hello", "", "world"]  # One empty text
        run_name = lm.activations.save_activations_per_sequence(
            texts=texts,
            layer_signature=layer_name,
            run_name="test_empty",
            verbose=False
        )
        
        # Should skip empty text, so only 2 batches
        batches = lm.store.list_run_batches("test_empty")
        assert len(batches) == 2
    
    def test_save_activations_per_sequence_model_none_error(self, setup_lm):
        """Test save_activations_per_sequence raises error when model is None."""
        lm = setup_lm
        lm.context.model = None
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        with pytest.raises(ValueError, match="Model must be initialized"):
            lm.activations.save_activations_per_sequence(
                texts=["test"],
                layer_signature=layer_name,
                run_name="test"
            )
    
    def test_save_activations_per_sequence_store_none_uses_default(self, setup_lm):
        """Test save_activations_per_sequence uses default store when None."""
        lm = setup_lm
        
        layer_names = lm.layers.get_layer_names()
        layer_name = layer_names[0]
        
        # Should use lm.store when store=None
        run_name = lm.activations.save_activations_per_sequence(
            texts=["test"],
            layer_signature=layer_name,
            run_name="test_store_none",
            store=None,
            verbose=False
        )
        
        batches = lm.store.list_run_batches("test_store_none")
        assert len(batches) == 1

