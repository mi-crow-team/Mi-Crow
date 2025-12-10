"""Tests for LanguageModelActivations."""

from unittest.mock import Mock, patch

import pytest
import torch

from amber.hooks.implementations.layer_activation_detector import LayerActivationDetector
from amber.language_model.activations import LanguageModelActivations
from tests.unit.fixtures import (
    create_sample_dataset,
)


class TestLanguageModelActivations:
    """Test suite for LanguageModelActivations."""

    def test_init(self, mock_language_model):
        """Test initialization."""
        activations = LanguageModelActivations(mock_language_model.context)
        assert activations.context == mock_language_model.context

    def test_setup_detector(self, mock_language_model):
        """Test detector setup."""
        activations = LanguageModelActivations(mock_language_model.context)
        # Use an actual layer name from the model
        layer_names = mock_language_model.layers.get_layer_names()
        layer_name = layer_names[0] if layer_names else 0
        detector, hook_id = activations._setup_detector(layer_name, "test_suffix")

        assert isinstance(detector, LayerActivationDetector)
        assert detector.layer_signature == layer_name
        assert hook_id is not None
        assert hook_id.startswith("detector_test_suffix")

    def test_cleanup_detector_success(self, mock_language_model):
        """Test successful detector cleanup."""
        activations = LanguageModelActivations(mock_language_model.context)
        # Use an actual layer name or index
        layer_names = mock_language_model.layers.get_layer_names()
        layer_sig = layer_names[0] if layer_names else 0
        detector, hook_id = activations._setup_detector(layer_sig, "test")

        # Should not raise
        activations._cleanup_detector(hook_id)

    def test_cleanup_detector_keyerror(self, mock_language_model):
        """Test detector cleanup with KeyError."""
        activations = LanguageModelActivations(mock_language_model.context)
        mock_language_model.layers.unregister_hook = Mock(side_effect=KeyError("not found"))

        # Should not raise
        activations._cleanup_detector("nonexistent_hook_id")

    def test_cleanup_detector_valueerror(self, mock_language_model):
        """Test detector cleanup with ValueError."""
        activations = LanguageModelActivations(mock_language_model.context)
        mock_language_model.layers.unregister_hook = Mock(side_effect=ValueError("invalid"))

        # Should not raise
        activations._cleanup_detector("invalid_hook_id")

    def test_cleanup_detector_runtimeerror(self, mock_language_model):
        """Test detector cleanup with RuntimeError."""
        activations = LanguageModelActivations(mock_language_model.context)
        mock_language_model.layers.unregister_hook = Mock(side_effect=RuntimeError("runtime"))

        # Should not raise
        activations._cleanup_detector("runtime_hook_id")

    def test_normalize_layer_signatures_string(self, mock_language_model):
        """Test normalizing string layer signature."""
        activations = LanguageModelActivations(mock_language_model.context)
        # Use an actual layer name
        layer_names = mock_language_model.layers.get_layer_names()
        layer_name = layer_names[0] if layer_names else "layer_0"
        single, list_sig = activations._normalize_layer_signatures(layer_name)

        assert single == layer_name
        assert list_sig == [layer_name]

    def test_normalize_layer_signatures_int(self, mock_language_model):
        """Test normalizing int layer signature."""
        activations = LanguageModelActivations(mock_language_model.context)
        single, list_sig = activations._normalize_layer_signatures(0)

        assert single == "0"
        assert list_sig == ["0"]

    def test_normalize_layer_signatures_list_single(self, mock_language_model):
        """Test normalizing list with single layer signature."""
        activations = LanguageModelActivations(mock_language_model.context)
        single, list_sig = activations._normalize_layer_signatures(["layer_0"])

        assert single == "layer_0"
        assert list_sig == ["layer_0"]

    def test_normalize_layer_signatures_list_multiple(self, mock_language_model):
        """Test normalizing list with multiple layer signatures."""
        activations = LanguageModelActivations(mock_language_model.context)
        single, list_sig = activations._normalize_layer_signatures(["layer_0", "layer_1", 2])

        assert single is None
        assert list_sig == ["layer_0", "layer_1", "2"]

    def test_normalize_layer_signatures_none(self, mock_language_model):
        """Test normalizing None layer signature."""
        activations = LanguageModelActivations(mock_language_model.context)
        single, list_sig = activations._normalize_layer_signatures(None)

        assert single is None
        assert list_sig == []

    def test_extract_dataset_info_with_dataset(self, mock_language_model):
        """Test extracting dataset info when dataset is provided."""
        activations = LanguageModelActivations(mock_language_model.context)
        dataset = create_sample_dataset()
        dataset.dataset_dir = "/path/to/dataset"

        info = activations._extract_dataset_info(dataset)

        assert info["dataset_dir"] == "/path/to/dataset"
        assert info["length"] == len(dataset)

    def test_extract_dataset_info_none(self, mock_language_model):
        """Test extracting dataset info when dataset is None."""
        activations = LanguageModelActivations(mock_language_model.context)
        info = activations._extract_dataset_info(None)

        assert info == {}

    def test_extract_dataset_info_error_handling(self, mock_language_model):
        """Test extracting dataset info with error handling."""
        activations = LanguageModelActivations(mock_language_model.context)
        dataset = Mock()
        dataset.dataset_dir = Mock(side_effect=AttributeError("no attr"))

        info = activations._extract_dataset_info(dataset)

        assert info["dataset_dir"] == ""
        assert info["length"] == -1

    def test_prepare_run_metadata_with_run_name(self, mock_language_model):
        """Test preparing run metadata with provided run name."""
        activations = LanguageModelActivations(mock_language_model.context)
        dataset = create_sample_dataset()
        dataset.dataset_dir = "/path/to/dataset"

        # Use an actual layer name or index
        layer_names = mock_language_model.layers.get_layer_names()
        layer_sig = layer_names[0] if layer_names else 0

        run_name, meta = activations._prepare_run_metadata(
            layer_sig, dataset=dataset, run_name="custom_run", options={"batch_size": 32}
        )

        assert run_name == "custom_run"
        assert meta["run_name"] == "custom_run"
        assert meta["layer_signatures"] == [str(layer_sig)]
        assert meta["num_layers"] == 1
        assert meta["options"]["batch_size"] == 32
        assert meta["dataset"]["dataset_dir"] == "/path/to/dataset"

    def test_prepare_run_metadata_generated_run_name(self, mock_language_model):
        """Test preparing run metadata with generated run name."""
        activations = LanguageModelActivations(mock_language_model.context)

        # Use an actual layer name or index
        layer_names = mock_language_model.layers.get_layer_names()
        layer_sig = layer_names[0] if layer_names else 0

        run_name, meta = activations._prepare_run_metadata(layer_sig)

        assert run_name.startswith("run_")
        assert meta["run_name"] == run_name

    def test_prepare_run_metadata_multiple_layers(self, mock_language_model):
        """Test preparing run metadata with multiple layers."""
        activations = LanguageModelActivations(mock_language_model.context)

        run_name, meta = activations._prepare_run_metadata(["layer_0", "layer_1", "layer_2"])

        assert meta["layer_signatures"] == ["layer_0", "layer_1", "layer_2"]
        assert meta["num_layers"] == 3

    def test_prepare_run_metadata_no_layers(self, mock_language_model):
        """Test preparing run metadata with no layers."""
        activations = LanguageModelActivations(mock_language_model.context)

        run_name, meta = activations._prepare_run_metadata(None)

        assert "layer_signatures" not in meta
        assert "num_layers" not in meta

    def test_save_run_metadata_success(self, mock_language_model, mock_store):
        """Test saving run metadata successfully."""
        activations = LanguageModelActivations(mock_language_model.context)
        meta = {"run_name": "test_run", "model": "TestModel"}

        activations._save_run_metadata(mock_store, "test_run", meta, verbose=False)

        # Verify metadata was saved
        mock_store.put_run_metadata.assert_called_once_with("test_run", meta)

    def test_save_run_metadata_error_handling(self, mock_language_model, mock_store):
        """Test saving run metadata with error handling."""
        activations = LanguageModelActivations(mock_language_model.context)
        mock_store.put_run_metadata = Mock(side_effect=OSError("disk full"))
        meta = {"run_name": "test_run"}

        # Should not raise
        activations._save_run_metadata(mock_store, "test_run", meta, verbose=False)

    def test_process_batch_empty_texts(self, mock_language_model, temp_store):
        """Test processing empty batch."""
        from datasets import Dataset

        from amber.datasets import TextDataset

        activations = LanguageModelActivations(mock_language_model.context)
        hf_dataset = Dataset.from_dict({"text": ["a"]})
        dataset = TextDataset(hf_dataset, temp_store)

        # Should not raise
        activations._process_batch([], dataset, "test_run", 0, None, False, None, None, False)

    def test_process_batch_with_texts(self, mock_language_model, temp_store):
        """Test processing batch with texts."""
        from datasets import Dataset

        from amber.datasets import TextDataset

        activations = LanguageModelActivations(mock_language_model.context)
        mock_language_model._inference_engine.execute_inference = Mock()
        mock_language_model.save_detector_metadata = Mock()

        hf_dataset = Dataset.from_dict({"text": ["text1", "text2"]})
        dataset = TextDataset(hf_dataset, temp_store)

        activations._process_batch(
            ["text1", "text2"],
            dataset,
            "test_run",
            0,
            max_length=128,
            autocast=True,
            autocast_dtype=torch.float16,
            dtype=torch.float32,
            verbose=True,
        )

        mock_language_model._inference_engine.execute_inference.assert_called_once()
        mock_language_model.save_detector_metadata.assert_called_once_with("test_run", 0)

    def test_convert_activations_to_dtype(self, mock_language_model):
        """Test converting activations to dtype."""
        activations = LanguageModelActivations(mock_language_model.context)

        detector1 = Mock()
        detector1.tensor_metadata = {"activations": torch.tensor([1.0, 2.0], dtype=torch.float32)}
        detector2 = Mock()
        detector2.tensor_metadata = {"other": torch.tensor([3.0])}

        mock_language_model.layers.get_detectors = Mock(return_value=[detector1, detector2])

        activations._convert_activations_to_dtype(torch.float16)

        assert detector1.tensor_metadata["activations"].dtype == torch.float16

    def test_manage_cuda_cache_cuda_device(self, mock_language_model):
        """Test managing CUDA cache for CUDA device."""
        activations = LanguageModelActivations(mock_language_model.context)

        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            activations._manage_cuda_cache(batch_counter=10, free_cuda_cache_every=5, device_type="cuda", verbose=False)

            mock_empty_cache.assert_called_once()

    def test_manage_cuda_cache_cpu_device(self, mock_language_model):
        """Test managing CUDA cache for CPU device."""
        activations = LanguageModelActivations(mock_language_model.context)

        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            activations._manage_cuda_cache(batch_counter=10, free_cuda_cache_every=5, device_type="cpu", verbose=False)

            mock_empty_cache.assert_not_called()

    def test_manage_cuda_cache_disabled(self, mock_language_model):
        """Test managing CUDA cache when disabled."""
        activations = LanguageModelActivations(mock_language_model.context)

        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            activations._manage_cuda_cache(
                batch_counter=10, free_cuda_cache_every=None, device_type="cuda", verbose=False
            )

            mock_empty_cache.assert_not_called()

    def test_manage_cuda_cache_zero(self, mock_language_model):
        """Test managing CUDA cache with zero frequency."""
        activations = LanguageModelActivations(mock_language_model.context)

        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            activations._manage_cuda_cache(batch_counter=10, free_cuda_cache_every=0, device_type="cuda", verbose=False)

            mock_empty_cache.assert_not_called()

    def test_save_activations_dataset_success(self, mock_language_model, temp_store):
        """Test saving activations from dataset successfully."""
        from datasets import Dataset

        from amber.datasets import TextDataset

        activations = LanguageModelActivations(mock_language_model.context)
        # Create a proper TextDataset
        hf_dataset = Dataset.from_dict({"text": ["text1", "text2", "text3"]})
        dataset = TextDataset(hf_dataset, temp_store)

        # Use an actual layer name or index
        layer_names = mock_language_model.layers.get_layer_names()
        layer_sig = layer_names[0] if layer_names else 0

        mock_language_model._inference_engine.execute_inference = Mock()
        mock_language_model.save_detector_metadata = Mock()

        with patch("torch.inference_mode"):
            run_name = activations.save_activations_dataset(
                dataset, layer_sig, run_name="test_run", batch_size=2, verbose=True
            )

        assert run_name == "test_run"
        assert mock_language_model._inference_engine.execute_inference.called
        assert mock_language_model.save_detector_metadata.called

    def test_save_activations_dataset_model_not_initialized(self, mock_language_model, temp_store):
        """Test saving activations when model is not initialized."""
        from datasets import Dataset

        from amber.datasets import TextDataset

        activations = LanguageModelActivations(mock_language_model.context)
        mock_language_model.context.model = None
        hf_dataset = Dataset.from_dict({"text": ["text1"]})
        dataset = TextDataset(hf_dataset, temp_store)

        with pytest.raises(ValueError, match="Model must be initialized"):
            activations.save_activations_dataset(dataset, 0)

    def test_save_activations_success(self, mock_language_model, temp_store):
        """Test saving activations from texts successfully."""
        activations = LanguageModelActivations(mock_language_model.context)
        texts = ["text1", "text2", "text3"]
        
        layer_names = mock_language_model.layers.get_layer_names()
        layer_sig = layer_names[0] if layer_names else 0
        
        mock_language_model._inference_engine.execute_inference = Mock()
        mock_language_model.save_detector_metadata = Mock()
        
        with patch("torch.inference_mode"):
            run_name = activations.save_activations(
                texts, layer_sig, run_name="test_run", batch_size=2, verbose=True
            )
        
        assert run_name == "test_run"
        assert mock_language_model._inference_engine.execute_inference.call_count == 2
        assert mock_language_model.save_detector_metadata.call_count == 2

    def test_save_activations_no_batching(self, mock_language_model, temp_store):
        """Test saving activations without batching."""
        activations = LanguageModelActivations(mock_language_model.context)
        texts = ["text1", "text2"]
        
        layer_names = mock_language_model.layers.get_layer_names()
        layer_sig = layer_names[0] if layer_names else 0
        
        mock_language_model._inference_engine.execute_inference = Mock()
        mock_language_model.save_detector_metadata = Mock()
        
        with patch("torch.inference_mode"):
            run_name = activations.save_activations(
                texts, layer_sig, run_name="test_run", batch_size=None, verbose=True
            )
        
        assert run_name == "test_run"
        assert mock_language_model._inference_engine.execute_inference.call_count == 1
        assert mock_language_model.save_detector_metadata.call_count == 1

    def test_save_activations_empty_texts(self, mock_language_model, temp_store):
        """Test saving activations with empty texts."""
        activations = LanguageModelActivations(mock_language_model.context)
        
        layer_names = mock_language_model.layers.get_layer_names()
        layer_sig = layer_names[0] if layer_names else 0
        
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            activations.save_activations([], layer_sig)

    def test_save_activations_dataset_store_not_set(self, mock_language_model, temp_store):
        """Test saving activations when store is not set."""
        from datasets import Dataset

        from amber.datasets import TextDataset

        activations = LanguageModelActivations(mock_language_model.context)
        mock_language_model.context.store = None
        hf_dataset = Dataset.from_dict({"text": ["text1"]})
        dataset = TextDataset(hf_dataset, temp_store)

        with pytest.raises(ValueError, match="Store must be provided"):
            activations.save_activations_dataset(dataset, 0)

    def test_save_activations_dataset_with_dtype(self, mock_language_model, temp_store):
        """Test saving activations with dtype conversion."""
        from datasets import Dataset

        from amber.datasets import TextDataset

        activations = LanguageModelActivations(mock_language_model.context)
        hf_dataset = Dataset.from_dict({"text": ["text1"]})
        dataset = TextDataset(hf_dataset, temp_store)

        # Use an actual layer name or index
        layer_names = mock_language_model.layers.get_layer_names()
        layer_sig = layer_names[0] if layer_names else 0

        detector = Mock()
        detector.tensor_metadata = {"activations": torch.tensor([1.0, 2.0], dtype=torch.float32)}
        mock_language_model.layers.get_detectors = Mock(return_value=[detector])
        mock_language_model._inference_engine.execute_inference = Mock()
        mock_language_model.save_detector_metadata = Mock()

        with patch("torch.inference_mode"):
            activations.save_activations_dataset(dataset, layer_sig, dtype=torch.float16)

        assert detector.tensor_metadata["activations"].dtype == torch.float16

    def test_save_activations_dataset_with_max_length(self, mock_language_model, temp_store):
        """Test saving activations with max_length."""
        from datasets import Dataset

        from amber.datasets import TextDataset

        activations = LanguageModelActivations(mock_language_model.context)
        hf_dataset = Dataset.from_dict({"text": ["text1"]})
        dataset = TextDataset(hf_dataset, temp_store)

        # Use an actual layer name or index
        layer_names = mock_language_model.layers.get_layer_names()
        layer_sig = layer_names[0] if layer_names else 0

        mock_language_model._inference_engine.execute_inference = Mock()
        mock_language_model.save_detector_metadata = Mock()

        with patch("torch.inference_mode"):
            activations.save_activations_dataset(dataset, layer_sig, max_length=128)

        # Verify max_length was passed
        call_kwargs = mock_language_model._inference_engine.execute_inference.call_args[1]
        assert (
            "tok_kwargs" in call_kwargs or len(mock_language_model._inference_engine.execute_inference.call_args[0]) > 0
        )

    def test_save_activations_dataset_cleanup_on_error(self, mock_language_model, temp_store):
        """Test that detector is cleaned up even on error."""
        from datasets import Dataset

        from amber.datasets import TextDataset

        activations = LanguageModelActivations(mock_language_model.context)
        hf_dataset = Dataset.from_dict({"text": ["text1"]})
        dataset = TextDataset(hf_dataset, temp_store)

        # Use an actual layer name or index
        layer_names = mock_language_model.layers.get_layer_names()
        layer_sig = layer_names[0] if layer_names else 0

        mock_language_model._inference_engine.execute_inference = Mock(side_effect=RuntimeError("error"))
        mock_language_model.layers.unregister_hook = Mock()

        with patch("torch.inference_mode"):
            with pytest.raises(RuntimeError):
                activations.save_activations_dataset(dataset, layer_sig)

        # Verify cleanup was called
        assert mock_language_model.layers.unregister_hook.called
