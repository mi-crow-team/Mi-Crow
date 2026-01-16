"""Tests for LanguageModel device handling."""

from __future__ import annotations

import pytest
import torch
from unittest.mock import MagicMock, patch

from mi_crow.language_model.language_model import LanguageModel
from mi_crow.language_model.utils import get_device_from_model
from tests.unit.fixtures.language_models import create_language_model_from_mock
from tests.unit.fixtures.stores import create_temp_store
from tests.unit.fixtures.models import create_mock_model
from tests.unit.fixtures.tokenizers import create_mock_tokenizer


class TestLanguageModelDeviceInitialization:
    """Tests for device handling during LanguageModel initialization."""

    def test_device_defaults_to_cpu_when_none(self, temp_store):
        """Test that device defaults to CPU when None is passed."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device=None)
        
        assert lm.context.device == "cpu"
        model_device = get_device_from_model(lm.model)
        assert model_device.type == "cpu"

    def test_device_cpu_explicit(self, temp_store):
        """Test that model is moved to CPU when device='cpu' is specified."""
        model = create_mock_model()
        if torch.cuda.is_available():
            model = model.cuda()
        
        tokenizer = create_mock_tokenizer()
        
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device="cpu")
        
        assert lm.context.device == "cpu"
        model_device = get_device_from_model(lm.model)
        assert model_device.type == "cpu"

    def test_device_cuda_normalized_to_cuda_0(self, temp_store):
        """Test that device='cuda' is normalized to 'cuda:0'."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device="cuda")
        
        assert lm.context.device == "cuda:0"
        model_device = get_device_from_model(lm.model)
        assert model_device.type == "cuda"
        assert str(model_device) == "cuda:0"

    def test_device_cuda_0_explicit(self, temp_store):
        """Test that model is moved to cuda:0 when device='cuda:0' is specified."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device="cuda:0")
        
        assert lm.context.device == "cuda:0"
        model_device = get_device_from_model(lm.model)
        assert str(model_device) == "cuda:0"

    def test_device_torch_device_object(self, temp_store):
        """Test that device can be specified as torch.device object."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        
        device = torch.device("cpu")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device=device)
        
        assert lm.context.device == "cpu"
        model_device = get_device_from_model(lm.model)
        assert model_device.type == "cpu"

    def test_device_torch_device_cuda(self, temp_store):
        """Test that device can be specified as torch.device('cuda:0')."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        
        device = torch.device("cuda:0")
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device=device)
        
        assert lm.context.device == "cuda:0"
        model_device = get_device_from_model(lm.model)
        assert str(model_device) == "cuda:0"

    def test_model_moved_from_cuda_to_cpu(self, temp_store):
        """Test that model is moved from CUDA to CPU when device='cpu' is specified."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = create_mock_model()
        model = model.cuda()
        tokenizer = create_mock_tokenizer()
        
        model_device_before = get_device_from_model(model)
        assert model_device_before.type == "cuda"
        
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device="cpu")
        
        model_device_after = get_device_from_model(lm.model)
        assert model_device_after.type == "cpu"

    def test_model_moved_from_cpu_to_cuda(self, temp_store):
        """Test that model is moved from CPU to CUDA when device='cuda' is specified."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        
        model_device_before = get_device_from_model(model)
        assert model_device_before.type == "cpu"
        
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device="cuda")
        
        model_device_after = get_device_from_model(lm.model)
        assert model_device_after.type == "cuda"
        assert str(model_device_after) == "cuda:0"

    def test_device_mps_when_available(self, temp_store):
        """Test that device='mps' works when MPS is available."""
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())
        
        if not mps_available:
            pytest.skip("MPS not available")
        
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device="mps")
        
        assert lm.context.device == "mps"
        model_device = get_device_from_model(lm.model)
        assert model_device.type == "mps"

    def test_device_cuda_raises_when_not_available(self, temp_store):
        """Test that device='cuda' raises ValueError when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            model = create_mock_model()
            tokenizer = create_mock_tokenizer()
            
            with pytest.raises(ValueError, match="CUDA is not available"):
                LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device="cuda")

    def test_device_mps_raises_when_not_available(self, temp_store):
        """Test that device='mps' raises ValueError when MPS is not available."""
        with patch("torch.backends.mps.is_available", return_value=False):
            model = create_mock_model()
            tokenizer = create_mock_tokenizer()
            
            with pytest.raises(ValueError, match="MPS is not available"):
                LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device="mps")


class TestLanguageModelDeviceInference:
    """Tests for device handling during inference."""

    def test_inference_always_uses_context_device(self, temp_store):
        """Test that inference engine always uses context.device, never falls back."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device="cpu")
        
        assert lm.context.device == "cpu"
        model_device_before = get_device_from_model(lm.model)
        assert model_device_before.type == "cpu"
        
        lm.model = lm.model.cuda()
        model_device_after_move = get_device_from_model(lm.model)
        assert model_device_after_move.type == "cuda"
        
        with patch.object(lm.inference, '_run_model_forward') as mock_forward:
            mock_output = MagicMock()
            mock_forward.return_value = mock_output
            
            lm.inference.execute_inference(["test"])
            
            model_device_after_inference = get_device_from_model(lm.model)
            assert model_device_after_inference.type == "cpu"
            assert lm.context.device == "cpu"

    def test_inference_moves_inputs_to_context_device(self, temp_store):
        """Test that inference engine moves all inputs to context.device."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device="cpu")
        
        with patch.object(lm.inference, '_run_model_forward') as mock_forward:
            mock_output = MagicMock()
            mock_forward.return_value = mock_output
            
            with patch('mi_crow.language_model.inference.move_tensors_to_device') as mock_move:
                mock_move.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
                
                lm.inference.execute_inference(["test"])
                
                assert mock_move.call_count >= 1
                call_args = mock_move.call_args
                called_device = call_args[0][1]
                assert str(called_device) == "cpu"


class TestLanguageModelDeviceFromHuggingface:
    """Tests for device handling in from_huggingface factory method."""

    def test_from_huggingface_moves_model_to_device(self, temp_store):
        """Test that from_huggingface moves model to specified device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        with patch('mi_crow.language_model.initialization.AutoTokenizer') as mock_tokenizer_class, \
             patch('mi_crow.language_model.initialization.AutoModelForCausalLM') as mock_model_class:
            
            mock_tokenizer = MagicMock()
            mock_model = MagicMock()
            mock_model_instance = torch.nn.Linear(10, 5)
            
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model_instance
            
            lm = LanguageModel.from_huggingface(
                "test/model",
                store=temp_store,
                device="cuda"
            )
            
            assert lm.context.device == "cuda:0"
            model_device = get_device_from_model(lm.model)
            assert model_device.type == "cuda"

    def test_from_huggingface_defaults_to_cpu(self, temp_store):
        """Test that from_huggingface defaults to CPU when device is None."""
        with patch('mi_crow.language_model.initialization.AutoTokenizer') as mock_tokenizer_class, \
             patch('mi_crow.language_model.initialization.AutoModelForCausalLM') as mock_model_class:
            
            mock_tokenizer = MagicMock()
            mock_model_instance = torch.nn.Linear(10, 5)
            
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model_instance
            
            lm = LanguageModel.from_huggingface(
                "test/model",
                store=temp_store,
                device=None
            )
            
            assert lm.context.device == "cpu"
            model_device = get_device_from_model(lm.model)
            assert model_device.type == "cpu"


class TestControllerDeviceHandling:
    """Tests for Controller device handling with context.device."""

    def test_controller_uses_context_device(self, temp_store):
        """Test that Controller uses context.device when applying modifications."""
        from mi_crow.hooks.controller import Controller
        
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        lm = LanguageModel(model=model, tokenizer=tokenizer, store=temp_store, device="cpu")
        
        class TestController(Controller):
            def modify_activations(self, module, inputs, output):
                return output * 2
        
        controller = TestController()
        lm.layers.register_hook(0, controller)
        
        assert controller.context is not None
        assert controller.context.device == "cpu"


class TestApplyModificationToOutputWithTargetDevice:
    """Tests for apply_modification_to_output with target_device parameter."""

    def test_apply_modification_uses_target_device_for_tensor(self, temp_store):
        """Test that apply_modification_to_output moves tensor to target_device."""
        from mi_crow.hooks.utils import apply_modification_to_output
        
        output = torch.tensor([1.0, 2.0, 3.0])
        modified = torch.tensor([4.0, 5.0, 6.0])
        target_device = torch.device("cpu")
        
        apply_modification_to_output(output, modified, target_device=target_device)
        
        assert torch.allclose(output, modified)
        assert output.device == target_device

    def test_apply_modification_without_target_device_backward_compatible(self, temp_store):
        """Test that apply_modification_to_output works without target_device (backward compatible)."""
        from mi_crow.hooks.utils import apply_modification_to_output
        
        output = torch.tensor([1.0, 2.0, 3.0])
        modified = torch.tensor([4.0, 5.0, 6.0])
        
        apply_modification_to_output(output, modified, target_device=None)
        
        assert torch.allclose(output, modified)
