"""Tests for Sae abstract base class."""

from unittest.mock import MagicMock, Mock

import pytest
import torch

from mi_crow.hooks.hook import HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT, HookType
from mi_crow.hooks.utils import extract_tensor_from_input, extract_tensor_from_output
from mi_crow.mechanistic.sae.sae import Sae


class ConcreteSae(Sae):
    """Concrete implementation of Sae for testing."""

    def _initialize_sae_engine(self):
        mock_engine = MagicMock()
        return mock_engine

    def modify_activations(self, module, inputs, output):
        return output * 0.5

    def process_activations(
        self, module: torch.nn.Module, input: HOOK_FUNCTION_INPUT, output: HOOK_FUNCTION_OUTPUT
    ) -> None:
        """Process activations for testing."""
        if self.hook_type == HookType.FORWARD:
            tensor = extract_tensor_from_output(output)

        else:
            tensor = extract_tensor_from_input(input)

        if tensor is None or not isinstance(tensor, torch.Tensor):
            return

        original_shape = tensor.shape
        if len(original_shape) > 2:
            batch_size, seq_len = original_shape[:2]
            tensor_flat = tensor.reshape(-1, original_shape[-1])

        else:
            batch_size = original_shape[0]
            seq_len = 1
            tensor_flat = tensor

        latents = self.encode(tensor_flat)
        latents_cpu = latents.detach().cpu()
        if len(original_shape) > 2:
            latents_reshaped = latents_cpu.reshape(batch_size, seq_len, -1)

        else:
            latents_reshaped = latents_cpu

        self.tensor_metadata["neurons"] = latents_reshaped
        batch_items = []
        n_items = latents_cpu.shape[0]
        for item_idx in range(n_items):
            item_latents = latents_cpu[item_idx]
            nonzero_mask = item_latents != 0
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).flatten().tolist()
            activations_map = {int(idx): float(item_latents[idx].item()) for idx in nonzero_indices}
            item_metadata = {"nonzero_indices": nonzero_indices, "activations": activations_map}
            batch_items.append(item_metadata)

        self.metadata["batch_items"] = batch_items

    def encode(self, x):
        return x

    def decode(self, x):
        return x

    def forward(self, x):
        return x

    def save(self, name):
        pass

    @staticmethod
    def load(path):
        return ConcreteSae(10, 20)


class TestSaeInitialization:
    """Tests for Sae initialization."""

    def test_init_with_parameters(self):
        """Test initialization with parameters."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        assert sae.context.n_latents == 100
        assert sae.context.n_inputs == 200
        assert sae.context.device == "cpu"
        assert sae.hook_type == HookType.FORWARD

    def test_init_with_device(self):
        """Test initialization with device."""
        sae = ConcreteSae(n_latents=100, n_inputs=200, device="cpu")
        assert sae.context.device == "cpu"

    def test_init_creates_context(self):
        """Test that context is created."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        assert sae.context is not None
        assert sae.context.autoencoder == sae

    def test_init_creates_trainer(self):
        """Test that trainer is created."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        assert sae.trainer is not None


class TestSaeMethods:
    """Tests for Sae methods."""

    def test_encode(self):
        """Test encode method."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        x = torch.randn(5, 200)
        result = sae.encode(x)
        assert result is not None

    def test_decode(self):
        """Test decode method."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        x = torch.randn(5, 100)
        result = sae.decode(x)
        assert result is not None

    def test_forward(self):
        """Test forward method."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        x = torch.randn(5, 200)
        result = sae.forward(x)
        assert result is not None

    def test_modify_activations(self):
        """Test modify_activations method."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        from torch import nn

        module = nn.Linear(200, 100)
        output = torch.randn(5, 100)
        result = sae.modify_activations(module, None, output)
        assert result is not None
        assert torch.allclose(result, output * 0.5)


class TestSaeStaticMethods:
    """Tests for Sae static methods."""

    def test_apply_activation_fn_relu(self):
        """Test _apply_activation_fn with relu."""
        tensor = torch.tensor([-1.0, 0.0, 1.0])
        result = Sae._apply_activation_fn(tensor, "relu")
        assert torch.equal(result, torch.relu(tensor))

    def test_apply_activation_fn_linear(self):
        """Test _apply_activation_fn with linear."""
        tensor = torch.tensor([-1.0, 0.0, 1.0])
        result = Sae._apply_activation_fn(tensor, "linear")
        assert torch.equal(result, tensor)

    def test_apply_activation_fn_none(self):
        """Test _apply_activation_fn with None."""
        tensor = torch.tensor([-1.0, 0.0, 1.0])
        result = Sae._apply_activation_fn(tensor, None)
        assert torch.equal(result, tensor)

    def test_apply_activation_fn_invalid_raises_error(self):
        """Test that invalid activation function raises ValueError."""
        tensor = torch.tensor([1.0])
        with pytest.raises(ValueError, match="Unknown activation function"):
            Sae._apply_activation_fn(tensor, "invalid")


class TestSaeProcessActivations:
    """Tests for SAE process_activations detector functionality."""

    def test_process_activations_2d_saves_batch_items(self):
        """Test that process_activations saves metadata for each item in 2D batch."""
        sae = ConcreteSae(n_latents=5, n_inputs=10)
        module = Mock()
        latents = torch.tensor(
            [
                [1.0, 0.0, 2.0, 0.0, 3.0],
                [0.0, 4.0, 0.0, 5.0, 0.0],
                [6.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        sae.encode = Mock(return_value=latents)
        output = torch.randn(3, 10)
        sae.process_activations(module, (), output)
        assert "batch_items" in sae.metadata
        batch_items = sae.metadata["batch_items"]
        assert len(batch_items) == 3
        item_0 = batch_items[0]
        assert item_0["nonzero_indices"] == [0, 2, 4]
        assert item_0["activations"] == {0: 1.0, 2: 2.0, 4: 3.0}
        item_1 = batch_items[1]
        assert item_1["nonzero_indices"] == [1, 3]
        assert item_1["activations"] == {1: 4.0, 3: 5.0}
        item_2 = batch_items[2]
        assert item_2["nonzero_indices"] == [0]
        assert item_2["activations"] == {0: 6.0}
        assert "neurons" in sae.tensor_metadata
        assert torch.equal(sae.tensor_metadata["neurons"], latents.cpu())

    def test_process_activations_3d_saves_batch_items(self):
        """Test that process_activations saves metadata for each item in 3D batch."""
        sae = ConcreteSae(n_latents=4, n_inputs=8)
        module = Mock()
        latents_flat = torch.tensor(
            [
                [1.0, 0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 4.0],
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 6.0, 7.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        sae.encode = Mock(return_value=latents_flat)
        output = torch.randn(2, 3, 8)
        sae.process_activations(module, (), output)
        assert "batch_items" in sae.metadata
        batch_items = sae.metadata["batch_items"]
        assert len(batch_items) == 6
        item_0 = batch_items[0]
        assert item_0["nonzero_indices"] == [0, 2]
        assert item_0["activations"] == {0: 1.0, 2: 2.0}
        item_5 = batch_items[5]
        assert item_5["nonzero_indices"] == []
        assert item_5["activations"] == {}
        assert "neurons" in sae.tensor_metadata
        neurons_tensor = sae.tensor_metadata["neurons"]
        assert neurons_tensor.shape == (2, 3, 4)
        assert torch.equal(neurons_tensor, latents_flat.reshape(2, 3, 4).cpu())

    def test_process_activations_handles_all_zeros(self):
        """Test that process_activations handles items with all zero activations."""
        sae = ConcreteSae(n_latents=3, n_inputs=6)
        module = Mock()
        latents = torch.zeros(2, 3)
        sae.encode = Mock(return_value=latents)
        output = torch.randn(2, 6)
        sae.process_activations(module, (), output)
        batch_items = sae.metadata["batch_items"]
        assert len(batch_items) == 2
        for item in batch_items:
            assert item["nonzero_indices"] == []
            assert item["activations"] == {}

    def test_process_activations_handles_all_nonzero(self):
        """Test that process_activations handles items with all nonzero activations."""
        sae = ConcreteSae(n_latents=3, n_inputs=6)
        module = Mock()
        latents = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        sae.encode = Mock(return_value=latents)
        output = torch.randn(2, 6)
        sae.process_activations(module, (), output)
        batch_items = sae.metadata["batch_items"]
        assert len(batch_items) == 2
        item_0 = batch_items[0]
        assert set(item_0["nonzero_indices"]) == {0, 1, 2}
        assert item_0["activations"] == {0: 1.0, 1: 2.0, 2: 3.0}
        item_1 = batch_items[1]
        assert set(item_1["nonzero_indices"]) == {0, 1, 2}
        assert item_1["activations"] == {0: 4.0, 1: 5.0, 2: 6.0}

    def test_process_activations_handles_negative_activations(self):
        """Test that process_activations correctly handles negative activations."""
        sae = ConcreteSae(n_latents=4, n_inputs=8)
        module = Mock()
        latents = torch.tensor(
            [
                [1.0, -2.0, 0.0, 3.0],
                [0.0, 0.0, -4.0, 0.0],
            ]
        )
        sae.encode = Mock(return_value=latents)
        output = torch.randn(2, 8)
        sae.process_activations(module, (), output)
        batch_items = sae.metadata["batch_items"]
        item_0 = batch_items[0]
        assert set(item_0["nonzero_indices"]) == {0, 1, 3}
        assert item_0["activations"] == {0: 1.0, 1: -2.0, 3: 3.0}
        item_1 = batch_items[1]
        assert item_1["nonzero_indices"] == [2]
        assert item_1["activations"] == {2: -4.0}

    def test_process_activations_handles_none_output(self):
        """Test that process_activations handles None output gracefully."""
        sae = ConcreteSae(n_latents=5, n_inputs=10)
        module = Mock()
        sae.process_activations(module, (), None)
        assert "batch_items" not in sae.metadata or len(sae.metadata.get("batch_items", [])) == 0

    def test_process_activations_handles_pre_forward_hook(self):
        """Test that process_activations works with PRE_FORWARD hook type."""
        sae = ConcreteSae(n_latents=3, n_inputs=6)
        sae.hook_type = HookType.PRE_FORWARD
        module = Mock()
        latents = torch.tensor(
            [
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 0.0],
            ]
        )
        sae.encode = Mock(return_value=latents)
        input_tensor = torch.randn(2, 6)
        sae.process_activations(module, (input_tensor,), None)
        assert "batch_items" in sae.metadata
        batch_items = sae.metadata["batch_items"]
        assert len(batch_items) == 2

    def test_process_activations_activations_are_floats(self):
        """Test that activations in metadata are Python floats, not tensors."""
        sae = ConcreteSae(n_latents=3, n_inputs=6)
        module = Mock()
        latents = torch.tensor(
            [
                [1.5, 2.7, 3.9],
            ]
        )
        sae.encode = Mock(return_value=latents)
        output = torch.randn(1, 6)
        sae.process_activations(module, (), output)
        batch_items = sae.metadata["batch_items"]
        item = batch_items[0]
        assert isinstance(item["activations"][0], float)
        assert isinstance(item["activations"][1], float)
        assert isinstance(item["activations"][2], float)
        assert abs(item["activations"][0] - 1.5) < 1e-6
        assert abs(item["activations"][1] - 2.7) < 1e-6
        assert abs(item["activations"][2] - 3.9) < 1e-6

    def test_process_activations_indices_are_ints(self):
        """Test that indices in metadata are Python ints."""
        sae = ConcreteSae(n_latents=5, n_inputs=10)
        module = Mock()
        latents = torch.tensor(
            [
                [1.0, 0.0, 2.0, 0.0, 3.0],
            ]
        )
        sae.encode = Mock(return_value=latents)
        output = torch.randn(1, 10)
        sae.process_activations(module, (), output)
        batch_items = sae.metadata["batch_items"]
        item = batch_items[0]
        assert all(isinstance(idx, int) for idx in item["nonzero_indices"])
        assert all(isinstance(idx, int) for idx in item["activations"].keys())


class TestSaeMultipleInheritance:
    """Tests for SAE multiple inheritance (Controller and Detector)."""

    def test_sae_has_both_controller_and_detector_methods(self):
        """Test that SAE has methods from both Controller and Detector."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        assert hasattr(sae, "modify_activations")
        assert callable(sae.modify_activations)
        assert hasattr(sae, "process_activations")
        assert callable(sae.process_activations)
        assert hasattr(sae, "metadata")
        assert hasattr(sae, "tensor_metadata")
        assert hasattr(sae, "store")

    def test_sae_hook_fn_calls_both_methods(self):
        """Test that SAE's _hook_fn calls both process_activations and modify_activations."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        from torch import nn

        module = nn.Linear(200, 100)
        input_tensor = torch.randn(2, 200)
        output_tensor = torch.randn(2, 100)
        process_called = []
        original_process = sae.process_activations

        def tracked_process(*args, **kwargs):
            process_called.append(True)
            return original_process(*args, **kwargs)

        sae.process_activations = tracked_process
        modify_called = []
        original_modify = sae.modify_activations

        def tracked_modify(*args, **kwargs):
            modify_called.append(True)
            return original_modify(*args, **kwargs)

        sae.modify_activations = tracked_modify
        sae._hook_fn(module, (input_tensor,), output_tensor)
        assert len(process_called) == 1, "process_activations should be called once"
        assert len(modify_called) == 1, "modify_activations should be called once"

    def test_sae_hook_fn_calls_process_before_modify(self):
        """Test that process_activations is called before modify_activations."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        from torch import nn

        module = nn.Linear(200, 100)
        input_tensor = torch.randn(2, 200)
        output_tensor = torch.randn(2, 100)
        call_order = []
        original_process = sae.process_activations

        def tracked_process(*args, **kwargs):
            call_order.append("process")
            return original_process(*args, **kwargs)

        sae.process_activations = tracked_process
        original_modify = sae.modify_activations

        def tracked_modify(*args, **kwargs):
            call_order.append("modify")
            return original_modify(*args, **kwargs)

        sae.modify_activations = tracked_modify
        sae._hook_fn(module, (input_tensor,), output_tensor)
        assert call_order == ["process", "modify"]

    def test_sae_hook_fn_handles_process_error_gracefully(self):
        """Test that process_activations errors don't prevent modify_activations from running."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        from torch import nn

        module = nn.Linear(200, 100)
        input_tensor = torch.randn(2, 200)
        output_tensor = torch.randn(2, 100)

        def failing_process(*args, **kwargs):
            raise ValueError("Process error")

        sae.process_activations = failing_process
        modify_called = []
        original_modify = sae.modify_activations

        def tracked_modify(*args, **kwargs):
            modify_called.append(True)
            return original_modify(*args, **kwargs)

        sae.modify_activations = tracked_modify
        sae._hook_fn(module, (input_tensor,), output_tensor)
        assert len(modify_called) == 1

    def test_sae_hook_fn_raises_on_modify_error(self):
        """Test that modify_activations errors raise RuntimeError."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        from torch import nn

        module = nn.Linear(200, 100)
        input_tensor = torch.randn(2, 200)
        output_tensor = torch.randn(2, 100)

        def failing_modify(*args, **kwargs):
            raise ValueError("Modify error")

        sae.modify_activations = failing_modify
        with pytest.raises(RuntimeError, match="Error in controller"):
            sae._hook_fn(module, (input_tensor,), output_tensor)

    def test_sae_hook_fn_respects_enabled_flag(self):
        """Test that SAE hook doesn't call methods when disabled."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        sae.disable()
        from torch import nn

        module = nn.Linear(200, 100)
        input_tensor = torch.randn(2, 200)
        output_tensor = torch.randn(2, 100)
        process_called = []
        original_process = sae.process_activations

        def tracked_process(*args, **kwargs):
            process_called.append(True)
            return original_process(*args, **kwargs)

        sae.process_activations = tracked_process
        modify_called = []
        original_modify = sae.modify_activations

        def tracked_modify(*args, **kwargs):
            modify_called.append(True)
            return original_modify(*args, **kwargs)

        sae.modify_activations = tracked_modify
        result = sae._hook_fn(module, (input_tensor,), output_tensor)
        assert len(process_called) == 0
        assert len(modify_called) == 0
        assert result is None


class TestSaeContext:
    """Tests for SAE context management."""

    def test_context_property_getter(self):
        """Test context property getter."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        context = sae.context
        assert context is not None
        assert context.autoencoder == sae
        assert context.n_latents == 100
        assert context.n_inputs == 200

    def test_context_property_setter(self):
        """Test context property setter."""
        from mi_crow.mechanistic.sae.autoencoder_context import AutoencoderContext

        sae = ConcreteSae(n_latents=100, n_inputs=200)
        new_context = AutoencoderContext(autoencoder=sae, n_latents=150, n_inputs=250)
        sae.context = new_context
        assert sae.context == new_context
        assert sae.context.n_latents == 150
        assert sae.context.n_inputs == 250

    def test_set_context_syncs_language_model_context(self):
        """Test that set_context syncs LanguageModelContext to AutoencoderContext."""
        from unittest.mock import Mock

        from mi_crow.language_model.context import LanguageModelContext

        sae = ConcreteSae(n_latents=100, n_inputs=200)
        lm_context = LanguageModelContext(language_model=Mock())
        lm_context.language_model = Mock()
        lm_context.model_id = "test_model"
        lm_context.store = Mock()
        sae.layer_signature = 5
        sae.set_context(lm_context)
        assert sae.context.lm == lm_context.language_model
        assert sae.context.model_id == "test_model"
        assert sae.context.store == lm_context.store
        assert sae.context.lm_layer_signature == 5

    def test_set_context_sets_hook_context(self):
        """Test that set_context also sets Hook context."""
        from unittest.mock import Mock

        from mi_crow.language_model.context import LanguageModelContext

        sae = ConcreteSae(n_latents=100, n_inputs=200)
        lm_context = LanguageModelContext(language_model=Mock())
        sae.set_context(lm_context)
        assert sae._context == lm_context

    def test_set_context_handles_none_context(self):
        """Test that set_context handles None context gracefully."""
        sae = ConcreteSae(n_latents=100, n_inputs=200)
        sae.set_context(None)
        assert sae._context is None

    def test_set_context_preserves_existing_store(self):
        """Test that set_context preserves existing store if context store is None."""
        from unittest.mock import Mock

        from mi_crow.language_model.context import LanguageModelContext

        sae = ConcreteSae(n_latents=100, n_inputs=200)
        existing_store = Mock()
        sae.context.store = existing_store
        lm_context = LanguageModelContext(language_model=Mock())
        lm_context.store = None
        sae.set_context(lm_context)
        assert sae.context.store == existing_store

    def test_set_context_updates_store_when_context_has_store(self):
        """Test that set_context updates store when context has store."""
        from unittest.mock import Mock

        from mi_crow.language_model.context import LanguageModelContext

        sae = ConcreteSae(n_latents=100, n_inputs=200)
        new_store = Mock()
        lm_context = LanguageModelContext(language_model=Mock())
        lm_context.store = new_store
        sae.set_context(lm_context)
        assert sae.context.store == new_store


class TestSaeApplyActivationFn:
    """Tests for _apply_activation_fn static method."""

    @pytest.mark.parametrize(
        "activation_fn,expected_fn",
        [
            ("relu", torch.relu),
            ("linear", lambda x: x),
            (None, lambda x: x),
        ],
    )
    def test_apply_activation_fn_valid(self, activation_fn, expected_fn):
        """Test _apply_activation_fn with valid activation functions."""
        tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = Sae._apply_activation_fn(tensor, activation_fn)
        expected = expected_fn(tensor)
        assert torch.equal(result, expected)

    def test_apply_activation_fn_relu_zeros_negatives(self):
        """Test that relu activation zeros negative values."""
        tensor = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
        result = Sae._apply_activation_fn(tensor, "relu")
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 5.0])
        assert torch.equal(result, expected)

    def test_apply_activation_fn_linear_preserves_values(self):
        """Test that linear activation preserves all values."""
        tensor = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
        result = Sae._apply_activation_fn(tensor, "linear")
        assert torch.equal(result, tensor)

    def test_apply_activation_fn_none_preserves_values(self):
        """Test that None activation preserves all values."""
        tensor = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
        result = Sae._apply_activation_fn(tensor, None)
        assert torch.equal(result, tensor)

    def test_apply_activation_fn_invalid_raises_error(self):
        """Test that invalid activation function raises ValueError."""
        tensor = torch.tensor([1.0])
        with pytest.raises(ValueError, match="Unknown activation function"):
            Sae._apply_activation_fn(tensor, "sigmoid")

        with pytest.raises(ValueError, match="Unknown activation function"):
            Sae._apply_activation_fn(tensor, "tanh")

        with pytest.raises(ValueError, match="Unknown activation function"):
            Sae._apply_activation_fn(tensor, "")

    def test_apply_activation_fn_preserves_tensor_properties(self):
        """Test that _apply_activation_fn preserves tensor device and dtype."""
        tensor = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        result = Sae._apply_activation_fn(tensor, "relu")
        assert result.dtype == tensor.dtype
        assert result.device == tensor.device
