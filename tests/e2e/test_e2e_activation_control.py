"""
End-to-end test for activation control with custom controllers.

Based on examples/03_load_concepts.ipynb - demonstrates
creating custom controllers to manipulate model activations during inference.
"""
import pytest
import torch
from typing import Any

from mi_crow.language_model.language_model import LanguageModel
from mi_crow.hooks import Controller, HookType
from mi_crow.store.local_store import LocalStore


class SimpleActivationController(Controller):
    """
    A simple controller that scales activations by a factor.
    Used to demonstrate activation manipulation.
    """
    
    def __init__(self, layer_signature, scale_factor=1.0, hook_id=None):
        super().__init__(hook_type=HookType.FORWARD, hook_id=hook_id, layer_signature=layer_signature)
        self.scale_factor = scale_factor
    
    def modify_activations(self, module, inputs, output):
        """Scale the output activations."""
        if isinstance(output, torch.Tensor):
            return output * self.scale_factor
        elif isinstance(output, (tuple, list)):
            # Scale the first tensor in the output
            modified = list(output)
            if isinstance(modified[0], torch.Tensor):
                modified[0] = modified[0] * self.scale_factor
            return tuple(modified) if isinstance(output, tuple) else modified
        return output


class ActivationCapturingController(Controller):
    """
    Controller that captures activations for verification.
    """
    
    def __init__(self, layer_signature, hook_id=None):
        super().__init__(hook_type=HookType.FORWARD, hook_id=hook_id, layer_signature=layer_signature)
        self.captured_inputs = []
        self.captured_outputs = []
        self.captured_modified = []
    
    def modify_activations(self, module, inputs, output):
        """Capture and modify activations."""
        # Store original
        if isinstance(output, torch.Tensor):
            self.captured_outputs.append(output.detach().cpu().clone())
        
        # Modify (double the activations)
        modified = output * 2.0 if isinstance(output, torch.Tensor) else output
        
        # Store modified - clone to ensure we capture the exact value
        if isinstance(modified, torch.Tensor):
            self.captured_modified.append(modified.detach().cpu().clone())
        
        return modified
    
    def clear_captures(self):
        """Clear captured activations."""
        self.captured_inputs.clear()
        self.captured_outputs.clear()
        self.captured_modified.clear()


def test_e2e_activation_control_amplification():
    """
    Test activation amplification workflow:
    1. Load language model
    2. Create amplification controller
    3. Register controller
    4. Run inference and verify amplification
    5. Disable controller and verify normal operation
    """
    MODEL_ID = "sshleifer/tiny-gpt2"
    LAYER_SIGNATURE = "gpt2lmheadmodel_transformer_h_0_mlp_c_fc"
    DEVICE = "cpu"
    
    # Step 1: Load language model
    print("\n📥 Loading language model...")
    import tempfile
    from pathlib import Path
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / "store")
    model = LanguageModel.from_huggingface(MODEL_ID, store=store)
    model.model.to(DEVICE)
    
    assert model.model is not None
    print(f"✅ Model loaded: {model.model_id}")
    
    # Step 2: Create amplification controller
    print("\n🎛️ Creating amplification controller...")
    SCALE_FACTOR = 2.0
    controller = SimpleActivationController(
        LAYER_SIGNATURE,
        scale_factor=SCALE_FACTOR,
        hook_id="test_amplifier"
    )
    
    assert controller.scale_factor == SCALE_FACTOR
    assert controller.enabled
    print(f"✅ Controller created: scale_factor={SCALE_FACTOR}")
    
    # Step 3: Register controller
    print("\n🔗 Registering controller...")
    hook_id = model.layers.register_hook(LAYER_SIGNATURE, controller)
    
    assert hook_id == "test_amplifier"
    registered_hooks = model.layers.get_hooks(LAYER_SIGNATURE)
    assert len(registered_hooks) > 0
    print(f"✅ Controller registered: {hook_id}")
    
    # Step 4: Run inference with controller enabled
    print("\n🔍 Running inference with controller enabled...")
    test_text = "The cat sat on the mat."
    
    output_with_controller = model.inference.execute_inference([test_text])
    
    assert output_with_controller is not None
    print("✅ Inference with controller completed")
    
    # Step 5: Disable controller and run again
    print("\n🔇 Disabling controller...")
    model.layers.disable_hook(hook_id)
    
    assert not controller.enabled
    print("✅ Controller disabled")
    
    print("\n🔍 Running inference with controller disabled...")
    output_without_controller = model.inference.execute_inference([test_text])
    
    assert output_without_controller is not None
    print("✅ Inference without controller completed")
    
    # Step 6: Re-enable and verify
    print("\n🔊 Re-enabling controller...")
    model.layers.enable_hook(hook_id)
    
    assert controller.enabled
    print("✅ Controller re-enabled")
    
    # Step 7: Cleanup
    print("\n🧹 Cleaning up...")
    model.layers.unregister_hook(hook_id)
    
    registered_hooks_after = model.layers.get_hooks(LAYER_SIGNATURE)
    assert len(registered_hooks_after) == len(registered_hooks) - 1
    print("✅ Controller unregistered")
    
    print("\n🎉 E2E activation amplification test completed successfully!")


def test_e2e_activation_control_with_controllers_parameter():
    """
    Test the with_controllers parameter for temporary control:
    1. Register a controller
    2. Run inference with with_controllers=False
    3. Verify controller was temporarily disabled
    4. Run with with_controllers=True
    5. Verify controller is active again
    """
    MODEL_ID = "sshleifer/tiny-gpt2"
    LAYER_SIGNATURE = "gpt2lmheadmodel_transformer_h_0_mlp_c_fc"
    DEVICE = "cpu"
    
    print("\n📥 Loading language model...")
    import tempfile
    from pathlib import Path
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / "store")
    model = LanguageModel.from_huggingface(MODEL_ID, store=store)
    model.model.to(DEVICE)
    
    print("\n🎛️ Creating and registering capturing controller...")
    controller = ActivationCapturingController(
        LAYER_SIGNATURE,
        hook_id="test_capturer"
    )
    hook_id = model.layers.register_hook(LAYER_SIGNATURE, controller)
    
    test_text = "The dog played in the park."
    
    # Step 1: Run with controllers disabled temporarily
    print("\n🔍 Running inference with with_controllers=False...")
    controller.clear_captures()
    output_no_controllers = model.inference.execute_inference([test_text], with_controllers=False)
    
    assert output_no_controllers is not None
    # Controller should not have captured anything since it was disabled
    assert len(controller.captured_outputs) == 0, "Controller should not capture when disabled"
    assert controller.enabled, "Controller should still be enabled after temporary disable"
    print("✅ Controller was temporarily disabled")
    
    # Step 2: Run with controllers enabled
    print("\n🔍 Running inference with with_controllers=True...")
    controller.clear_captures()
    output_with_controllers = model.inference.execute_inference([test_text], with_controllers=True)
    
    assert output_with_controllers is not None
    # Controller should have captured activations
    assert len(controller.captured_outputs) > 0, "Controller should capture when enabled"
    print(f"✅ Controller captured {len(controller.captured_outputs)} activation tensors")
    
    # Step 3: Verify modifications were applied
    print("\n🔬 Verifying modifications...")
    assert len(controller.captured_outputs) == len(controller.captured_modified), \
        "Should have same number of captured outputs and modified outputs"
    
    for i, original in enumerate(controller.captured_outputs):
        expected = original * 2.0
        modified = controller.captured_modified[i]
        
        assert torch.allclose(modified, expected, rtol=1e-3, atol=1e-3), \
            f"Modification not applied correctly at index {i}. " \
            f"Max difference: {torch.max(torch.abs(modified - expected)).item():.6f}"
    
    print("✅ Modifications verified")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    model.layers.unregister_hook(hook_id)
    print("✅ Controller unregistered")
    
    print("\n🎉 E2E with_controllers parameter test completed successfully!")


def test_e2e_multiple_controllers():
    """
    Test using multiple controllers on different layers:
    1. Load language model
    2. Register controllers on multiple layers
    3. Run inference with all controllers
    4. Verify all controllers are active
    5. Disable specific controllers
    """
    MODEL_ID = "sshleifer/tiny-gpt2"
    LAYER_1 = "gpt2lmheadmodel_transformer_h_0_mlp_c_fc"
    LAYER_2 = "gpt2lmheadmodel_transformer_h_1_mlp_c_fc"
    DEVICE = "cpu"
    
    print("\n📥 Loading language model...")
    import tempfile
    from pathlib import Path
    temp_dir = tempfile.mkdtemp()
    store = LocalStore(Path(temp_dir) / "store")
    model = LanguageModel.from_huggingface(MODEL_ID, store=store)
    model.model.to(DEVICE)
    
    print("\n🎛️ Creating multiple controllers...")
    controller_1 = SimpleActivationController(LAYER_1, scale_factor=1.5, hook_id="amplify_layer_0")
    controller_2 = SimpleActivationController(LAYER_2, scale_factor=0.5, hook_id="suppress_layer_1")
    
    print("\n🔗 Registering controllers...")
    hook_id_1 = model.layers.register_hook(LAYER_1, controller_1)
    hook_id_2 = model.layers.register_hook(LAYER_2, controller_2)
    
    assert hook_id_1 == "amplify_layer_0"
    assert hook_id_2 == "suppress_layer_1"
    print(f"✅ Registered controllers: {hook_id_1}, {hook_id_2}")
    
    print("\n🔍 Running inference with both controllers...")
    test_text = "The quick brown fox jumps over the lazy dog."
    output = model.inference.execute_inference([test_text])
    
    assert output is not None
    print("✅ Inference with multiple controllers completed")
    
    print("\n🔇 Disabling first controller...")
    model.layers.disable_hook(hook_id_1)
    
    assert not controller_1.enabled
    assert controller_2.enabled
    print("✅ First controller disabled, second still enabled")
    
    print("\n🔍 Running inference with only second controller...")
    output = model.inference.execute_inference([test_text])
    
    assert output is not None
    print("✅ Inference with partial controllers completed")
    
    print("\n🧹 Cleaning up...")
    model.layers.unregister_hook(hook_id_1)
    model.layers.unregister_hook(hook_id_2)
    print("✅ All controllers unregistered")
    
    print("\n🎉 E2E multiple controllers test completed successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

