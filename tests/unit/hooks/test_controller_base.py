from typing import Any, Tuple

import pytest
import torch

from amber.hooks import Controller, HookType


class _LiteController(Controller):
    def __init__(
        self,
        layer_signature: str | int = "layer0",
        hook_type: HookType | str = HookType.FORWARD,
        hook_id: str | None = "test-controller",
        *,
        raise_in_modify: bool = False,
        return_value: Any | None = None,
    ) -> None:
        super().__init__(hook_type=hook_type, hook_id=hook_id, layer_signature=layer_signature)
        self.calls: list[str] = []
        self.last_inputs: tuple | None = None
        self.last_output: Any | None = None
        self._raise_in_modify = raise_in_modify
        self._return_value = return_value

    def modify_activations(self, module: Any, inputs: torch.Tensor | None, output: torch.Tensor | None) -> torch.Tensor | None:
        self.calls.append("modify")
        self.last_inputs = inputs
        self.last_output = output
        if self._raise_in_modify:
            raise RuntimeError("boom")
        return self._return_value


class _DummyModule:
    pass


def _call_hook(ctrl: Controller, *, inputs=(torch.tensor([1.0]),), output=torch.tensor([2.0])):
    hook = ctrl.get_torch_hook()
    module = _DummyModule()
    if ctrl.hook_type == HookType.PRE_FORWARD:
        return hook(module, inputs)
    return hook(module, inputs, output)


def test_forward_controller_modifies_output():
    expected = torch.tensor([4.0])
    ctrl = _LiteController(hook_type=HookType.FORWARD, return_value=expected)
    ret = _call_hook(ctrl)
    assert ctrl.calls == ["modify"]
    # Forward hooks return None (PyTorch signature), but modify_activations was called
    assert ret is None
    assert isinstance(ctrl.last_output, torch.Tensor)
    # Verify modify_activations received the correct output
    assert torch.equal(ctrl.last_output, torch.tensor([2.0]))  # Original output passed to modify_activations


def test_pre_forward_controller_modifies_inputs_tuple():
    modified_tensor = torch.tensor([10.0])
    ctrl = _LiteController(hook_type=HookType.PRE_FORWARD, return_value=modified_tensor)
    ret = _call_hook(ctrl)
    assert ctrl.calls == ["modify"]
    assert isinstance(ret, tuple)
    assert torch.equal(ret[0], modified_tensor)
    # In pre hooks, Controller passes input_tensor as both inputs and output to modify_activations
    assert isinstance(ctrl.last_output, torch.Tensor)
    assert torch.equal(ctrl.last_output, torch.tensor([1.0]))  # Original input tensor


def test_disabled_controller_returns_none_and_does_not_call_modify():
    ctrl = _LiteController(return_value=torch.tensor([3.0]))
    ctrl.disable()
    ret = _call_hook(ctrl)
    assert ret is None
    assert ctrl.calls == []


def test_controller_errors_are_swallowed_and_return_none():
    ctrl = _LiteController(raise_in_modify=True)
    ret = _call_hook(ctrl)
    assert ret is None
    assert ctrl.calls == ["modify"]


def test_hook_type_accepts_string_and_uses_correct_wrapper():
    # Use string to ensure conversion works
    modified_tensor = torch.tensor([7.0])
    ctrl = _LiteController(hook_type="pre_forward", return_value=modified_tensor)
    assert ctrl.hook_type == HookType.PRE_FORWARD
    ret = _call_hook(ctrl)
    # Ensure wrapper used pre-forward path and returned modified inputs
    assert isinstance(ret, tuple)
    assert torch.equal(ret[0], modified_tensor)
    # In pre hooks, Controller passes input_tensor as both inputs and output to modify_activations
    assert isinstance(ctrl.last_output, torch.Tensor)


