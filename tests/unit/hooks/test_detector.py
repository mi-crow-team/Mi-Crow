import types
from typing import Any, Dict, List

import pytest
import torch

from amber.hooks import Detector, HookType


class _LiteDetector(Detector):
    """
    Minimal Detector implementation for testing.
    Allows injecting behavior to track calls and returned metadata.
    """

    def __init__(
        self,
        layer_signature: str | int = "layer0",
        hook_type: HookType | str = HookType.FORWARD,
        hook_id: str | None = "test-detector",
        store: Any | None = None,
        *,
        collect_returns: Dict[str, Any] | None = None,
        raise_in_process: bool = False,
        raise_in_collect: bool = False,
    ):
        super().__init__(hook_type=hook_type, hook_id=hook_id, store=store, layer_signature=layer_signature)
        self.calls: List[str] = []
        self.last_args: Dict[str, Any] | None = None
        self._collect_returns = collect_returns
        self._raise_in_process = raise_in_process
        self._raise_in_collect = raise_in_collect

    def process_activations(self, module: Any, inputs: tuple, output: Any) -> None:
        self.calls.append("process")
        self.last_args = {"module": module, "inputs": inputs, "output": output}
        if self._raise_in_process:
            raise RuntimeError("process crash")

    def collect_metadata(self, module: Any, inputs: tuple, output: Any) -> Dict[str, Any] | None:
        self.calls.append("collect")
        if self._raise_in_collect:
            raise RuntimeError("collect crash")
        return self._collect_returns


class _DummyModule:
    pass


def _call_hook(det: Detector, *, inputs=(torch.tensor([1.0]),), output=torch.tensor([2.0])):
    """Utility: call the torch hook wrapper respecting hook type."""
    torch_hook = det.get_torch_hook()
    module = _DummyModule()
    if det.hook_type == HookType.PRE_FORWARD:
        return torch_hook(module, inputs)
    return torch_hook(module, inputs, output)


def test_forward_hook_calls_process_activations():
    """Test that forward hook calls process_activations."""
    det = _LiteDetector(hook_type=HookType.FORWARD)

    _call_hook(det)

    # process_activations should be called
    assert det.calls == ["process"]
    assert det.last_args is not None
    assert "output" in det.last_args


def test_pre_forward_hook_calls_with_none_output():
    """Test that pre_forward hook calls process_activations with None output."""
    det = _LiteDetector(hook_type=HookType.PRE_FORWARD)

    _call_hook(det)

    # Output passed to detector is None for pre hooks
    assert det.last_args is not None and det.last_args["output"] is None
    assert det.calls == ["process"]


def test_disable_prevents_execution():
    """Test that disable prevents hook execution."""
    det = _LiteDetector()
    det.disable()
    _call_hook(det)
    assert det.calls == []


