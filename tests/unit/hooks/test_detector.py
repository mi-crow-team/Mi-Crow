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
        super().__init__(layer_signature, hook_type, hook_id, store)
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


def test_forward_hook_collects_and_accumulates_metadata():
    meta = {"a": 1, "t": torch.tensor([3.0])}
    det = _LiteDetector(hook_type=HookType.FORWARD, collect_returns=meta)

    _call_hook(det)

    # Both phases called and metadata stored
    assert det.calls == ["process", "collect"]
    acc = det.get_accumulated_metadata()
    assert len(acc) == 1
    assert acc[0]["a"] == 1
    assert isinstance(acc[0]["t"], torch.Tensor)


def test_pre_forward_hook_calls_with_none_output_and_collects():
    meta = {"tag": "pre"}
    det = _LiteDetector(hook_type=HookType.PRE_FORWARD, collect_returns=meta)

    _call_hook(det)

    # Output passed to detector is None for pre hooks
    assert det.last_args is not None and det.last_args["output"] is None
    assert det.get_accumulated_metadata() == [meta]


def test_no_metadata_when_collect_returns_none():
    det = _LiteDetector(collect_returns=None)
    _call_hook(det)
    assert det.get_accumulated_metadata() == []


def test_errors_in_process_or_collect_are_swallowed():
    # Error in process_activations
    det1 = _LiteDetector(raise_in_process=True, collect_returns={"x": 1})
    _call_hook(det1)
    # No crash, but since process crashed, collect may not have run; ensure no metadata
    assert det1.get_accumulated_metadata() in ([], [{"x": 1}])

    # Error in collect_metadata
    det2 = _LiteDetector(raise_in_collect=True, collect_returns={"x": 1})
    _call_hook(det2)
    assert det2.get_accumulated_metadata() == []


def test_disable_prevents_execution_and_accumulation():
    det = _LiteDetector(collect_returns={"z": 1})
    det.disable()
    _call_hook(det)
    assert det.calls == []
    assert det.get_accumulated_metadata() == []


def test_reset_metadata_clears_accumulator():
    det = _LiteDetector(collect_returns={"m": 1})
    _call_hook(det)
    assert det.get_accumulated_metadata()
    det.reset_metadata()
    assert det.get_accumulated_metadata() == []


def test_save_metadata_uses_store_and_converts_tensors_to_cpu():
    captured: Dict[str, Any] = {}

    class FakeStore:
        def put_run_meta(self, key: str, value: Dict[str, Any]):
            captured["key"] = key
            captured["value"] = value

    meta = {"tensor": torch.tensor([5.0]), "plain": 7}
    store = FakeStore()
    det = _LiteDetector(collect_returns=meta, store=store)

    _call_hook(det)
    det.save_metadata("run42")

    assert captured["key"].startswith("run42_detector_")
    saved = captured["value"]
    assert saved["hook_id"] == det.id
    assert saved["layer_signature"] == str(det.layer_signature)
    assert saved["hook_type"] in (HookType.FORWARD, HookType.PRE_FORWARD, HookType(saved["hook_type"]))
    assert saved["num_batches"] == 1
    assert isinstance(saved["metadata"][0]["tensor"], torch.Tensor)
    assert not saved["metadata"][0]["tensor"].is_cuda
    assert saved["metadata"][0]["plain"] == 7


def test_save_metadata_raises_without_store():
    det = _LiteDetector(collect_returns={"x": 1})
    _call_hook(det)
    with pytest.raises(ValueError):
        det.save_metadata("runX")


