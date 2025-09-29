from typing import Dict, List

import torch
import pytest

from amber.store import Store, LocalStore


class FakeMemStore(Store):
    def __init__(self):
        # Store each batch as a dict[str, Tensor] under its computed key
        self._runs: Dict[str, Dict[str, torch.Tensor]] = {}

    # Single-tensor helpers (not used in mem tests, but could be added if needed)
    def put_tensor(self, key: str, tensor: torch.Tensor) -> None:  # pragma: no cover - not used in current tests
        # For completeness, store as a 1-item batch under a synthetic run
        self._runs.setdefault("__single__", {})[key] = tensor

    def get_tensor(self, key: str) -> torch.Tensor:  # pragma: no cover - not used in current tests
        return self._runs.get("__single__", {})[key]

    # Run-batch API
    def put_run_batch(self, run_id: str, batch_index: int, tensors: List[torch.Tensor] | Dict[str, torch.Tensor]) -> str:
        if isinstance(tensors, dict):
            to_save = tensors
        else:
            to_save = {f"item_{i}": t for i, t in enumerate(tensors)}
        key = self._run_batch_key(run_id, batch_index)
        self._runs.setdefault(run_id, {})
        # flatten into storage with composite keys
        for name, t in to_save.items():
            self._runs[run_id][f"{key}:{name}"] = t
        return key

    def get_run_batch(self, run_id: str, batch_index: int) -> List[torch.Tensor] | Dict[str, torch.Tensor]:
        key = self._run_batch_key(run_id, batch_index)
        bucket = self._runs.get(run_id, {})
        # collect items for this batch
        items = {k.split(":", 1)[1]: v for k, v in bucket.items() if k.startswith(f"{key}:")}
        names = list(items.keys())
        if names and all(n.startswith("item_") for n in names):
            try:
                pairs = sorted(((int(n.split("_", 1)[1]), v) for n, v in items.items()), key=lambda x: x[0])
                if [i for i, _ in pairs] == list(range(len(pairs))):
                    return [v for _, v in pairs]
            except Exception:
                pass
        return items

    def list_run_batches(self, run_id: str) -> List[int]:
        bucket = self._runs.get(run_id, {})
        seen: set[int] = set()
        for k in bucket.keys():
            if ":" in k:
                prefix = k.split(":", 1)[0]
                name = prefix.split("/")[-1]
                if name.startswith("batch_") and name.endswith(".safetensors"):
                    try:
                        idx = int(name[len("batch_") : len("batch_") + 6])
                        seen.add(idx)
                    except Exception:
                        continue
        return sorted(seen)

    def delete_run(self, run_id: str) -> None:
        self._runs.pop(run_id, None)


@pytest.fixture()
def mem() -> FakeMemStore:
    return FakeMemStore()


@pytest.fixture()
def local(tmp_path) -> LocalStore:
    return LocalStore(tmp_path)


def test_localstore_tensor_roundtrip(local: LocalStore):
    t = torch.randn(3, 4)
    local.put_tensor("tensors/single.safetensors", t)
    got = local.get_tensor("tensors/single.safetensors")
    assert torch.allclose(t, got)


def test_run_batches_mem_store(mem: FakeMemStore):
    # three batches under run_id "runA"
    batches = [[torch.randn(2) for _ in range(3)],
               [torch.randn(2) for _ in range(2)],
               [torch.randn(2) for _ in range(4)]]
    for i, b in enumerate(batches):
        mem.put_run_batch("runA", i, b)
    # listing should be [0,1,2]
    assert mem.list_run_batches("runA") == [0, 1, 2]
    # get by index
    got1 = mem.get_run_batch("runA", 1)
    assert isinstance(got1, list) and len(got1) == 2
    for a, b in zip(batches[1], got1):
        assert torch.allclose(a, b)
    # iterate in order
    for expected, loaded in zip(batches, mem.iter_run_batches("runA")):
        for a, b in zip(expected, loaded):
            assert torch.allclose(a, b)
    # delete run
    mem.delete_run("runA")
    assert mem.list_run_batches("runA") == []


def test_run_batches_local_store(local: LocalStore):
    batches = [[torch.randn(3, 3) for _ in range(2)],
               [torch.randn(3, 3) for _ in range(1)]]
    for i, b in enumerate(batches):
        key = local.put_run_batch("myrun", i, b)
        # ensure files were created under expected path
        assert key.endswith(f"batch_{i:06d}.safetensors")
        assert (local.base_path / key).exists()
    # listing
    assert local.list_run_batches("myrun") == [0, 1]
    # retrieval
    back0 = local.get_run_batch("myrun", 0)
    assert isinstance(back0, list) and len(back0) == 2
    for a, b in zip(batches[0], back0):
        assert torch.allclose(a, b)

from typing import Dict, List

import torch
import pytest

from amber.store import Store, LocalStore


class FakeMemStore(Store):
    def __init__(self):
        # Store each batch as a dict[str, Tensor] under its computed key
        self._runs: Dict[str, Dict[str, torch.Tensor]] = {}

    # Single-tensor helpers (not used in mem tests, but could be added if needed)
    def put_tensor(self, key: str, tensor: torch.Tensor) -> None:  # pragma: no cover - not used in current tests
        # For completeness, store as a 1-item batch under a synthetic run
        self._runs.setdefault("__single__", {})[key] = tensor

    def get_tensor(self, key: str) -> torch.Tensor:  # pragma: no cover - not used in current tests
        return self._runs.get("__single__", {})[key]

    # Run-batch API
    def put_run_batch(self, run_id: str, batch_index: int, tensors: List[torch.Tensor] | Dict[str, torch.Tensor]) -> str:
        if isinstance(tensors, dict):
            to_save = tensors
        else:
            to_save = {f"item_{i}": t for i, t in enumerate(tensors)}
        key = self._run_batch_key(run_id, batch_index)
        self._runs.setdefault(run_id, {})
        # flatten into storage with composite keys
        for name, t in to_save.items():
            self._runs[run_id][f"{key}:{name}"] = t
        return key

    def get_run_batch(self, run_id: str, batch_index: int) -> List[torch.Tensor] | Dict[str, torch.Tensor]:
        key = self._run_batch_key(run_id, batch_index)
        bucket = self._runs.get(run_id, {})
        # collect items for this batch
        items = {k.split(":", 1)[1]: v for k, v in bucket.items() if k.startswith(f"{key}:")}
        names = list(items.keys())
        if names and all(n.startswith("item_") for n in names):
            try:
                pairs = sorted(((int(n.split("_", 1)[1]), v) for n, v in items.items()), key=lambda x: x[0])
                if [i for i, _ in pairs] == list(range(len(pairs))):
                    return [v for _, v in pairs]
            except Exception:
                pass
        return items

    def list_run_batches(self, run_id: str) -> List[int]:
        bucket = self._runs.get(run_id, {})
        seen: set[int] = set()
        for k in bucket.keys():
            if ":" in k:
                prefix = k.split(":", 1)[0]
                name = prefix.split("/")[-1]
                if name.startswith("batch_") and name.endswith(".safetensors"):
                    try:
                        idx = int(name[len("batch_") : len("batch_") + 6])
                        seen.add(idx)
                    except Exception:
                        continue
        return sorted(seen)

    def delete_run(self, run_id: str) -> None:
        self._runs.pop(run_id, None)


@pytest.fixture()
def mem() -> FakeMemStore:
    return FakeMemStore()


@pytest.fixture()
def local(tmp_path) -> LocalStore:
    return LocalStore(tmp_path)


def test_localstore_tensor_roundtrip(local: LocalStore):
    t = torch.randn(3, 4)
    local.put_tensor("tensors/single.safetensors", t)
    got = local.get_tensor("tensors/single.safetensors")
    assert torch.allclose(t, got)


def test_run_batches_mem_store(mem: FakeMemStore):
    # three batches under run_id "runA"
    batches = [[torch.randn(2) for _ in range(3)],
               [torch.randn(2) for _ in range(2)],
               [torch.randn(2) for _ in range(4)]]
    for i, b in enumerate(batches):
        mem.put_run_batch("runA", i, b)
    # listing should be [0,1,2]
    assert mem.list_run_batches("runA") == [0, 1, 2]
    # get by index
    got1 = mem.get_run_batch("runA", 1)
    assert isinstance(got1, list) and len(got1) == 2
    for a, b in zip(batches[1], got1):
        assert torch.allclose(a, b)
    # iterate in order
    for expected, loaded in zip(batches, mem.iter_run_batches("runA")):
        for a, b in zip(expected, loaded):
            assert torch.allclose(a, b)
    # delete run
    mem.delete_run("runA")
    assert mem.list_run_batches("runA") == []


def test_run_batches_local_store(local: LocalStore):
    batches = [[torch.randn(3, 3) for _ in range(2)],
               [torch.randn(3, 3) for _ in range(1)]]
    for i, b in enumerate(batches):
        key = local.put_run_batch("myrun", i, b)
        # ensure files were created under expected path
        assert key.endswith(f"batch_{i:06d}.safetensors")
        assert (local.base_path / key).exists()
    # listing
    assert local.list_run_batches("myrun") == [0, 1]
    # retrieval
    back0 = local.get_run_batch("myrun", 0)
    assert isinstance(back0, list) and len(back0) == 2
    for a, b in zip(batches[0], back0):
        assert torch.allclose(a, b)


def test_iter_run_batch_range_mem(mem: FakeMemStore):
    batches = [[torch.randn(2) for _ in range(1)],
               [torch.randn(2) for _ in range(2)],
               [torch.randn(2) for _ in range(3)]]
    for i, b in enumerate(batches):
        mem.put_run_batch("runB", i, b)
    # range start=1, stop=3 should yield batches 1 and 2
    outs = list(mem.iter_run_batch_range("runB", start=1, stop=3, step=1))
    assert len(outs) == 2
    for exp, got in zip(batches[1:3], outs):
        for a, b in zip(exp, got):
            assert torch.allclose(a, b)
    # step=2 should pick indices 0 and 2
    outs2 = list(mem.iter_run_batch_range("runB", start=0, stop=3, step=2))
    assert len(outs2) == 2
    for exp, got in zip([batches[0], batches[2]], outs2):
        for a, b in zip(exp, got):
            assert torch.allclose(a, b)


def test_iter_run_batch_range_local(local: LocalStore):
    batches = [[torch.randn(3, 3) for _ in range(1)],
               [torch.randn(3, 3) for _ in range(2)],
               [torch.randn(3, 3) for _ in range(1)]]
    for i, b in enumerate(batches):
        local.put_run_batch("range_run", i, b)
    outs = list(local.iter_run_batch_range("range_run", start=1, stop=3, step=1))
    assert len(outs) == 2
    for exp, got in zip(batches[1:3], outs):
        for a, b in zip(exp, got):
            assert torch.allclose(a, b)
    outs2 = list(local.iter_run_batch_range("range_run", start=0, stop=3, step=2))
    assert len(outs2) == 2
    for exp, got in zip([batches[0], batches[2]], outs2):
        for a, b in zip(exp, got):
            assert torch.allclose(a, b)
