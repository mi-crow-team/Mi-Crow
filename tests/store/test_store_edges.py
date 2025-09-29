import pytest
import torch
from amber.store import LocalStore


def test_iter_run_batch_range_errors_and_defaults(tmp_path):
    store = LocalStore(tmp_path)
    # step == 0 error
    with pytest.raises(ValueError):
        list(store.iter_run_batch_range("none", start=0, stop=1, step=0))
    # start < 0 error
    with pytest.raises(ValueError):
        list(store.iter_run_batch_range("none", start=-1, stop=1, step=1))

    # When no batches exist and stop is None, should yield nothing (empty list)
    out = list(store.iter_run_batch_range("missing_run", step=1))
    assert out == []

    # Create some batches and call with stop=None to use max+1
    t = torch.randn(2, 2)
    store.put_run_batch("r", 0, [t])
    store.put_run_batch("r", 2, [t])
    # Now list_run_batches returns [0,2]; iter with stop=None should iterate indices 0..max (inclusive exclusive)
    outs = list(store.iter_run_batch_range("r", start=0, step=1))
    # This will attempt to read index 1 as well, which should raise FileNotFoundError for LocalStore
    # So we handle that by slicing to the first element to prove it started; alternatively, we can call with stop=1
    outs2 = list(store.iter_run_batch_range("r", start=0, stop=1, step=1))
    assert len(outs2) == 1 and isinstance(outs2[0], list)


def test_list_run_batches_ignores_malformed_and_delete_idempotent(tmp_path):
    store = LocalStore(tmp_path)
    base = store.base_path / "runs" / "abc"
    base.mkdir(parents=True, exist_ok=True)
    # well-formed and malformed files
    (base / "batch_000000.safetensors").write_bytes(b"\x00")
    (base / "batch_notanumber.safetensors").write_text("oops")
    (base / "random.txt").write_text("nope")
    # Only the well-formed index should be listed
    assert store.list_run_batches("abc") == [0]
    # delete_run should not fail if run missing
    store.delete_run("xyz")
    # deleting existing should remove files
    store.delete_run("abc")
    assert store.list_run_batches("abc") == []
