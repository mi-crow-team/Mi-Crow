import io
import json
from pathlib import Path

import pytest
from datasets import Dataset

from amber.adapters.text_snippet_dataset import TextSnippetDataset


# ---------- Helpers ----------

def _make_base_ds(n=5):
    return Dataset.from_dict({"text": [f"row-{i}" for i in range(n)]})


# ---------- Core API tests using in-memory dataset persisted to tmp cache ----------

def test_core_length_and_indexing_and_iteration(tmp_path):
    ds = _make_base_ds(10)
    snd = TextSnippetDataset(ds, cache_dir=tmp_path)

    # __len__
    assert len(snd) == 10

    # int index
    assert snd[0] == "row-0"

    # slice contiguous
    assert snd[2:5] == ["row-2", "row-3", "row-4"]

    # slice with step (non-unit)
    assert snd[1:7:2] == ["row-1", "row-3", "row-5"]

    # sequence of indices
    assert snd[[0, 3, 9]] == ["row-0", "row-3", "row-9"]

    # get_batch normal
    assert snd.get_batch(3, 4) == ["row-3", "row-4", "row-5", "row-6"]
    # get_batch edge: zero size
    assert snd.get_batch(0, 0) == []
    # get_batch edge: start beyond end
    assert snd.get_batch(100, 4) == []

    # get_batch_by_indices
    assert snd.get_batch_by_indices([2, 1, 2, 0]) == ["row-2", "row-1", "row-2", "row-0"]
    # empty indices
    assert snd.get_batch_by_indices([]) == []

    # iter_items
    assert list(snd.iter_items())[:4] == ["row-0", "row-1", "row-2", "row-3"]

    # iter_batches
    batches = list(snd.iter_batches(batch_size=3))
    flat = [x for b in batches for x in b]
    assert flat[:6] == ["row-0", "row-1", "row-2", "row-3", "row-4", "row-5"]

    # iter_batches invalid
    with pytest.raises(ValueError):
        _ = list(snd.iter_batches(batch_size=0))

    # head
    assert snd.head(3) == ["row-0", "row-1", "row-2"]


def test_init_requires_text_column(tmp_path):
    bad = Dataset.from_dict({"content": ["a", "b"]})
    with pytest.raises(ValueError):
        TextSnippetDataset(bad, cache_dir=tmp_path)


# ---------- from_local tests ----------

def test_from_local_directory_txt_recursive(tmp_path):
    # Create nested structure with .txt files
    root = tmp_path / "texts"
    sub = root / "a" / "b"
    sub.mkdir(parents=True)
    (root / "x.txt").write_text("hello", encoding="utf-8")
    (sub / "y.txt").write_text("world", encoding="utf-8")
    (sub / "ignore.md").write_text("nope", encoding="utf-8")

    snd = TextSnippetDataset.from_local(root, cache_dir=tmp_path / "cache")
    texts = list(snd.iter_items())
    # Order is sorted by path according to implementation
    assert set(texts) == {"hello", "world"}
    assert len(snd) == 2


def test_from_local_directory_txt_non_recursive(tmp_path):
    root = tmp_path / "texts"
    sub = root / "sub"
    sub.mkdir(parents=True)
    (root / "a.txt").write_text("A", encoding="utf-8")
    (sub / "b.txt").write_text("B", encoding="utf-8")

    snd = TextSnippetDataset.from_local(root, cache_dir=tmp_path / "cache2", recursive=False)
    assert list(snd.iter_items()) == ["A"]


@pytest.mark.parametrize(
    "ext,writer,kwargs,expected",
    [
        (".jsonl", "jsonl", {"records": [{"content": "c1"}, {"content": "c2"}]}, ["c1", "c2"]),
        (".csv", "csv", {"header": "text", "rows": [["t1"], ["t2"]]}, ["t1", "t2"]),
        (".tsv", "tsv", {"header": "text", "rows": [["u1"], ["u2"]]}, ["u1", "u2"]),
    ],
)
def test_from_local_structured_files(tmp_path, ext, writer, kwargs, expected):
    p = tmp_path / f"data{ext}"
    if writer == "jsonl":
        with p.open("w", encoding="utf-8") as f:
            for rec in kwargs["records"]:
                f.write(json.dumps(rec) + "\n")
        # content column -> rename via text_field
        snd = TextSnippetDataset.from_local(p, cache_dir=tmp_path / "cjsonl", text_field="content")
    elif writer == "csv":
        # minimal CSV
        lines = [kwargs["header"], *[row[0] for row in kwargs["rows"]]]
        p.write_text("\n".join(lines), encoding="utf-8")
        snd = TextSnippetDataset.from_local(p, cache_dir=tmp_path / "ccsv")
    else:  # tsv
        lines = [kwargs["header"], *[row[0] for row in kwargs["rows"]]]
        p.write_text("\n".join(lines), encoding="utf-8")
        snd = TextSnippetDataset.from_local(p, cache_dir=tmp_path / "ctsv")

    assert list(snd.iter_items()) == expected


def test_from_local_errors(tmp_path):
    # nonexistent path
    with pytest.raises(FileNotFoundError):
        TextSnippetDataset.from_local(tmp_path / "nope", cache_dir=tmp_path / "x")

    # unsupported suffix
    p = tmp_path / "file.bad"
    p.write_text("data", encoding="utf-8")
    with pytest.raises(ValueError):
        TextSnippetDataset.from_local(p, cache_dir=tmp_path / "y")


# ---------- from_hf tests (monkeypatch load_dataset to avoid network) ----------

def test_from_hf_basic_and_filters_and_limit(monkeypatch, tmp_path):
    base = Dataset.from_dict({
        "text": ["a", "b", "c", "d"],
        "lang": ["en", "pl", "en", "pl"],
    })

    def fake_load_dataset(path, split, revision=None):  # signature aligned with usage
        assert path == "some/repo"
        assert split == "train"
        return base

    monkeypatch.setattr("amber.adapters.text_snippet_dataset.load_dataset", fake_load_dataset)

    # No filters, limit 2
    ds1 = TextSnippetDataset.from_huggingface("some/repo", split="train", cache_dir=tmp_path / "h1", limit=2)
    assert list(ds1.iter_items()) == ["a", "b"]

    # Filters keep only lang == 'pl'
    ds2 = TextSnippetDataset.from_huggingface(
        "some/repo",
        split="train",
        cache_dir=tmp_path / "h2",
        filters={"lang": "pl"},
    )
    assert list(ds2.iter_items()) == ["b", "d"]


def test_from_hf_rename_and_missing_field_error(monkeypatch, tmp_path):
    base = Dataset.from_dict({"content": ["x", "y"]})

    def fake_load_dataset(path, split, revision=None):
        return base

    monkeypatch.setattr("amber.adapters.text_snippet_dataset.load_dataset", fake_load_dataset)

    # Rename content -> text
    ds = TextSnippetDataset.from_huggingface("repo", cache_dir=tmp_path / "h3", text_field="content")
    assert list(ds.iter_items()) == ["x", "y"]

    # Missing field should raise
    base2 = Dataset.from_dict({"other": ["q"]})

    def fake2(path, split, revision=None):
        return base2

    monkeypatch.setattr("amber.adapters.text_snippet_dataset.load_dataset", fake2)
    with pytest.raises(ValueError):
        TextSnippetDataset.from_huggingface("repo", cache_dir=tmp_path / "h4", text_field="content")


def test_duck_typed_cache_dir_accepts_base_path(tmp_path: Path):
    # Prepare a simple dataset with a text column
    base = Dataset.from_dict({"text": ["x", "y", "z"]})

    class Duck:
        def __init__(self, base_path):
            self.base_path = base_path

    duck = Duck(tmp_path / "cache_duck")
    dset = TextSnippetDataset(base, cache_dir=duck)
    assert len(dset) == 3
    # head to ensure it was reloaded correctly from disk
    assert dset.head(2) == ["x", "y"]


def test_from_local_raises_when_missing_text_field(tmp_path: Path):
    # Create JSONL with a different field than requested
    data_path = tmp_path / "data.jsonl"
    with data_path.open("w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps({"body": f"t{i}"}) + "\n")

    with pytest.raises(ValueError):
        # text_field defaults to "text", which does not exist in the file
        TextSnippetDataset.from_local(data_path, cache_dir=tmp_path / "cache", text_field="text")
