from typing import List
from unittest.mock import patch

import pytest
from datasets import Dataset, IterableDataset, IterableDatasetDict

from mi_crow.datasets.loading_strategy import LoadingStrategy
from mi_crow.datasets.text_dataset import TextDataset
from tests.unit.fixtures.stores import create_temp_store


def _build_dataset(texts: List[str], extra_field: str | None = None) -> Dataset:
    data = {"text" if extra_field is None else extra_field: texts}
    if extra_field is not None:
        data["other"] = [f"meta-{i}" for i in range(len(texts))]

    return Dataset.from_dict(data)


def test_init_standard_dataset_keeps_only_text(temp_store):
    ds = _build_dataset(["hello", "world"], extra_field="content")
    text_ds = TextDataset(ds, store=temp_store, text_field="content")
    assert len(text_ds) == 2
    assert text_ds[0] == "hello"
    assert text_ds[1] == "world"


def test_init_missing_column_raises(temp_store):
    ds = Dataset.from_dict({"text": ["a"]})
    with pytest.raises(ValueError, match="must have a 'missing' column"):
        TextDataset(ds, store=temp_store, text_field="missing")


def test_validate_text_field_rejects_blank(temp_store):
    ds = Dataset.from_dict({"text": ["a"]})
    with pytest.raises(ValueError, match="non-empty string"):
        TextDataset(ds, store=temp_store, text_field=" ")


def test_len_iterable_strategy_not_supported(temp_store):
    iterable = IterableDatasetDict({"train": Dataset.from_dict({"text": ["a"]})})["train"]
    text_ds = TextDataset(
        iterable,
        store=temp_store,
        loading_strategy=LoadingStrategy.STREAMING,
    )
    with pytest.raises(NotImplementedError):
        len(text_ds)


def test_getitem_sequence_and_slice(temp_store):
    ds = Dataset.from_dict({"text": ["a", "b", "c", "d"]})
    text_ds = TextDataset(ds, store=temp_store)
    assert text_ds[1] == "b"
    assert text_ds[-1] == "d"
    assert text_ds[1:3] == ["b", "c"]
    assert text_ds[[0, 2, 3]] == ["a", "c", "d"]


def test_getitem_invalid_index_errors(temp_store):
    ds = Dataset.from_dict({"text": ["a"]})
    text_ds = TextDataset(ds, store=temp_store)
    with pytest.raises(IndexError):
        text_ds[5]

    with pytest.raises(TypeError):
        text_ds[1.5]


def test_iter_items_missing_text_field_raises(temp_store):
    ds = Dataset.from_dict({"content": ["a"]})
    text_ds = TextDataset(ds, store=temp_store, text_field="content")
    text_ds._text_field = "missing"
    text_ds._ds = Dataset.from_dict({"other": ["a"]})
    with pytest.raises(ValueError, match="not found in dataset row"):
        list(text_ds.iter_items())


def test_iter_batches_memory_dataset(temp_store):
    ds = Dataset.from_dict({"text": [f"row-{i}" for i in range(5)]})
    text_ds = TextDataset(ds, store=temp_store)
    batches = list(text_ds.iter_batches(batch_size=2))
    assert batches == [["row-0", "row-1"], ["row-2", "row-3"], ["row-4"]]


def test_iter_batches_iterable_dataset(temp_store):
    iterable = IterableDatasetDict({"train": Dataset.from_dict({"text": ["a", "b", "c"]})})["train"]
    text_ds = TextDataset(
        iterable,
        store=temp_store,
        loading_strategy=LoadingStrategy.STREAMING,
    )
    batches = list(text_ds.iter_batches(batch_size=2))
    assert batches == [["a", "b"], ["c"]]


@patch("mi_crow.datasets.text_dataset.load_dataset")
def test_from_huggingface_filters_and_limit(mock_load_dataset, temp_store):
    ds = Dataset.from_dict({"text": ["keep", "drop", "keep"]})
    mock_load_dataset.return_value = ds
    dataset = TextDataset.from_huggingface(
        "repo",
        temp_store,
        filters={"text": "keep"},
        limit=1,
        loading_strategy=LoadingStrategy.MEMORY,
    )
    mock_load_dataset.assert_called_once()
    assert len(dataset) == 1
    assert dataset[0] == "keep"


@patch("mi_crow.datasets.text_dataset.load_dataset", side_effect=RuntimeError("boom"))
def test_from_huggingface_wraps_errors(mock_load_dataset, temp_store):
    with pytest.raises(RuntimeError, match="Failed to load text dataset"):
        TextDataset.from_huggingface("repo", temp_store)


def test_from_csv_and_json(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("text\nhello\nworld\n", encoding="utf-8")
    json_path = tmp_path / "sample.jsonl"
    json_path.write_text('{"text":"hi"}\n{"text":"there"}\n', encoding="utf-8")
    csv_store = create_temp_store(tmp_path, base_path=tmp_path / "csv_store")
    json_store = create_temp_store(tmp_path, base_path=tmp_path / "json_store")
    with patch(
        "mi_crow.datasets.base_dataset.BaseDataset._save_and_load_dataset",
        side_effect=lambda ds, use_memory_mapping=True: ds,
    ):
        csv_ds = TextDataset.from_csv(csv_path, csv_store)
        json_ds = TextDataset.from_json(json_path, json_store)

    assert csv_ds[0] == "hello"
    assert json_ds[1] == "there"


def test_from_local_directory_reads_txt(tmp_path, temp_store):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.txt").write_text("alpha", encoding="utf-8")
    (data_dir / "b.txt").write_text("beta", encoding="utf-8")
    dataset = TextDataset.from_local(data_dir, temp_store)
    assert sorted(list(dataset.iter_items())) == ["alpha", "beta"]


def test_from_local_invalid_path(tmp_path, temp_store):
    with pytest.raises(FileNotFoundError):
        TextDataset.from_local(tmp_path / "missing", temp_store)


def test_from_local_directory_without_txt(tmp_path, temp_store):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "ignored.md").write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="No .txt files found"):
        TextDataset.from_local(data_dir, temp_store)


def test_iter_batches_invalid_batch_size(temp_store):
    ds = Dataset.from_dict({"text": ["a"]})
    text_ds = TextDataset(ds, store=temp_store)
    with pytest.raises(ValueError):
        list(text_ds.iter_batches(0))


def test_from_local_invalid_file_type(tmp_path, temp_store):
    bad_file = tmp_path / "file.md"
    bad_file.write_text("content", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        TextDataset.from_local(bad_file, temp_store)


class TestTextDatasetExtractTexts:
    """Tests for text extraction methods."""

    def test_extract_texts_from_batch_returns_as_is(self, temp_store):
        """Test extract_texts_from_batch returns batch as-is for TextDataset."""
        ds = Dataset.from_dict({"text": ["a", "b", "c"]})
        text_ds = TextDataset(ds, store=temp_store)
        batch = ["text1", "text2", "text3"]
        result = text_ds.extract_texts_from_batch(batch)
        assert result == batch
        assert result is batch

    def test_extract_texts_from_batch_empty_batch(self, temp_store):
        """Test extract_texts_from_batch with empty batch."""
        ds = Dataset.from_dict({"text": ["a"]})
        text_ds = TextDataset(ds, store=temp_store)
        batch = []
        result = text_ds.extract_texts_from_batch(batch)
        assert result == []

    def test_get_all_texts_memory_strategy(self, temp_store):
        """Test get_all_texts with MEMORY strategy."""
        ds = Dataset.from_dict({"text": ["text1", "text2", "text3"]})
        text_ds = TextDataset(ds, store=temp_store)
        texts = text_ds.get_all_texts()
        assert texts == ["text1", "text2", "text3"]
        assert isinstance(texts, list)

    def test_get_all_texts_iterable_only(self, temp_store):
        """Test get_all_texts with STREAMING strategy."""
        iterable = IterableDatasetDict({"train": Dataset.from_dict({"text": ["a", "b"]})})["train"]
        text_ds = TextDataset(
            iterable,
            store=temp_store,
            loading_strategy=LoadingStrategy.STREAMING,
        )
        texts = text_ds.get_all_texts()
        assert texts == ["a", "b"]


class TestTextDatasetFromDisk:
    """Tests for from_disk factory method."""

    def test_from_disk_success(self, temp_store, tmp_path):
        """Test from_disk loads dataset successfully."""
        from datasets import Dataset

        ds = Dataset.from_dict({"text": ["text1", "text2", "text3"]})
        dataset_dir = tmp_path / "datasets"
        dataset_dir.mkdir(parents=True)
        ds.save_to_disk(str(dataset_dir))
        temp_store.base_path = tmp_path
        loaded_ds = TextDataset.from_disk(temp_store)
        assert len(loaded_ds) == 3
        assert loaded_ds[0] == "text1"

    def test_from_disk_none_store_raises(self, tmp_path):
        """Test from_disk raises error when store is None."""
        with pytest.raises(ValueError, match="store cannot be None"):
            TextDataset.from_disk(None)

    def test_from_disk_streaming_strategy_raises(self, temp_store):
        """Test from_disk raises error for STREAMING strategy."""
        with pytest.raises(ValueError, match="STREAMING loading strategy not supported"):
            TextDataset.from_disk(temp_store, loading_strategy=LoadingStrategy.STREAMING)

    def test_from_disk_missing_directory_raises(self, temp_store, tmp_path):
        """Test from_disk raises error when dataset directory doesn't exist."""
        temp_store.base_path = tmp_path
        temp_store.dataset_prefix = "datasets"
        with pytest.raises(FileNotFoundError, match="Dataset directory not found"):
            TextDataset.from_disk(temp_store)

    def test_from_disk_no_arrow_files_raises(self, temp_store, tmp_path):
        """Test from_disk raises error when no Arrow files found."""
        from pathlib import Path

        dataset_dir = tmp_path / "datasets"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "not_arrow.txt").write_text("test")
        temp_store.base_path = tmp_path
        with pytest.raises(FileNotFoundError, match="No Arrow files found"):
            TextDataset.from_disk(temp_store)

    def test_from_disk_load_error_raises_runtime_error(self, temp_store, tmp_path):
        """Test from_disk raises RuntimeError when load_from_disk fails."""
        from pathlib import Path
        from unittest.mock import patch

        dataset_dir = tmp_path / "datasets"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "data.arrow").touch()
        temp_store.base_path = tmp_path
        with patch("mi_crow.datasets.text_dataset.load_from_disk", side_effect=Exception("Load error")):
            with pytest.raises(RuntimeError, match="Failed to load dataset"):
                TextDataset.from_disk(temp_store)

    def test_from_disk_with_custom_text_field(self, temp_store, tmp_path):
        """Test from_disk with custom text field."""
        from datasets import Dataset

        ds = Dataset.from_dict({"text": ["text1", "text2"]})
        dataset_dir = tmp_path / "datasets"
        dataset_dir.mkdir(parents=True)
        ds.save_to_disk(str(dataset_dir))
        temp_store.base_path = tmp_path
        loaded_ds = TextDataset.from_disk(temp_store, text_field="text")
        assert len(loaded_ds) == 2


class TestTextDatasetFromHuggingfaceStreaming:
    """Tests for from_huggingface with streaming."""

    def test_from_huggingface_streaming_with_filters_raises(self, temp_store):
        """Test from_huggingface raises error when streaming with filters."""
        from unittest.mock import patch

        with patch("mi_crow.datasets.text_dataset.load_dataset") as mock_load:
            mock_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
            mock_load.return_value = mock_ds
            with pytest.raises(
                (NotImplementedError, RuntimeError), match="filters and limit are not supported when streaming"
            ):
                TextDataset.from_huggingface("test/dataset", temp_store, streaming=True, filters={"key": "value"})

    def test_from_huggingface_streaming_with_limit_raises(self, temp_store):
        """Test from_huggingface raises error when streaming with limit."""
        from unittest.mock import patch

        with patch("mi_crow.datasets.text_dataset.load_dataset") as mock_load:
            mock_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
            mock_load.return_value = mock_ds
            with pytest.raises(
                (NotImplementedError, RuntimeError), match="filters and limit are not supported when streaming"
            ):
                TextDataset.from_huggingface("test/dataset", temp_store, streaming=True, limit=10)

    def test_from_huggingface_streaming_with_stratify_raises(self, temp_store):
        """Test from_huggingface raises error when streaming with stratify."""
        from unittest.mock import patch

        with patch("mi_crow.datasets.text_dataset.load_dataset") as mock_load:
            mock_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
            mock_load.return_value = mock_ds
            with pytest.raises(NotImplementedError, match="Stratification and drop_na are not supported for streaming"):
                TextDataset.from_huggingface("test/dataset", temp_store, streaming=True, stratify_by="key")

    def test_from_huggingface_streaming_with_drop_na_raises(self, temp_store):
        """Test from_huggingface raises error when streaming with drop_na."""
        from unittest.mock import patch

        with patch("mi_crow.datasets.text_dataset.load_dataset") as mock_load:
            mock_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
            mock_load.return_value = mock_ds
            with pytest.raises(NotImplementedError, match="Stratification and drop_na are not supported for streaming"):
                TextDataset.from_huggingface("test/dataset", temp_store, streaming=True, drop_na=True)

    def test_from_huggingface_load_error_raises_runtime_error(self, temp_store):
        """Test from_huggingface raises RuntimeError when load_dataset fails."""
        from unittest.mock import patch

        with patch("mi_crow.datasets.text_dataset.load_dataset", side_effect=Exception("Load error")):
            with pytest.raises(RuntimeError, match="Failed to load text dataset from HuggingFace Hub"):
                TextDataset.from_huggingface("test/dataset", temp_store)

    def test_extract_texts_from_batch_integration_with_iter_batches(self, temp_store):
        """Test extract_texts_from_batch works with iter_batches output."""
        ds = Dataset.from_dict({"text": ["t1", "t2", "t3", "t4"]})
        text_ds = TextDataset(ds, store=temp_store)
        for batch in text_ds.iter_batches(batch_size=2):
            extracted = text_ds.extract_texts_from_batch(batch)
            assert extracted == batch
            assert all(isinstance(text, str) for text in extracted)
