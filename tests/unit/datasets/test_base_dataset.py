"""Tests for BaseDataset abstract class."""

import tempfile
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset, IterableDataset

from amber.datasets.base_dataset import BaseDataset
from amber.datasets.loading_strategy import LoadingStrategy


class ConcreteBaseDataset(BaseDataset):
    """Concrete implementation of BaseDataset for testing."""

    def __len__(self) -> int:
        if self._loading_strategy == LoadingStrategy.STREAMING:
            raise NotImplementedError("len() not supported for STREAMING datasets")
        return self._ds.num_rows

    def __getitem__(self, idx):
        if self._loading_strategy == LoadingStrategy.STREAMING:
            raise NotImplementedError("Indexing not supported for STREAMING datasets")
        if isinstance(idx, int):
            return self._ds[idx]
        elif isinstance(idx, slice):
            # Use select to get a proper dataset view, then convert to list
            start, stop, step = idx.indices(len(self))
            if step != 1:
                indices = list(range(start, stop, step))
                selected = self._ds.select(indices)
            else:
                selected = self._ds.select(range(start, stop))
            # Convert dataset to list of rows
            return [selected[i] for i in range(len(selected))]
        else:
            selected = self._ds.select(list(idx))
            return [selected[i] for i in range(len(selected))]

    def iter_items(self):
        for item in self._ds:
            yield item

    def iter_batches(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got: {batch_size}")
        for i in range(0, len(self), batch_size):
            end = min(i + batch_size, len(self))
            yield list(self[i:end])

    def extract_texts_from_batch(self, batch):
        # Dummy implementation for testing
        return [item["text"] for item in batch if "text" in item]

    def get_all_texts(self):
        # Dummy implementation for testing
        return [item["text"] for item in self._ds]


class TestBaseDatasetInitialization:
    """Tests for BaseDataset initialization."""

    def test_init_with_memory_strategy(self, temp_store):
        """Test initialization with MEMORY strategy."""
        ds = Dataset.from_dict({"text": ["a", "b", "c"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.MEMORY)
        assert dataset._loading_strategy == LoadingStrategy.MEMORY
        assert not dataset._is_iterable
        assert len(dataset) == 3

    def test_init_with_dynamic_load_strategy(self, temp_store):
        """Test initialization with DISK strategy."""
        ds = Dataset.from_dict({"text": ["a", "b", "c"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.DISK)
        assert dataset._loading_strategy == LoadingStrategy.DISK
        assert not dataset._is_iterable
        assert len(dataset) == 3

    def test_init_with_iterable_only_strategy(self, temp_store):
        """Test initialization with STREAMING strategy."""
        iter_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
        dataset = ConcreteBaseDataset(iter_ds, temp_store, LoadingStrategy.STREAMING)
        assert dataset._loading_strategy == LoadingStrategy.STREAMING
        assert dataset._is_iterable

    def test_init_with_none_store_raises_error(self):
        """Test that None store raises ValueError."""
        ds = Dataset.from_dict({"text": ["a"]})
        with pytest.raises(ValueError, match="store cannot be None"):
            ConcreteBaseDataset(ds, None, LoadingStrategy.MEMORY)

    def test_init_with_invalid_loading_strategy_raises_error(self, temp_store):
        """Test that invalid loading strategy raises ValueError."""
        ds = Dataset.from_dict({"text": ["a"]})
        with pytest.raises(ValueError, match="loading_strategy must be a LoadingStrategy"):
            ConcreteBaseDataset(ds, temp_store, "invalid")

    def test_init_converts_iterable_to_dataset_for_memory(self, temp_store):
        """Test that IterableDataset is converted to Dataset for MEMORY strategy."""
        iter_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
        dataset = ConcreteBaseDataset(iter_ds, temp_store, LoadingStrategy.MEMORY)
        assert not dataset._is_iterable
        assert len(dataset) == 1

    def test_init_converts_iterable_to_dataset_for_dynamic_load(self, temp_store):
        """Test that IterableDataset is converted to Dataset for DISK strategy."""
        iter_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
        dataset = ConcreteBaseDataset(iter_ds, temp_store, LoadingStrategy.DISK)
        assert not dataset._is_iterable
        assert len(dataset) == 1

    def test_init_converts_dataset_to_iterable_for_iterable_only(self, temp_store):
        """Test that Dataset is converted to IterableDataset for STREAMING strategy."""
        ds = Dataset.from_dict({"text": ["a", "b"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.STREAMING)
        assert dataset._is_iterable


class TestBaseDatasetProperties:
    """Tests for BaseDataset properties."""

    def test_is_streaming_memory_false(self, temp_store):
        """Test is_streaming property for MEMORY strategy."""
        ds = Dataset.from_dict({"text": ["a"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.MEMORY)
        assert not dataset.is_streaming

    def test_is_streaming_dynamic_load_true(self, temp_store):
        """Test is_streaming property for DISK strategy."""
        ds = Dataset.from_dict({"text": ["a"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.DISK)
        assert dataset.is_streaming

    def test_is_streaming_iterable_only_true(self, temp_store):
        """Test is_streaming property for STREAMING strategy."""
        iter_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
        dataset = ConcreteBaseDataset(iter_ds, temp_store, LoadingStrategy.STREAMING)
        assert dataset.is_streaming


class TestBaseDatasetGetBatch:
    """Tests for get_batch method."""

    def test_get_batch_memory_strategy(self, temp_store):
        """Test get_batch with MEMORY strategy."""
        ds = Dataset.from_dict({"text": ["a", "b", "c", "d", "e"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.MEMORY)
        batch = dataset.get_batch(0, 3)
        assert len(batch) == 3

    def test_get_batch_dynamic_load_strategy(self, temp_store):
        """Test get_batch with DISK strategy."""
        ds = Dataset.from_dict({"text": ["a", "b", "c", "d", "e"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.DISK)
        batch = dataset.get_batch(1, 2)
        assert len(batch) == 2

    def test_get_batch_iterable_only_raises_error(self, temp_store):
        """Test that get_batch raises NotImplementedError for STREAMING."""
        iter_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
        dataset = ConcreteBaseDataset(iter_ds, temp_store, LoadingStrategy.STREAMING)
        with pytest.raises(NotImplementedError, match="get_batch not supported"):
            dataset.get_batch(0, 1)

    def test_get_batch_invalid_batch_size_returns_empty(self, temp_store):
        """Test get_batch with invalid batch_size returns empty list."""
        ds = Dataset.from_dict({"text": ["a", "b"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.MEMORY)
        batch = dataset.get_batch(0, 0)
        assert batch == []

    def test_get_batch_start_out_of_bounds_returns_empty(self, temp_store):
        """Test get_batch with start out of bounds returns empty list."""
        ds = Dataset.from_dict({"text": ["a", "b"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.MEMORY)
        batch = dataset.get_batch(10, 5)
        assert batch == []

    def test_get_batch_partial_batch(self, temp_store):
        """Test get_batch returns partial batch when reaching end."""
        ds = Dataset.from_dict({"text": ["a", "b", "c"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.MEMORY)
        batch = dataset.get_batch(2, 5)
        assert len(batch) == 1


class TestBaseDatasetHead:
    """Tests for head method."""

    def test_head_memory_strategy(self, temp_store):
        """Test head with MEMORY strategy."""
        ds = Dataset.from_dict({"text": ["a", "b", "c", "d", "e"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.MEMORY)
        items = dataset.head(3)
        assert len(items) == 3

    def test_head_iterable_only_strategy(self, temp_store):
        """Test head with STREAMING strategy."""
        iter_ds = IterableDataset.from_generator(lambda: iter([{"text": f"text_{i}"} for i in range(10)]))
        dataset = ConcreteBaseDataset(iter_ds, temp_store, LoadingStrategy.STREAMING)
        items = dataset.head(3)
        assert len(items) == 3

    def test_head_default_n(self, temp_store):
        """Test head with default n=5."""
        ds = Dataset.from_dict({"text": [f"text_{i}" for i in range(10)]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.MEMORY)
        items = dataset.head()
        assert len(items) == 5

    def test_head_n_larger_than_dataset(self, temp_store):
        """Test head with n larger than dataset size."""
        ds = Dataset.from_dict({"text": ["a", "b"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.MEMORY)
        items = dataset.head(10)
        assert len(items) == 2


class TestBaseDatasetSample:
    """Tests for sample method."""

    def test_sample_memory_strategy(self, temp_store):
        """Test sample with MEMORY strategy."""
        ds = Dataset.from_dict({"text": ["a", "b", "c", "d", "e"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.MEMORY)
        items = dataset.sample(3)
        assert len(items) == 3
        # All items should be from the dataset
        all_items = list(dataset.iter_items())
        assert all(item in all_items for item in items)

    def test_sample_dynamic_load_strategy(self, temp_store):
        """Test sample with DISK strategy."""
        ds = Dataset.from_dict({"text": ["a", "b", "c", "d", "e"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.DISK)
        items = dataset.sample(2)
        assert len(items) == 2

    def test_sample_iterable_only_raises_error(self, temp_store):
        """Test that sample raises NotImplementedError for STREAMING."""
        iter_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
        dataset = ConcreteBaseDataset(iter_ds, temp_store, LoadingStrategy.STREAMING)
        with pytest.raises(NotImplementedError, match="sample\\(\\) not supported"):
            dataset.sample(1)

    def test_sample_n_zero_returns_empty(self, temp_store):
        """Test sample with n=0 returns empty list."""
        ds = Dataset.from_dict({"text": ["a", "b"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.MEMORY)
        items = dataset.sample(0)
        assert items == []

    def test_sample_n_larger_than_dataset(self, temp_store):
        """Test sample with n larger than dataset returns all in random order."""
        ds = Dataset.from_dict({"text": ["a", "b", "c"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.MEMORY)
        items = dataset.sample(10)
        assert len(items) == 3
        # Should contain all items (order may differ)
        # Convert to lists for comparison since dicts are unhashable
        all_items = list(dataset.iter_items())
        assert len(items) == len(all_items)
        # Check that all items from dataset are in sampled items
        for item in all_items:
            assert item in items


class TestBaseDatasetFactoryMethods:
    """Tests for BaseDataset factory methods."""

    def test_from_huggingface_success(self, temp_store):
        """Test from_huggingface factory method."""
        with patch("amber.datasets.base_dataset.load_dataset") as mock_load:
            mock_ds = Dataset.from_dict({"text": ["a", "b"]})
            mock_load.return_value = mock_ds

            dataset = ConcreteBaseDataset.from_huggingface("test/dataset", temp_store, split="train")
            assert len(dataset) == 2
            mock_load.assert_called_once()

    def test_from_huggingface_with_streaming(self, temp_store):
        """Test from_huggingface with streaming=True."""
        with patch("amber.datasets.base_dataset.load_dataset") as mock_load:
            mock_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
            mock_load.return_value = mock_ds

            dataset = ConcreteBaseDataset.from_huggingface(
                "test/dataset", temp_store, split="train", loading_strategy=LoadingStrategy.STREAMING
            )
            assert dataset._is_iterable
            mock_load.assert_called_once_with(
                path="test/dataset",
                split="train",
                revision=None,
                streaming=True,
            )

    def test_from_huggingface_empty_repo_id_raises_error(self, temp_store):
        """Test that empty repo_id raises ValueError."""
        with pytest.raises(ValueError, match="repo_id must be a non-empty string"):
            ConcreteBaseDataset.from_huggingface("", temp_store)

    def test_from_huggingface_none_store_raises_error(self):
        """Test that None store raises ValueError."""
        with pytest.raises(ValueError, match="store cannot be None"):
            ConcreteBaseDataset.from_huggingface("test/dataset", None)

    def test_from_huggingface_load_failure_raises_error(self, temp_store):
        """Test that load failure raises RuntimeError."""
        with patch("amber.datasets.base_dataset.load_dataset") as mock_load:
            mock_load.side_effect = Exception("Network error")
            with pytest.raises(RuntimeError, match="Failed to load dataset"):
                ConcreteBaseDataset.from_huggingface("test/dataset", temp_store)

    def test_from_csv_success(self, temp_store):
        """Test from_csv factory method."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text\n")
            f.write("Hello\n")
            f.write("World\n")
            csv_path = f.name

        try:
            dataset = ConcreteBaseDataset.from_csv(csv_path, temp_store)
            items = list(dataset.iter_items())
            assert len(items) == 2
        finally:
            Path(csv_path).unlink()

    def test_from_csv_file_not_found_raises_error(self, temp_store):
        """Test that non-existent CSV file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ConcreteBaseDataset.from_csv("/nonexistent/file.csv", temp_store)

    def test_from_csv_directory_raises_error(self, temp_store):
        """Test that directory path raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Source must be a file"):
                ConcreteBaseDataset.from_csv(tmpdir, temp_store)

    def test_from_json_success(self, temp_store):
        """Test from_json factory method."""
        import json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"text": "Hello"}, {"text": "World"}], f)
            json_path = f.name

        try:
            dataset = ConcreteBaseDataset.from_json(json_path, temp_store)
            items = list(dataset.iter_items())
            assert len(items) == 2
        finally:
            Path(json_path).unlink()

    def test_from_json_file_not_found_raises_error(self, temp_store):
        """Test that non-existent JSON file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ConcreteBaseDataset.from_json("/nonexistent/file.json", temp_store)

    def test_from_json_directory_raises_error(self, temp_store):
        """Test that directory path raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Source must be a file"):
                ConcreteBaseDataset.from_json(tmpdir, temp_store)


class TestBaseDatasetPostProcessing:
    """Tests for BaseDataset post-processing methods."""

    def test_drop_na(self):
        """Test _drop_na method."""
        data = {
            "text": ["a", "b", None, "d", "", "  "],
            "label": [1, None, 3, 4, 5, 6],
            "other": ["x", "y", "z", "w", "v", "u"],
        }
        ds = Dataset.from_dict(data)

        cleaned_ds = BaseDataset._drop_na(ds, ["text"])
        assert len(cleaned_ds) == 3
        assert cleaned_ds["text"] == ["a", "b", "d"]

        cleaned_ds_label = BaseDataset._drop_na(ds, ["label"])
        assert len(cleaned_ds_label) == 5
        assert cleaned_ds_label["label"] == [1, 3, 4, 5, 6]

        cleaned_ds_both = BaseDataset._drop_na(ds, ["text", "label"])
        assert len(cleaned_ds_both) == 2
        assert cleaned_ds_both["text"] == ["a", "d"]

    def test_stratified_sample(self):
        """Test _stratified_sample method."""
        data = {
            "text": ["a"] * 10 + ["b"] * 20 + ["c"] * 5,
            "label": ["A"] * 10 + ["B"] * 20 + ["C"] * 5,
        }
        ds = Dataset.from_dict(data)
        sampled_ds = BaseDataset._stratified_sample(ds, stratify_by="label", sample_size=10, seed=42)
        assert len(sampled_ds) == 10

        counts = defaultdict(int)
        for item in sampled_ds:
            counts[item["label"]] += 1

        assert 2 <= counts["A"] <= 4
        assert 5 <= counts["B"] <= 7
        assert 1 <= counts["C"] <= 2

    def test_stratified_sample_with_limit_larger_than_dataset(self):
        """Test stratified sample when limit > dataset size."""
        data = {"label": ["A", "B", "A"]}
        ds = Dataset.from_dict(data)
        sampled_ds = BaseDataset._stratified_sample(ds, stratify_by="label", sample_size=100, seed=42)
        assert len(sampled_ds) == 3
        assert sorted(sampled_ds["label"]) == ["A", "A", "B"]

    def test_postprocess_non_streaming_dataset_integration(self):
        """Test full post-processing pipeline."""
        data = {
            "text": ["a", "b", None, "d", "e", "f"],
            "label": ["A", "B", "A", "B", "A", "B"],
        }
        ds = Dataset.from_dict(data)
        processed_ds = BaseDataset._postprocess_non_streaming_dataset(
            ds,
            filters={"label": "B"},
            limit=2,
            stratify_by="label",
            stratify_seed=42,
            drop_na_columns=["text"],
        )

        assert len(processed_ds) == 2
        assert all(item["label"] == "B" for item in processed_ds)
        assert all(item["text"] is not None for item in processed_ds)

    def test_postprocess_non_streaming_dataset_invalid_limit_raises(self):
        """limit must be > 0 both with and without stratification."""
        data = {
            "text": ["a", "b", "c"],
            "label": ["A", "B", "A"],
        }
        ds = Dataset.from_dict(data)

        # No stratification: limit <= 0 should raise
        with pytest.raises(ValueError, match="limit must be > 0"):
            BaseDataset._postprocess_non_streaming_dataset(ds, limit=0)

        # With stratification: sample_size (derived from limit) must be > 0
        with pytest.raises(ValueError, match="limit must be > 0 when stratifying"):
            BaseDataset._postprocess_non_streaming_dataset(
                ds,
                limit=0,
                stratify_by="label",
            )

    def test_stratified_sample_invalid_column_and_sample_size_raises(self):
        """_stratified_sample should validate column name and sample_size."""
        data = {
            "text": ["a", "b", "c"],
            "label": ["A", "B", "A"],
        }
        ds = Dataset.from_dict(data)

        # Invalid stratify_by column
        with pytest.raises(ValueError, match="Column 'missing' not found"):
            BaseDataset._stratified_sample(ds, stratify_by="missing", sample_size=2, seed=42)

        # Non‑positive sample_size
        with pytest.raises(ValueError, match="sample_size must be greater than 0"):
            BaseDataset._stratified_sample(ds, stratify_by="label", sample_size=0, seed=42)

    def test_from_csv_with_drop_na_and_stratify(self, temp_store):
        """Test from_csv with drop_na and stratification."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text,label\n")
            f.write("a,A\n")
            f.write("b,B\n")
            f.write(",A\n")  # Empty text
            f.write("d,B\n")
            f.write("e,A\n")
            csv_path = f.name

        try:
            dataset = ConcreteBaseDataset.from_csv(
                csv_path, temp_store, drop_na_columns=["text"], stratify_by="label", stratify_seed=42
            )
            assert len(dataset) == 4
            texts = dataset.get_all_texts()
            assert "a" in texts
            assert "b" in texts
            assert "d" in texts
            assert "e" in texts
            assert "" not in texts
        finally:
            Path(csv_path).unlink()


class TestBaseDatasetSourceLoaders:
    """Tests for BaseDataset source loading methods (_load_csv_source, _load_json_source)."""

    def test_load_csv_source_streaming(self):
        """Test loading CSV in streaming mode."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text\n")
            f.write("a\n")
            f.write("b\n")
            csv_path = f.name

        try:
            ds = BaseDataset._load_csv_source(csv_path, delimiter=",", streaming=True)
            assert isinstance(ds, IterableDataset)
            items = list(ds)
            assert len(items) == 2
            assert items[0]["text"] == "a"
        finally:
            Path(csv_path).unlink()

    def test_load_csv_source_malformed_raises_runtime_error(self):
        """Test that malformed CSV raises RuntimeError."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text\n")
            csv_path = f.name

        try:
            with patch("amber.datasets.base_dataset.load_dataset") as mock_load:
                mock_load.side_effect = Exception("CSV Parse Error")
                with pytest.raises(RuntimeError, match="Failed to load CSV dataset"):
                    BaseDataset._load_csv_source(csv_path, delimiter=",", streaming=False)
        finally:
            Path(csv_path).unlink()

    def test_load_json_source_streaming(self):
        """Test loading JSON in streaming mode."""
        import json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"text": "a"}, {"text": "b"}], f)
            json_path = f.name

        try:
            ds = BaseDataset._load_json_source(json_path, streaming=True)
            assert isinstance(ds, IterableDataset)
            items = list(ds)
            assert len(items) == 2
            assert items[0]["text"] == "a"
        finally:
            Path(json_path).unlink()

    def test_load_json_source_malformed_raises_runtime_error(self):
        """Test that malformed JSON raises RuntimeError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid_json")
            json_path = f.name

        try:
            with pytest.raises(RuntimeError, match="Failed to load JSON dataset"):
                BaseDataset._load_json_source(json_path, streaming=False)
        finally:
            Path(json_path).unlink()
