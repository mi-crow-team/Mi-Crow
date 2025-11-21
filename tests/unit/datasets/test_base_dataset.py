"""Tests for BaseDataset abstract class."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datasets import Dataset, IterableDataset

from amber.datasets.base_dataset import BaseDataset
from amber.datasets.loading_strategy import LoadingStrategy
from amber.store.store import Store
from tests.unit.fixtures.stores import create_temp_store


class ConcreteBaseDataset(BaseDataset):
    """Concrete implementation of BaseDataset for testing."""

    def __len__(self) -> int:
        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            raise NotImplementedError("len() not supported for ITERABLE_ONLY datasets")
        return self._ds.num_rows

    def __getitem__(self, idx):
        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            raise NotImplementedError("Indexing not supported for ITERABLE_ONLY datasets")
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
        """Test initialization with DYNAMIC_LOAD strategy."""
        ds = Dataset.from_dict({"text": ["a", "b", "c"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.DYNAMIC_LOAD)
        assert dataset._loading_strategy == LoadingStrategy.DYNAMIC_LOAD
        assert not dataset._is_iterable
        assert len(dataset) == 3

    def test_init_with_iterable_only_strategy(self, temp_store):
        """Test initialization with ITERABLE_ONLY strategy."""
        iter_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
        dataset = ConcreteBaseDataset(iter_ds, temp_store, LoadingStrategy.ITERABLE_ONLY)
        assert dataset._loading_strategy == LoadingStrategy.ITERABLE_ONLY
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
        """Test that IterableDataset is converted to Dataset for DYNAMIC_LOAD strategy."""
        iter_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
        dataset = ConcreteBaseDataset(iter_ds, temp_store, LoadingStrategy.DYNAMIC_LOAD)
        assert not dataset._is_iterable
        assert len(dataset) == 1

    def test_init_converts_dataset_to_iterable_for_iterable_only(self, temp_store):
        """Test that Dataset is converted to IterableDataset for ITERABLE_ONLY strategy."""
        ds = Dataset.from_dict({"text": ["a", "b"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.ITERABLE_ONLY)
        assert dataset._is_iterable


class TestBaseDatasetProperties:
    """Tests for BaseDataset properties."""

    def test_is_streaming_memory_false(self, temp_store):
        """Test is_streaming property for MEMORY strategy."""
        ds = Dataset.from_dict({"text": ["a"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.MEMORY)
        assert not dataset.is_streaming

    def test_is_streaming_dynamic_load_true(self, temp_store):
        """Test is_streaming property for DYNAMIC_LOAD strategy."""
        ds = Dataset.from_dict({"text": ["a"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.DYNAMIC_LOAD)
        assert dataset.is_streaming

    def test_is_streaming_iterable_only_true(self, temp_store):
        """Test is_streaming property for ITERABLE_ONLY strategy."""
        iter_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
        dataset = ConcreteBaseDataset(iter_ds, temp_store, LoadingStrategy.ITERABLE_ONLY)
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
        """Test get_batch with DYNAMIC_LOAD strategy."""
        ds = Dataset.from_dict({"text": ["a", "b", "c", "d", "e"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.DYNAMIC_LOAD)
        batch = dataset.get_batch(1, 2)
        assert len(batch) == 2

    def test_get_batch_iterable_only_raises_error(self, temp_store):
        """Test that get_batch raises NotImplementedError for ITERABLE_ONLY."""
        iter_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
        dataset = ConcreteBaseDataset(iter_ds, temp_store, LoadingStrategy.ITERABLE_ONLY)
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
        """Test head with ITERABLE_ONLY strategy."""
        iter_ds = IterableDataset.from_generator(
            lambda: iter([{"text": f"text_{i}"} for i in range(10)])
        )
        dataset = ConcreteBaseDataset(iter_ds, temp_store, LoadingStrategy.ITERABLE_ONLY)
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
        """Test sample with DYNAMIC_LOAD strategy."""
        ds = Dataset.from_dict({"text": ["a", "b", "c", "d", "e"]})
        dataset = ConcreteBaseDataset(ds, temp_store, LoadingStrategy.DYNAMIC_LOAD)
        items = dataset.sample(2)
        assert len(items) == 2

    def test_sample_iterable_only_raises_error(self, temp_store):
        """Test that sample raises NotImplementedError for ITERABLE_ONLY."""
        iter_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
        dataset = ConcreteBaseDataset(iter_ds, temp_store, LoadingStrategy.ITERABLE_ONLY)
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
            
            dataset = ConcreteBaseDataset.from_huggingface(
                "test/dataset",
                temp_store,
                split="train"
            )
            assert len(dataset) == 2
            mock_load.assert_called_once()

    def test_from_huggingface_with_streaming(self, temp_store):
        """Test from_huggingface with streaming=True."""
        with patch("amber.datasets.base_dataset.load_dataset") as mock_load:
            mock_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
            mock_load.return_value = mock_ds
            
            dataset = ConcreteBaseDataset.from_huggingface(
                "test/dataset",
                temp_store,
                split="train",
                loading_strategy=LoadingStrategy.ITERABLE_ONLY
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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

