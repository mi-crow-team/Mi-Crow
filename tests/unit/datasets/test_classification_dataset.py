"""Tests for ClassificationDataset."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from datasets import Dataset, IterableDataset

from amber.datasets.classification_dataset import ClassificationDataset
from amber.datasets.loading_strategy import LoadingStrategy
from tests.unit.fixtures.stores import create_temp_store


class TestClassificationDatasetInitialization:
    """Tests for ClassificationDataset initialization."""

    def test_init_with_single_category_field(self, temp_store):
        """Test initialization with single category field."""
        ds = Dataset.from_dict({"text": ["Text 1", "Text 2"], "category": ["cat_a", "cat_b"]})
        dataset = ClassificationDataset(ds, temp_store, category_field="category")
        assert dataset._text_field == "text"
        assert dataset._category_fields == ["category"]
        assert len(dataset) == 2

    def test_init_with_multiple_category_fields(self, temp_store):
        """Test initialization with multiple category fields."""
        ds = Dataset.from_dict({"text": ["Text 1", "Text 2"], "label1": ["a", "b"], "label2": ["x", "y"]})
        dataset = ClassificationDataset(ds, temp_store, category_field=["label1", "label2"])
        assert dataset._category_fields == ["label1", "label2"]
        assert len(dataset) == 2

    def test_init_with_custom_text_field(self, temp_store):
        """Test initialization with custom text field."""
        ds = Dataset.from_dict({"content": ["Text 1", "Text 2"], "category": ["a", "b"]})
        dataset = ClassificationDataset(ds, temp_store, text_field="content", category_field="category")
        assert dataset._text_field == "content"
        assert len(dataset) == 2

    def test_init_empty_text_field_raises_error(self, temp_store):
        """Test that empty text_field raises ValueError."""
        ds = Dataset.from_dict({"text": ["a"], "category": ["b"]})
        with pytest.raises(ValueError, match="text_field must be a non-empty string"):
            ClassificationDataset(ds, temp_store, text_field="")

    def test_init_empty_category_field_raises_error(self, temp_store):
        """Test that empty category_field raises ValueError."""
        ds = Dataset.from_dict({"text": ["a"], "category": ["b"]})
        with pytest.raises(ValueError, match="category_field cannot be empty"):
            ClassificationDataset(ds, temp_store, category_field=[])

    def test_init_missing_text_field_raises_error(self, temp_store):
        """Test that missing text field raises ValueError."""
        ds = Dataset.from_dict({"other": ["a"], "category": ["b"]})
        with pytest.raises(ValueError, match="Dataset must have a 'text' column"):
            ClassificationDataset(ds, temp_store)

    def test_init_missing_category_field_raises_error(self, temp_store):
        """Test that missing category field raises ValueError."""
        ds = Dataset.from_dict({"text": ["a"]})
        with pytest.raises(ValueError, match="Dataset must have a 'category' column"):
            ClassificationDataset(ds, temp_store)

    def test_init_missing_one_of_multiple_category_fields_raises_error(self, temp_store):
        """Test that missing one of multiple category fields raises ValueError."""
        ds = Dataset.from_dict({"text": ["a"], "label1": ["b"]})
        with pytest.raises(ValueError, match="Dataset must have a 'label2' column"):
            ClassificationDataset(ds, temp_store, category_field=["label1", "label2"])

    def test_init_with_iterable_dataset(self, temp_store):
        """Test initialization with IterableDataset."""
        iter_ds = IterableDataset.from_generator(lambda: iter([{"text": "a", "category": "b"}]))
        dataset = ClassificationDataset(iter_ds, temp_store, loading_strategy=LoadingStrategy.ITERABLE_ONLY)
        assert dataset._is_iterable


class TestClassificationDatasetLen:
    """Tests for __len__ method."""

    def test_len_memory_strategy(self, temp_store):
        """Test len with MEMORY strategy."""
        ds = Dataset.from_dict({"text": ["a", "b", "c"], "category": ["x", "y", "z"]})
        dataset = ClassificationDataset(ds, temp_store, LoadingStrategy.MEMORY)
        assert len(dataset) == 3

    def test_len_iterable_only_raises_error(self, temp_store):
        """Test that len raises NotImplementedError for ITERABLE_ONLY."""
        iter_ds = IterableDataset.from_generator(lambda: iter([{"text": "a", "category": "b"}]))
        dataset = ClassificationDataset(iter_ds, temp_store, LoadingStrategy.ITERABLE_ONLY)
        with pytest.raises(NotImplementedError):
            len(dataset)


class TestClassificationDatasetGetItem:
    """Tests for __getitem__ method."""

    def test_getitem_single_index(self, temp_store):
        """Test getting single item by index."""
        ds = Dataset.from_dict({"text": ["Text 1", "Text 2"], "category": ["cat_a", "cat_b"]})
        dataset = ClassificationDataset(ds, temp_store)
        item = dataset[0]
        assert item == {"text": "Text 1", "category": "cat_a"}

    def test_getitem_slice(self, temp_store):
        """Test getting items by slice."""
        ds = Dataset.from_dict({"text": ["Text 1", "Text 2", "Text 3"], "category": ["a", "b", "c"]})
        dataset = ClassificationDataset(ds, temp_store)
        items = dataset[0:2]
        assert len(items) == 2
        assert items[0] == {"text": "Text 1", "category": "a"}

    def test_getitem_list_of_indices(self, temp_store):
        """Test getting items by list of indices."""
        ds = Dataset.from_dict({"text": ["Text 1", "Text 2", "Text 3"], "category": ["a", "b", "c"]})
        dataset = ClassificationDataset(ds, temp_store)
        items = dataset[[0, 2]]
        assert len(items) == 2
        assert items[0]["text"] == "Text 1"
        assert items[1]["text"] == "Text 3"

    def test_getitem_negative_index(self, temp_store):
        """Test getting item with negative index."""
        ds = Dataset.from_dict({"text": ["Text 1", "Text 2"], "category": ["a", "b"]})
        dataset = ClassificationDataset(ds, temp_store)
        item = dataset[-1]
        assert item == {"text": "Text 2", "category": "b"}

    def test_getitem_out_of_bounds_raises_error(self, temp_store):
        """Test that out of bounds index raises IndexError."""
        ds = Dataset.from_dict({"text": ["Text 1"], "category": ["a"]})
        dataset = ClassificationDataset(ds, temp_store)
        with pytest.raises(IndexError, match="Index 5 out of bounds"):
            _ = dataset[5]

    def test_getitem_empty_dataset_raises_error(self, temp_store):
        """Test that indexing empty dataset raises ValueError."""
        ds = Dataset.from_dict({"text": [], "category": []})
        dataset = ClassificationDataset(ds, temp_store)
        with pytest.raises(ValueError, match="Cannot index into empty dataset"):
            _ = dataset[0]

    def test_getitem_iterable_only_raises_error(self, temp_store):
        """Test that indexing raises NotImplementedError for ITERABLE_ONLY."""
        iter_ds = IterableDataset.from_generator(lambda: iter([{"text": "a", "category": "b"}]))
        dataset = ClassificationDataset(iter_ds, temp_store, LoadingStrategy.ITERABLE_ONLY)
        with pytest.raises(NotImplementedError):
            _ = dataset[0]

    def test_getitem_multiple_category_fields(self, temp_store):
        """Test getting item with multiple category fields."""
        ds = Dataset.from_dict({"text": ["Text 1"], "label1": ["a"], "label2": ["b"]})
        dataset = ClassificationDataset(ds, temp_store, category_field=["label1", "label2"])
        item = dataset[0]
        assert item == {"text": "Text 1", "label1": "a", "label2": "b"}


class TestClassificationDatasetIterItems:
    """Tests for iter_items method."""

    def test_iter_items_memory_strategy(self, temp_store):
        """Test iter_items with MEMORY strategy."""
        ds = Dataset.from_dict({"text": ["Text 1", "Text 2"], "category": ["a", "b"]})
        dataset = ClassificationDataset(ds, temp_store)
        items = list(dataset.iter_items())
        assert len(items) == 2
        assert items[0] == {"text": "Text 1", "category": "a"}

    def test_iter_items_iterable_only(self, temp_store):
        """Test iter_items with ITERABLE_ONLY strategy."""
        iter_ds = IterableDataset.from_generator(
            lambda: iter([{"text": "Text 1", "category": "a"}, {"text": "Text 2", "category": "b"}])
        )
        dataset = ClassificationDataset(iter_ds, temp_store, LoadingStrategy.ITERABLE_ONLY)
        items = list(dataset.iter_items())
        assert len(items) == 2


class TestClassificationDatasetIterBatches:
    """Tests for iter_batches method."""

    def test_iter_batches_memory_strategy(self, temp_store):
        """Test iter_batches with MEMORY strategy."""
        ds = Dataset.from_dict({"text": ["Text 1", "Text 2", "Text 3"], "category": ["a", "b", "c"]})
        dataset = ClassificationDataset(ds, temp_store)
        batches = list(dataset.iter_batches(batch_size=2))
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1

    def test_iter_batches_iterable_only(self, temp_store):
        """Test iter_batches with ITERABLE_ONLY strategy."""
        iter_ds = IterableDataset.from_generator(
            lambda: iter([{"text": f"Text {i}", "category": f"cat_{i}"} for i in range(5)])
        )
        dataset = ClassificationDataset(iter_ds, temp_store, LoadingStrategy.ITERABLE_ONLY)
        batches = list(dataset.iter_batches(batch_size=2))
        assert len(batches) == 3

    def test_iter_batches_invalid_batch_size_raises_error(self, temp_store):
        """Test that invalid batch_size raises ValueError."""
        ds = Dataset.from_dict({"text": ["Text 1"], "category": ["a"]})
        dataset = ClassificationDataset(ds, temp_store)
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            list(dataset.iter_batches(batch_size=0))


class TestClassificationDatasetGetCategories:
    """Tests for get_categories method."""

    def test_get_categories_single_field_memory(self, temp_store):
        """Test get_categories with single category field and MEMORY strategy."""
        ds = Dataset.from_dict({"text": ["Text 1", "Text 2", "Text 3"], "category": ["a", "b", "a"]})
        dataset = ClassificationDataset(ds, temp_store)
        categories = dataset.get_categories()
        assert isinstance(categories, list)
        assert set(categories) == {"a", "b"}

    def test_get_categories_single_field_iterable(self, temp_store):
        """Test get_categories with single category field and ITERABLE_ONLY."""
        iter_ds = IterableDataset.from_generator(
            lambda: iter(
                [
                    {"text": "Text 1", "category": "a"},
                    {"text": "Text 2", "category": "b"},
                    {"text": "Text 3", "category": "a"},
                ]
            )
        )
        dataset = ClassificationDataset(iter_ds, temp_store, LoadingStrategy.ITERABLE_ONLY)
        categories = dataset.get_categories()
        assert isinstance(categories, list)
        assert set(categories) == {"a", "b"}

    def test_get_categories_multiple_fields(self, temp_store):
        """Test get_categories with multiple category fields."""
        ds = Dataset.from_dict({"text": ["Text 1", "Text 2"], "label1": ["a", "b"], "label2": ["x", "y"]})
        dataset = ClassificationDataset(ds, temp_store, category_field=["label1", "label2"])
        categories = dataset.get_categories()
        assert isinstance(categories, dict)
        assert set(categories["label1"]) == {"a", "b"}
        assert set(categories["label2"]) == {"x", "y"}

    def test_get_categories_excludes_none(self, temp_store):
        """Test that get_categories excludes None values."""
        ds = Dataset.from_dict({"text": ["Text 1", "Text 2"], "category": ["a", None]})
        dataset = ClassificationDataset(ds, temp_store)
        categories = dataset.get_categories()
        assert "a" in categories
        assert None not in categories


class TestClassificationDatasetGetTexts:
    """Tests for get_texts method."""

    def test_get_texts_memory_strategy(self, temp_store):
        """Test get_texts with MEMORY strategy."""
        ds = Dataset.from_dict({"text": ["Text 1", "Text 2"], "category": ["a", "b"]})
        dataset = ClassificationDataset(ds, temp_store)
        texts = dataset.get_all_texts()
        assert texts == ["Text 1", "Text 2"]

    def test_get_all_texts_iterable_only(self, temp_store):
        """Test get_all_texts with ITERABLE_ONLY strategy."""
        iter_ds = IterableDataset.from_generator(
            lambda: iter([{"text": "Text 1", "category": "a"}, {"text": "Text 2", "category": "b"}])
        )
        dataset = ClassificationDataset(iter_ds, temp_store, LoadingStrategy.ITERABLE_ONLY)
        texts = dataset.get_all_texts()
        assert texts == ["Text 1", "Text 2"]


class TestClassificationDatasetGetCategoriesForTexts:
    """Tests for get_categories_for_texts method."""

    def test_get_categories_for_texts_single_field(self, temp_store):
        """Test get_categories_for_texts with single category field."""
        ds = Dataset.from_dict({"text": ["Text 1", "Text 2", "Text 3"], "category": ["a", "b", "a"]})
        dataset = ClassificationDataset(ds, temp_store)
        categories = dataset.get_categories_for_texts(["Text 1", "Text 2"])
        assert categories == ["a", "b"]

    def test_get_categories_for_texts_multiple_fields(self, temp_store):
        """Test get_categories_for_texts with multiple category fields."""
        ds = Dataset.from_dict({"text": ["Text 1", "Text 2"], "label1": ["a", "b"], "label2": ["x", "y"]})
        dataset = ClassificationDataset(ds, temp_store, category_field=["label1", "label2"])
        categories = dataset.get_categories_for_texts(["Text 1"])
        assert isinstance(categories[0], dict)
        assert categories[0] == {"label1": "a", "label2": "x"}

    def test_get_categories_for_texts_iterable_only_raises_error(self, temp_store):
        """Test that get_categories_for_texts raises NotImplementedError for ITERABLE_ONLY."""
        iter_ds = IterableDataset.from_generator(lambda: iter([{"text": "a", "category": "b"}]))
        dataset = ClassificationDataset(iter_ds, temp_store, LoadingStrategy.ITERABLE_ONLY)
        with pytest.raises(NotImplementedError):
            dataset.get_categories_for_texts(["a"])

    def test_get_categories_for_texts_empty_list_raises_error(self, temp_store):
        """Test that empty texts list raises ValueError."""
        ds = Dataset.from_dict({"text": ["Text 1"], "category": ["a"]})
        dataset = ClassificationDataset(ds, temp_store)
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            dataset.get_categories_for_texts([])


class TestClassificationDatasetFactoryMethods:
    """Tests for ClassificationDataset factory methods."""

    def test_from_huggingface_success(self, temp_store):
        """Test from_huggingface factory method."""
        with patch("amber.datasets.classification_dataset.load_dataset") as mock_load:
            mock_ds = Dataset.from_dict({"text": ["a", "b"], "category": ["x", "y"]})
            mock_load.return_value = mock_ds

            dataset = ClassificationDataset.from_huggingface(
                "test/dataset", temp_store, text_field="text", category_field="category"
            )
            assert len(dataset) == 2

    def test_from_huggingface_with_filters(self, temp_store):
        """Test from_huggingface with filters."""
        with patch("amber.datasets.classification_dataset.load_dataset") as mock_load:
            mock_ds = Dataset.from_dict({"text": ["a", "b", "c"], "category": ["x", "y", "x"]})
            mock_load.return_value = mock_ds

            dataset = ClassificationDataset.from_huggingface("test/dataset", temp_store, filters={"category": "x"})
            # After filtering, should have 2 items with category "x"
            assert len(dataset) == 2

    def test_from_huggingface_with_limit(self, temp_store):
        """Test from_huggingface with limit."""
        with patch("amber.datasets.classification_dataset.load_dataset") as mock_load:
            mock_ds = Dataset.from_dict(
                {"text": [f"text_{i}" for i in range(10)], "category": [f"cat_{i}" for i in range(10)]}
            )
            mock_load.return_value = mock_ds

            dataset = ClassificationDataset.from_huggingface("test/dataset", temp_store, limit=5)
            assert len(dataset) == 5

    def test_from_huggingface_invalid_limit_raises_error(self, temp_store):
        """Test that invalid limit raises ValueError."""
        with patch("amber.datasets.classification_dataset.load_dataset") as mock_load:
            mock_ds = Dataset.from_dict({"text": ["a"], "category": ["b"]})
            mock_load.return_value = mock_ds

            with pytest.raises((ValueError, RuntimeError), match="limit"):
                ClassificationDataset.from_huggingface("test/dataset", temp_store, limit=0)

    def test_from_csv_success(self, tmp_path):
        """Test from_csv factory method."""
        # Use separate store to avoid cache conflicts
        store = create_temp_store(tmp_path / "csv_store")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text,category\n")
            f.write("Hello,a\n")
            f.write("World,b\n")
            csv_path = f.name

        try:
            dataset = ClassificationDataset.from_csv(
                csv_path,
                store,
                text_field="text",
                category_field="category",
                loading_strategy=LoadingStrategy.ITERABLE_ONLY,  # Use streaming to avoid cache
            )
            items = list(dataset.iter_items())
            assert len(items) == 2
        finally:
            Path(csv_path).unlink()

    def test_from_json_success(self, tmp_path):
        """Test from_json factory method."""
        import json

        # Use separate store to avoid cache conflicts
        store = create_temp_store(tmp_path / "json_store")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"text": "Hello", "category": "a"}, {"text": "World", "category": "b"}], f)
            json_path = f.name

        try:
            dataset = ClassificationDataset.from_json(
                json_path,
                store,
                text_field="text",
                category_field="category",
                loading_strategy=LoadingStrategy.ITERABLE_ONLY,  # Use streaming to avoid cache
            )
            items = list(dataset.iter_items())
            assert len(items) == 2
        finally:
            Path(json_path).unlink()


def test_classification_dataset_slice_with_step(temp_store):
    ds = Dataset.from_dict({"text": ["t1", "t2", "t3", "t4"], "category": ["a", "b", "c", "d"]})
    dataset = ClassificationDataset(ds, temp_store)
    items = dataset[0:4:2]
    assert len(items) == 2


def test_classification_dataset_sequence_invalid(temp_store):
    ds = Dataset.from_dict({"text": ["t1"], "category": ["a"]})
    dataset = ClassificationDataset(ds, temp_store)
    with pytest.raises(IndexError):
        dataset[[0, 5]]


def test_classification_dataset_iter_batches_iterable_only(temp_store):
    iter_ds = IterableDataset.from_generator(
        lambda: iter([{"text": "t1", "category": "a"}, {"text": "t2", "category": "b"}])
    )
    dataset = ClassificationDataset(iter_ds, temp_store, LoadingStrategy.ITERABLE_ONLY)
    batches = list(dataset.iter_batches(1))
    assert len(batches) == 2


def test_classification_dataset_get_categories_multiple_labels_iterable(temp_store):
    iter_ds = IterableDataset.from_generator(
        lambda: iter(
            [
                {"text": "t1", "label1": "x", "label2": "y"},
                {"text": "t2", "label1": "z", "label2": "y"},
            ]
        )
    )
    dataset = ClassificationDataset(
        iter_ds, temp_store, LoadingStrategy.ITERABLE_ONLY, category_field=["label1", "label2"]
    )
    categories = dataset.get_categories()
    assert categories["label1"] == ["x", "z"]


class TestClassificationDatasetExtractTexts:
    """Tests for text extraction methods."""

    def test_extract_texts_from_batch_single_category(self, temp_store):
        """Test extract_texts_from_batch extracts text from dict batch."""
        ds = Dataset.from_dict({"text": ["Text 1", "Text 2"], "category": ["a", "b"]})
        dataset = ClassificationDataset(ds, temp_store)

        batch = [{"text": "Hello", "category": "a"}, {"text": "World", "category": "b"}]
        result = dataset.extract_texts_from_batch(batch)

        assert result == ["Hello", "World"]
        assert all(isinstance(text, str) for text in result)

    def test_extract_texts_from_batch_multiple_categories(self, temp_store):
        """Test extract_texts_from_batch with multiple category fields."""
        ds = Dataset.from_dict({"text": ["Text 1"], "label1": ["a"], "label2": ["b"]})
        dataset = ClassificationDataset(ds, temp_store, category_field=["label1", "label2"])

        batch = [{"text": "First", "label1": "x", "label2": "y"}, {"text": "Second", "label1": "p", "label2": "q"}]
        result = dataset.extract_texts_from_batch(batch)

        assert result == ["First", "Second"]

    def test_extract_texts_from_batch_empty_batch(self, temp_store):
        """Test extract_texts_from_batch with empty batch."""
        ds = Dataset.from_dict({"text": ["Text 1"], "category": ["a"]})
        dataset = ClassificationDataset(ds, temp_store)

        result = dataset.extract_texts_from_batch([])

        assert result == []

    def test_extract_texts_from_batch_missing_text_field_raises(self, temp_store):
        """Test extract_texts_from_batch raises error if text field missing."""
        ds = Dataset.from_dict({"text": ["Text 1"], "category": ["a"]})
        dataset = ClassificationDataset(ds, temp_store)

        batch = [{"category": "a"}]  # Missing text field

        with pytest.raises(ValueError, match="'text' key not found"):
            dataset.extract_texts_from_batch(batch)

    def test_extract_texts_from_batch_integration_with_iter_batches(self, temp_store):
        """Test extract_texts_from_batch works with iter_batches output."""
        ds = Dataset.from_dict({"text": ["t1", "t2", "t3"], "category": ["a", "b", "c"]})
        dataset = ClassificationDataset(ds, temp_store)

        for batch in dataset.iter_batches(batch_size=2):
            extracted = dataset.extract_texts_from_batch(batch)
            assert all(isinstance(text, str) for text in extracted)
            assert len(extracted) == len(batch)
