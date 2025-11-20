import pytest
import tempfile
from pathlib import Path
from datasets import Dataset, IterableDataset
from unittest.mock import Mock, patch

from amber.adapters.classification_dataset import ClassificationDataset
from amber.adapters.loading_strategy import LoadingStrategy
from amber.store.local_store import LocalStore


@pytest.fixture
def temp_store(tmp_path):
    return LocalStore(tmp_path / "store")


@pytest.fixture
def sample_dataset():
    return Dataset.from_dict({
        "text": ["Hello world", "Test text", "Another example"],
        "category": ["A", "B", "A"]
    })


def test_classification_dataset_init(sample_dataset, temp_store):
    """Test basic initialization."""
    ds = ClassificationDataset(sample_dataset, temp_store)
    assert len(ds) == 3
    assert ds._text_field == "text"
    assert ds._category_field == "category"


def test_classification_dataset_init_custom_fields(temp_store):
    """Test initialization with custom field names."""
    ds = Dataset.from_dict({
        "content": ["Text 1", "Text 2"],
        "label": ["X", "Y"]
    })
    cls_ds = ClassificationDataset(ds, temp_store, text_field="content", category_field="label")
    assert cls_ds._text_field == "content"
    assert cls_ds._category_field == "label"


def test_classification_dataset_init_missing_text_field(temp_store):
    """Test initialization fails when text field is missing."""
    ds = Dataset.from_dict({"category": ["A", "B"]})
    with pytest.raises(ValueError, match="text"):
        ClassificationDataset(ds, temp_store)


def test_classification_dataset_init_missing_category_field(temp_store):
    """Test initialization fails when category field is missing."""
    ds = Dataset.from_dict({"text": ["A", "B"]})
    with pytest.raises(ValueError, match="category"):
        ClassificationDataset(ds, temp_store)


def test_classification_dataset_getitem_single(sample_dataset, temp_store):
    """Test getting a single item by index."""
    ds = ClassificationDataset(sample_dataset, temp_store)
    item = ds[0]
    assert item == {"text": "Hello world", "category": "A"}


def test_classification_dataset_getitem_slice(sample_dataset, temp_store):
    """Test getting items by slice."""
    ds = ClassificationDataset(sample_dataset, temp_store)
    items = ds[0:2]
    assert len(items) == 2
    assert items[0] == {"text": "Hello world", "category": "A"}
    assert items[1] == {"text": "Test text", "category": "B"}


def test_classification_dataset_getitem_slice_with_step(sample_dataset, temp_store):
    """Test getting items by slice with step."""
    ds = ClassificationDataset(sample_dataset, temp_store)
    items = ds[0:3:2]
    assert len(items) == 2
    assert items[0] == {"text": "Hello world", "category": "A"}
    assert items[1] == {"text": "Another example", "category": "A"}


def test_classification_dataset_getitem_list(sample_dataset, temp_store):
    """Test getting items by list of indices."""
    ds = ClassificationDataset(sample_dataset, temp_store)
    items = ds[[0, 2]]
    assert len(items) == 2
    assert items[0] == {"text": "Hello world", "category": "A"}
    assert items[1] == {"text": "Another example", "category": "A"}


def test_classification_dataset_getitem_invalid_type(sample_dataset, temp_store):
    """Test getting item with invalid index type."""
    ds = ClassificationDataset(sample_dataset, temp_store)
    with pytest.raises(TypeError):
        _ = ds["invalid"]


def test_classification_dataset_iter_items(sample_dataset, temp_store):
    """Test iterating over items."""
    ds = ClassificationDataset(sample_dataset, temp_store)
    items = list(ds.iter_items())
    assert len(items) == 3
    assert items[0] == {"text": "Hello world", "category": "A"}


def test_classification_dataset_iter_items_missing_field(temp_store):
    """Test iterating when field is missing."""
    ds = Dataset.from_dict({
        "text": ["Text 1"],
        "category": ["A"]
    })
    cls_ds = ClassificationDataset(ds, temp_store)
    # Should work with default fields
    items = list(cls_ds.iter_items())
    assert len(items) == 1


def test_classification_dataset_iter_batches(sample_dataset, temp_store):
    """Test iterating over batches."""
    ds = ClassificationDataset(sample_dataset, temp_store)
    batches = list(ds.iter_batches(batch_size=2))
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1
    # Verify structure
    assert isinstance(batches[0][0], dict)
    assert "text" in batches[0][0]
    assert "category" in batches[0][0]


def test_classification_dataset_iter_batches_invalid_size(sample_dataset, temp_store):
    """Test iterating with invalid batch size."""
    ds = ClassificationDataset(sample_dataset, temp_store)
    with pytest.raises(ValueError, match="batch_size"):
        list(ds.iter_batches(batch_size=0))


def test_classification_dataset_get_categories(sample_dataset, temp_store):
    """Test getting unique categories."""
    ds = ClassificationDataset(sample_dataset, temp_store)
    categories = ds.get_categories()
    assert set(categories) == {"A", "B"}


def test_classification_dataset_get_texts(sample_dataset, temp_store):
    """Test getting all texts."""
    ds = ClassificationDataset(sample_dataset, temp_store)
    texts = ds.get_texts()
    assert texts == ["Hello world", "Test text", "Another example"]


def test_classification_dataset_get_categories_for_texts(sample_dataset, temp_store):
    """Test getting categories for specific texts."""
    ds = ClassificationDataset(sample_dataset, temp_store)
    categories = ds.get_categories_for_texts(["Hello world", "Test text"])
    assert categories == ["A", "B"]


def test_classification_dataset_streaming_not_implemented(temp_store):
    """Test that streaming datasets raise NotImplementedError for len and indexing."""
    iter_ds = IterableDataset.from_generator(lambda: iter([{"text": "a", "category": "X"}]))
    ds = ClassificationDataset(iter_ds, temp_store, loading_strategy=LoadingStrategy.ITERABLE_ONLY)
    
    with pytest.raises(NotImplementedError):
        _ = len(ds)
    
    with pytest.raises(NotImplementedError):
        _ = ds[0]
    
    with pytest.raises(NotImplementedError):
        _ = ds.get_categories_for_texts(["a"])


def test_classification_dataset_streaming_iter_items(temp_store):
    """Test iterating streaming dataset."""
    iter_ds = IterableDataset.from_generator(
        lambda: iter([{"text": "a", "category": "X"}, {"text": "b", "category": "Y"}])
    )
    ds = ClassificationDataset(iter_ds, temp_store, loading_strategy=LoadingStrategy.ITERABLE_ONLY)
    items = list(ds.iter_items())
    assert len(items) == 2


def test_classification_dataset_streaming_iter_batches(temp_store):
    """Test iterating batches from streaming dataset."""
    iter_ds = IterableDataset.from_generator(
        lambda: iter([{"text": "a", "category": "X"}, {"text": "b", "category": "Y"}])
    )
    ds = ClassificationDataset(iter_ds, temp_store, loading_strategy=LoadingStrategy.ITERABLE_ONLY)
    batches = list(ds.iter_batches(batch_size=1))
    assert len(batches) == 2
    assert len(batches[0]) == 1


def test_classification_dataset_from_huggingface(temp_store):
    """Test loading from HuggingFace Hub."""
    with patch("amber.adapters.classification_dataset.load_dataset") as mock_load:
        mock_ds = Dataset.from_dict({"text": ["a"], "category": ["X"]})
        mock_load.return_value = mock_ds
        
        ds = ClassificationDataset.from_huggingface(
            "test/dataset",
            temp_store,
            text_field="text",
            category_field="category"
        )
        assert len(ds) == 1
        mock_load.assert_called_once()


def test_classification_dataset_from_huggingface_with_filters(temp_store):
    """Test loading from HuggingFace with filters."""
    with patch("amber.adapters.classification_dataset.load_dataset") as mock_load:
        mock_ds = Dataset.from_dict({
            "text": ["a", "b", "c"],
            "category": ["X", "Y", "X"]
        })
        mock_load.return_value = mock_ds
        
        ds = ClassificationDataset.from_huggingface(
            "test/dataset",
            temp_store,
            filters={"category": "X"}
        )
        # Filter should be applied
        mock_load.assert_called_once()


def test_classification_dataset_from_huggingface_with_limit(temp_store):
    """Test loading from HuggingFace with limit."""
    with patch("amber.adapters.classification_dataset.load_dataset") as mock_load:
        mock_ds = Dataset.from_dict({
            "text": ["a", "b", "c"],
            "category": ["X", "Y", "Z"]
        })
        mock_load.return_value = mock_ds
        
        ds = ClassificationDataset.from_huggingface(
            "test/dataset",
            temp_store,
            limit=2
        )
        assert len(ds) == 2


def test_classification_dataset_from_csv(tmp_path):
    """Test loading from CSV file."""
    # Use a separate store to avoid caching conflicts
    store = LocalStore(tmp_path / "csv_store")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("text,category\n")
        f.write("Hello,A\n")
        f.write("World,B\n")
        csv_path = f.name
    
    try:
        ds = ClassificationDataset.from_csv(
            csv_path,
            store,
            text_field="text",
            category_field="category",
            loading_strategy=LoadingStrategy.ITERABLE_ONLY  # Use streaming to avoid cache conflict
        )
        items = list(ds.iter_items())
        assert len(items) == 2
        assert items[0]["category"] == "A"
    finally:
        Path(csv_path).unlink()


def test_classification_dataset_from_json(tmp_path):
    """Test loading from JSON file."""
    import json
    # Use a separate store to avoid caching conflicts
    store = LocalStore(tmp_path / "json_store")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([
            {"text": "Hello", "category": "A"},
            {"text": "World", "category": "B"}
        ], f)
        json_path = f.name
    
    try:
        ds = ClassificationDataset.from_json(
            json_path,
            store,
            text_field="text",
            category_field="category",
            loading_strategy=LoadingStrategy.ITERABLE_ONLY  # Use streaming to avoid cache conflict
        )
        items = list(ds.iter_items())
        assert len(items) == 2
    finally:
        Path(json_path).unlink()


def test_classification_dataset_get_categories_with_none(temp_store):
    """Test getting categories excluding None values."""
    ds = Dataset.from_dict({
        "text": ["a", "b", "c"],
        "category": ["X", None, "Y"]
    })
    cls_ds = ClassificationDataset(ds, temp_store)
    categories = cls_ds.get_categories()
    assert None not in categories
    assert set(categories) == {"X", "Y"}


def test_classification_dataset_streaming_get_categories(temp_store):
    """Test getting categories from streaming dataset."""
    iter_ds = IterableDataset.from_generator(
        lambda: iter([
            {"text": "a", "category": "X"},
            {"text": "b", "category": "Y"},
            {"text": "c", "category": "X"}
        ])
    )
    ds = ClassificationDataset(iter_ds, temp_store, loading_strategy=LoadingStrategy.ITERABLE_ONLY)
    categories = ds.get_categories()
    assert set(categories) == {"X", "Y"}


def test_classification_dataset_streaming_get_texts(temp_store):
    """Test getting texts from streaming dataset."""
    iter_ds = IterableDataset.from_generator(
        lambda: iter([
            {"text": "a", "category": "X"},
            {"text": "b", "category": "Y"}
        ])
    )
    ds = ClassificationDataset(iter_ds, temp_store, loading_strategy=LoadingStrategy.ITERABLE_ONLY)
    texts = ds.get_texts()
    assert texts == ["a", "b"]

