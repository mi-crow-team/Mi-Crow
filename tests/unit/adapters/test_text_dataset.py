import pytest
import tempfile
from pathlib import Path
from datasets import Dataset, IterableDataset
from unittest.mock import patch

from amber.adapters.text_dataset import TextDataset
from amber.adapters.loading_strategy import LoadingStrategy
from amber.store.local_store import LocalStore


@pytest.fixture
def temp_store(tmp_path):
    return LocalStore(tmp_path / "store")


@pytest.fixture
def sample_dataset():
    return Dataset.from_dict({
        "text": ["Hello world", "Test text", "Another example"]
    })


def test_text_dataset_init(sample_dataset, temp_store):
    """Test basic initialization."""
    ds = TextDataset(sample_dataset, temp_store)
    assert len(ds) == 3
    assert ds._text_field == "text"


def test_text_dataset_init_custom_field(temp_store):
    """Test initialization with custom field name."""
    ds = Dataset.from_dict({"content": ["Text 1", "Text 2"]})
    text_ds = TextDataset(ds, temp_store, text_field="content")
    assert text_ds._text_field == "content"


def test_text_dataset_init_missing_field(temp_store):
    """Test initialization fails when text field is missing."""
    ds = Dataset.from_dict({"other": ["A", "B"]})
    with pytest.raises(ValueError, match="text"):
        TextDataset(ds, temp_store)


def test_text_dataset_getitem_single(sample_dataset, temp_store):
    """Test getting a single item by index."""
    ds = TextDataset(sample_dataset, temp_store)
    item = ds[0]
    assert item == "Hello world"


def test_text_dataset_getitem_slice(sample_dataset, temp_store):
    """Test getting items by slice."""
    ds = TextDataset(sample_dataset, temp_store)
    items = ds[0:2]
    assert len(items) == 2
    assert items == ["Hello world", "Test text"]


def test_text_dataset_getitem_slice_with_step(sample_dataset, temp_store):
    """Test getting items by slice with step."""
    ds = TextDataset(sample_dataset, temp_store)
    items = ds[0:3:2]
    assert len(items) == 2
    assert items == ["Hello world", "Another example"]


def test_text_dataset_getitem_list(sample_dataset, temp_store):
    """Test getting items by list of indices."""
    ds = TextDataset(sample_dataset, temp_store)
    items = ds[[0, 2]]
    assert len(items) == 2
    assert items == ["Hello world", "Another example"]


def test_text_dataset_getitem_invalid_type(sample_dataset, temp_store):
    """Test getting item with invalid index type."""
    ds = TextDataset(sample_dataset, temp_store)
    with pytest.raises(TypeError):
        _ = ds["invalid"]


def test_text_dataset_iter_items(sample_dataset, temp_store):
    """Test iterating over items."""
    ds = TextDataset(sample_dataset, temp_store)
    items = list(ds.iter_items())
    assert len(items) == 3
    assert items == ["Hello world", "Test text", "Another example"]


def test_text_dataset_iter_batches(sample_dataset, temp_store):
    """Test iterating over batches."""
    ds = TextDataset(sample_dataset, temp_store)
    batches = list(ds.iter_batches(batch_size=2))
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1


def test_text_dataset_iter_batches_invalid_size(sample_dataset, temp_store):
    """Test iterating with invalid batch size."""
    ds = TextDataset(sample_dataset, temp_store)
    with pytest.raises(ValueError, match="batch_size"):
        list(ds.iter_batches(batch_size=0))


def test_text_dataset_streaming_not_implemented(temp_store):
    """Test that streaming datasets raise NotImplementedError for len and indexing."""
    iter_ds = IterableDataset.from_generator(lambda: iter([{"text": "a"}]))
    ds = TextDataset(iter_ds, temp_store, loading_strategy=LoadingStrategy.ITERABLE_ONLY)
    
    with pytest.raises(NotImplementedError):
        _ = len(ds)
    
    with pytest.raises(NotImplementedError):
        _ = ds[0]


def test_text_dataset_streaming_iter_items(temp_store):
    """Test iterating streaming dataset."""
    iter_ds = IterableDataset.from_generator(
        lambda: iter([{"text": "a"}, {"text": "b"}])
    )
    ds = TextDataset(iter_ds, temp_store, loading_strategy=LoadingStrategy.ITERABLE_ONLY)
    items = list(ds.iter_items())
    assert len(items) == 2
    assert items == ["a", "b"]


def test_text_dataset_streaming_iter_batches(temp_store):
    """Test iterating batches from streaming dataset."""
    iter_ds = IterableDataset.from_generator(
        lambda: iter([{"text": "a"}, {"text": "b"}])
    )
    ds = TextDataset(iter_ds, temp_store, loading_strategy=LoadingStrategy.ITERABLE_ONLY)
    batches = list(ds.iter_batches(batch_size=1))
    assert len(batches) == 2
    assert len(batches[0]) == 1


def test_text_dataset_from_huggingface(temp_store):
    """Test loading from HuggingFace Hub."""
    with patch("amber.adapters.text_dataset.load_dataset") as mock_load:
        mock_ds = Dataset.from_dict({"text": ["a"]})
        mock_load.return_value = mock_ds
        
        ds = TextDataset.from_huggingface(
            "test/dataset",
            temp_store,
            text_field="text"
        )
        assert len(ds) == 1
        mock_load.assert_called_once()


def test_text_dataset_from_huggingface_with_filters(temp_store):
    """Test loading from HuggingFace with filters."""
    with patch("amber.adapters.text_dataset.load_dataset") as mock_load:
        mock_ds = Dataset.from_dict({
            "text": ["a", "b", "c"],
            "other": [1, 2, 3]
        })
        mock_load.return_value = mock_ds
        
        ds = TextDataset.from_huggingface(
            "test/dataset",
            temp_store,
            filters={"other": 1}
        )
        mock_load.assert_called_once()


def test_text_dataset_from_huggingface_with_limit(temp_store):
    """Test loading from HuggingFace with limit."""
    with patch("amber.adapters.text_dataset.load_dataset") as mock_load:
        mock_ds = Dataset.from_dict({"text": ["a", "b", "c"]})
        mock_load.return_value = mock_ds
        
        ds = TextDataset.from_huggingface(
            "test/dataset",
            temp_store,
            limit=2
        )
        assert len(ds) == 2


def test_text_dataset_from_csv(tmp_path):
    """Test loading from CSV file."""
    # Use a separate store to avoid caching conflicts
    store = LocalStore(tmp_path / "csv_store")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("text\n")
        f.write("Hello\n")
        f.write("World\n")
        csv_path = f.name
    
    try:
        ds = TextDataset.from_csv(
            csv_path,
            store,
            text_field="text",
            loading_strategy=LoadingStrategy.ITERABLE_ONLY  # Use streaming to avoid cache conflict
        )
        items = list(ds.iter_items())
        assert len(items) == 2
        assert items[0] == "Hello"
    finally:
        Path(csv_path).unlink()


def test_text_dataset_from_json(tmp_path):
    """Test loading from JSON file."""
    import json
    # Use a separate store to avoid caching conflicts
    store = LocalStore(tmp_path / "json_store")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([
            {"text": "Hello"},
            {"text": "World"}
        ], f)
        json_path = f.name
    
    try:
        ds = TextDataset.from_json(
            json_path,
            store,
            text_field="text",
            loading_strategy=LoadingStrategy.ITERABLE_ONLY  # Use streaming to avoid cache conflict
        )
        items = list(ds.iter_items())
        assert len(items) == 2
    finally:
        Path(json_path).unlink()


def test_text_dataset_from_local_directory(tmp_path):
    """Test loading from local directory of .txt files."""
    # Use a separate store to avoid caching conflicts
    store = LocalStore(tmp_path / "local_store")
    
    test_dir = tmp_path / "txt_files"
    test_dir.mkdir()
    (test_dir / "file1.txt").write_text("Content 1")
    (test_dir / "file2.txt").write_text("Content 2")
    
    ds = TextDataset.from_local(
        str(test_dir),
        store,
        loading_strategy=LoadingStrategy.ITERABLE_ONLY  # Use streaming to avoid cache conflict
    )
    items = list(ds.iter_items())
    assert len(items) == 2
    assert "Content 1" in items
    assert "Content 2" in items


def test_text_dataset_from_local_directory_recursive(temp_store):
    """Test loading from local directory recursively."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file2.txt").write_text("Content 2")
        
        ds = TextDataset.from_local(
            tmpdir,
            temp_store,
            recursive=True
        )
        assert len(ds) == 2


def test_text_dataset_from_local_directory_non_recursive(temp_store):
    """Test loading from local directory non-recursively."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file2.txt").write_text("Content 2")
        
        ds = TextDataset.from_local(
            tmpdir,
            temp_store,
            recursive=False
        )
        assert len(ds) == 1


def test_text_dataset_from_local_file_not_found(temp_store):
    """Test loading from non-existent file."""
    with pytest.raises(FileNotFoundError):
        TextDataset.from_local(
            "/nonexistent/path",
            temp_store
        )


def test_text_dataset_from_local_json_file(tmp_path):
    """Test loading from local JSON file."""
    import json
    store = LocalStore(tmp_path / "json_file_store")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([{"text": "Hello"}], f)
        json_path = f.name
    
    try:
        ds = TextDataset.from_local(
            json_path,
            store,
            loading_strategy=LoadingStrategy.ITERABLE_ONLY
        )
        items = list(ds.iter_items())
        assert len(items) == 1
    finally:
        Path(json_path).unlink()


def test_text_dataset_from_local_csv_file(tmp_path):
    """Test loading from local CSV file."""
    store = LocalStore(tmp_path / "csv_file_store")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("text\nHello\n")
        csv_path = f.name
    
    try:
        ds = TextDataset.from_local(
            csv_path,
            store,
            loading_strategy=LoadingStrategy.ITERABLE_ONLY
        )
        items = list(ds.iter_items())
        assert len(items) == 1
    finally:
        Path(csv_path).unlink()


def test_text_dataset_from_local_tsv_file(tmp_path):
    """Test loading from local TSV file."""
    store = LocalStore(tmp_path / "tsv_file_store")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write("text\nHello\n")
        tsv_path = f.name
    
    try:
        ds = TextDataset.from_local(
            tsv_path,
            store,
            loading_strategy=LoadingStrategy.ITERABLE_ONLY
        )
        items = list(ds.iter_items())
        assert len(items) == 1
    finally:
        Path(tsv_path).unlink()


def test_text_dataset_from_local_unsupported_file(tmp_path):
    """Test loading from unsupported file type."""
    store = LocalStore(tmp_path / "unsupported_store")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
        f.write("content")
        xyz_path = f.name
    
    try:
        with pytest.raises(ValueError, match="Unsupported"):
            TextDataset.from_local(
                xyz_path,
                store
            )
    finally:
        Path(xyz_path).unlink()


def test_text_dataset_removes_extra_columns(temp_store):
    """Test that extra columns are removed for memory efficiency."""
    ds = Dataset.from_dict({
        "text": ["a", "b"],
        "extra1": [1, 2],
        "extra2": ["x", "y"]
    })
    text_ds = TextDataset(ds, temp_store)
    # Should only have text column
    assert "text" in text_ds._ds.column_names
    assert "extra1" not in text_ds._ds.column_names
    assert "extra2" not in text_ds._ds.column_names


def test_text_dataset_renames_column(temp_store):
    """Test that custom text field is renamed to 'text'."""
    ds = Dataset.from_dict({"content": ["a", "b"]})
    text_ds = TextDataset(ds, temp_store, text_field="content")
    assert "text" in text_ds._ds.column_names
    assert "content" not in text_ds._ds.column_names

