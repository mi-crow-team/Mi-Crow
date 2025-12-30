"""Dataset fixtures for testing."""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from datasets import Dataset, IterableDataset

from mi_crow.datasets.text_dataset import TextDataset
from mi_crow.datasets.classification_dataset import ClassificationDataset
from mi_crow.datasets.loading_strategy import LoadingStrategy
from mi_crow.store.store import Store


def create_sample_dataset(
    num_samples: int = 5,
    text_field: str = "text",
    category_field: str = "category",
    include_categories: bool = False,
) -> Dataset:
    """
    Create a sample HuggingFace Dataset for testing.
    
    Args:
        num_samples: Number of samples
        text_field: Name of text field
        category_field: Name of category field
        include_categories: Whether to include category field
        
    Returns:
        Dataset instance
    """
    texts = [f"Sample text {i}" for i in range(num_samples)]
    data = {text_field: texts}
    
    if include_categories:
        categories = [f"cat_{i % 3}" for i in range(num_samples)]
        data[category_field] = categories
    
    return Dataset.from_dict(data)


def create_text_dataset(
    store: Store,
    texts: Optional[List[str]] = None,
    loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
    text_field: str = "text",
) -> TextDataset:
    """
    Create a TextDataset for testing.
    
    Args:
        store: Store instance
        texts: Optional list of texts (defaults to sample texts)
        loading_strategy: Loading strategy
        text_field: Name of text field
        
    Returns:
        TextDataset instance
    """
    if texts is None:
        texts = ["Hello world", "Test text", "Another example"]
    
    ds = Dataset.from_dict({text_field: texts})
    return TextDataset(ds, store, loading_strategy=loading_strategy, text_field=text_field)


def create_classification_dataset(
    store: Store,
    texts: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
    text_field: str = "text",
    category_field: str = "category",
) -> ClassificationDataset:
    """
    Create a ClassificationDataset for testing.
    
    Args:
        store: Store instance
        texts: Optional list of texts
        categories: Optional list of categories
        loading_strategy: Loading strategy
        text_field: Name of text field
        category_field: Name of category field
        
    Returns:
        ClassificationDataset instance
    """
    if texts is None:
        texts = ["Text 1", "Text 2", "Text 3"]
    if categories is None:
        categories = ["cat_a", "cat_b", "cat_a"]
    
    ds = Dataset.from_dict({
        text_field: texts,
        category_field: categories,
    })
    return ClassificationDataset(
        ds,
        store,
        loading_strategy=loading_strategy,
        text_field=text_field,
        category_field=category_field,
    )

