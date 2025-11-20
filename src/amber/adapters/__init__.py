"""Compatibility layer: re-export datasets as adapters for backward compatibility."""
from amber.datasets.base_dataset import BaseDataset
from amber.datasets.text_dataset import TextDataset
from amber.datasets.classification_dataset import ClassificationDataset
from amber.datasets.loading_strategy import LoadingStrategy, IndexLike
from amber.datasets.text_snippet_dataset import TextSnippetDataset

__all__ = [
    "BaseDataset",
    "TextDataset",
    "ClassificationDataset",
    "LoadingStrategy",
    "IndexLike",
    "TextSnippetDataset",
]

