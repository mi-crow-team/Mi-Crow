from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Union, Optional, Any

from datasets import Dataset, load_dataset, load_from_disk, IterableDataset

from amber.store import Store
from amber.adapters.loading_strategy import LoadingStrategy, IndexLike


class BaseDataset(ABC):
    """
    Abstract base class for datasets with support for multiple sources,
    loading strategies, and Store integration.
    """

    def __init__(
        self,
        ds: Dataset | IterableDataset,
        store: Optional[Store] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
    ):
        """
        Initialize dataset.

        Args:
            ds: HuggingFace Dataset or IterableDataset
            store: Optional Store instance for caching/persistence
            cache_dir: Optional cache directory path (used if store is None)
            loading_strategy: How to load data (STREAM or MEMORY)
        """
        self._store = store
        self._loading_strategy = loading_strategy
        self._cache_dir: Optional[Path] = None

        # Determine cache directory
        if store is not None:
            if hasattr(store, "base_path"):
                self._cache_dir = Path(store.base_path) / "datasets"
            else:
                self._cache_dir = Path(cache_dir) if cache_dir else None
        else:
            self._cache_dir = Path(cache_dir) if cache_dir else None

        # Handle loading strategy
        is_iterable_input = isinstance(ds, IterableDataset)
        if loading_strategy == LoadingStrategy.MEMORY and is_iterable_input:
            # Convert IterableDataset to regular Dataset for memory loading
            ds = Dataset.from_generator(lambda: iter(ds))
            self._is_iterable = False
        elif loading_strategy == LoadingStrategy.STREAM and not is_iterable_input:
            # Convert Dataset to IterableDataset for streaming
            ds = ds.to_iterable_dataset()
            self._is_iterable = True
        else:
            self._is_iterable = is_iterable_input

        self._ds = ds

        # Cache to disk if cache_dir is provided and not streaming
        if self._cache_dir and not self._is_iterable:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._ds.save_to_disk(str(self._cache_dir))
            self._ds = load_from_disk(str(self._cache_dir))

    @property
    def is_streaming(self) -> bool:
        """Whether this dataset is streaming (IterableDataset)."""
        return self._is_iterable

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: IndexLike) -> Any:
        """Get item(s) by index."""
        pass

    @abstractmethod
    def iter_items(self) -> Iterator[Any]:
        """Iterate over items one by one."""
        pass

    @abstractmethod
    def iter_batches(self, batch_size: int) -> Iterator[List[Any]]:
        """Iterate over items in batches."""
        pass

    # --- Factory methods ---

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str,
        *,
        split: str = "train",
        cache_dir: Optional[Union[str, Path]] = None,
        store: Optional[Store] = None,
        loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
        revision: Optional[str] = None,
        streaming: Optional[bool] = None,
        **kwargs,
    ) -> "BaseDataset":
        """
        Load dataset from HuggingFace Hub.

        Args:
            repo_id: HuggingFace dataset repository ID
            split: Dataset split (e.g., "train", "validation")
            cache_dir: Optional cache directory
            store: Optional Store instance
            loading_strategy: Loading strategy (STREAM or MEMORY)
            revision: Optional git revision/branch/tag
            streaming: Optional override for streaming (if None, uses loading_strategy)
            **kwargs: Additional arguments passed to load_dataset
        """
        # Determine if we should use streaming
        use_streaming = streaming if streaming is not None else (loading_strategy == LoadingStrategy.STREAM)

        ds = load_dataset(
            path=repo_id,
            split=split,
            revision=revision,
            streaming=use_streaming,
            **kwargs,
        )

        return cls(ds, store=store, cache_dir=cache_dir, loading_strategy=loading_strategy)

    @classmethod
    def from_csv(
        cls,
        source: Union[str, Path],
        *,
        cache_dir: Optional[Union[str, Path]] = None,
        store: Optional[Store] = None,
        loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
        text_field: str = "text",
        delimiter: str = ",",
        **kwargs,
    ) -> "BaseDataset":
        """
        Load dataset from CSV file.

        Args:
            source: Path to CSV file
            cache_dir: Optional cache directory
            store: Optional Store instance
            loading_strategy: Loading strategy
            text_field: Name of the column containing text
            delimiter: CSV delimiter (default: comma)
            **kwargs: Additional arguments passed to load_dataset
        """
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"CSV file not found: {source}")

        if loading_strategy == LoadingStrategy.STREAM:
            ds = load_dataset(
                "csv",
                data_files=str(p),
                split="train",
                delimiter=delimiter,
                streaming=True,
                **kwargs,
            )
        else:
            ds = load_dataset(
                "csv",
                data_files=str(p),
                split="train",
                delimiter=delimiter,
                **kwargs,
            )

        return cls(ds, store=store, cache_dir=cache_dir, loading_strategy=loading_strategy)

    @classmethod
    def from_json(
        cls,
        source: Union[str, Path],
        *,
        cache_dir: Optional[Union[str, Path]] = None,
        store: Optional[Store] = None,
        loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
        text_field: str = "text",
        **kwargs,
    ) -> "BaseDataset":
        """
        Load dataset from JSON or JSONL file.

        Args:
            source: Path to JSON or JSONL file
            cache_dir: Optional cache directory
            store: Optional Store instance
            loading_strategy: Loading strategy
            text_field: Name of the field containing text (for JSON objects)
            **kwargs: Additional arguments passed to load_dataset
        """
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"JSON file not found: {source}")

        if loading_strategy == LoadingStrategy.STREAM:
            ds = load_dataset(
                "json",
                data_files=str(p),
                split="train",
                streaming=True,
                **kwargs,
            )
        else:
            ds = load_dataset(
                "json",
                data_files=str(p),
                split="train",
                **kwargs,
            )

        return cls(ds, store=store, cache_dir=cache_dir, loading_strategy=loading_strategy)

    def get_batch(self, start: int, batch_size: int) -> List[Any]:
        """
        Get a contiguous batch of items.

        Args:
            start: Starting index
            batch_size: Number of items to retrieve

        Returns:
            List of items
        """
        if self._is_iterable:
            raise NotImplementedError("get_batch not supported for streaming datasets. Use iter_batches instead.")
        if batch_size <= 0:
            return []
        end = min(start + batch_size, len(self))
        if start >= end:
            return []
        return self[start:end]

    def head(self, n: int = 5) -> List[Any]:
        """Get first n items."""
        if self._is_iterable:
            items = []
            for i, item in enumerate(self.iter_items()):
                if i >= n:
                    break
                items.append(item)
            return items
        return self[:n]

