from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Union, Optional, Any

from datasets import Dataset, load_dataset, load_from_disk, IterableDataset

from amber.store.store import Store
from amber.datasets.loading_strategy import LoadingStrategy, IndexLike


class BaseDataset(ABC):
    """
    Abstract base class for datasets with support for multiple sources,
    loading strategies, and Store integration.
    """

    def __init__(
            self,
            ds: Dataset | IterableDataset,
            store: Store,
            loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
    ):
        """
        Initialize dataset.

        Args:
            ds: HuggingFace Dataset or IterableDataset
            store: Store instance for caching/persistence
            loading_strategy: How to load data (MEMORY, DYNAMIC_LOAD, or ITERABLE_ONLY)
        """
        self._store = store
        self._loading_strategy = loading_strategy
        self._dataset_dir: Path = Path(store.base_path) / store.dataset_prefix

        is_iterable_input = isinstance(ds, IterableDataset)

        if loading_strategy == LoadingStrategy.MEMORY:
            # MEMORY: Convert to Dataset if needed, save to disk, load fully into memory
            if is_iterable_input:
                ds = Dataset.from_generator(lambda: iter(ds))
            self._is_iterable = False
            if self._dataset_dir:
                self._dataset_dir.mkdir(parents=True, exist_ok=True)
                self._ds = ds
                self._ds.save_to_disk(str(self._dataset_dir))
                self._ds = load_from_disk(str(self._dataset_dir))
            else:
                self._ds = ds
        elif loading_strategy == LoadingStrategy.DYNAMIC_LOAD:
            # DYNAMIC_LOAD: Save to disk, use memory-mapped Arrow files (supports len/getitem)
            if is_iterable_input:
                ds = Dataset.from_generator(lambda: iter(ds))
            self._is_iterable = False
            if self._dataset_dir:
                self._dataset_dir.mkdir(parents=True, exist_ok=True)
                self._ds = ds
                self._ds.save_to_disk(str(self._dataset_dir))
                # Load from disk with memory-mapping (default behavior of load_from_disk)
                self._ds = load_from_disk(str(self._dataset_dir))
            else:
                self._ds = ds
        elif loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            # ITERABLE_ONLY: Convert to IterableDataset, don't save to disk (no len/getitem)
            if not is_iterable_input:
                ds = ds.to_iterable_dataset()
            self._is_iterable = True
            self._ds = ds
            # Don't save to disk for iterable-only mode
        else:
            raise ValueError(f"Unknown loading strategy: {loading_strategy}")

    def get_batch(self, start: int, batch_size: int) -> List[Any]:
        """
        Get a contiguous batch of items.

        Args:
            start: Starting index
            batch_size: Number of items to retrieve

        Returns:
            List of items
        """
        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            raise NotImplementedError("get_batch not supported for ITERABLE_ONLY datasets. Use iter_batches instead.")
        if batch_size <= 0:
            return []
        end = min(start + batch_size, len(self))
        if start >= end:
            return []
        return self[start:end]

    def head(self, n: int = 5) -> List[Any]:
        """
        Get first n items.

        Works for all loading strategies.
        """
        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            items = []
            for i, item in enumerate(self.iter_items()):
                if i >= n:
                    break
                items.append(item)
            return items
        return self[:n]

    def sample(self, n: int = 5) -> List[Any]:
        """
        Get n random items from the dataset.

        Works for MEMORY and DYNAMIC_LOAD strategies only.

        Args:
            n: Number of items to sample

        Returns:
            List of n randomly sampled items
        """
        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            raise NotImplementedError(
                "sample() not supported for ITERABLE_ONLY datasets. Use iter_items() and sample manually.")

        dataset_len = len(self)
        if n <= 0:
            return []
        if n >= dataset_len:
            # Return all items in random order
            indices = list(range(dataset_len))
            random.shuffle(indices)
            return [self[i] for i in indices]

        # Sample n random indices
        indices = random.sample(range(dataset_len), n)
        # Use __getitem__ with list of indices
        return self[indices]

    @property
    def is_streaming(self) -> bool:
        """Whether this dataset is streaming (DYNAMIC_LOAD or ITERABLE_ONLY)."""
        return self._loading_strategy in (LoadingStrategy.DYNAMIC_LOAD, LoadingStrategy.ITERABLE_ONLY)

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
            store: Store,
            *,
            split: str = "train",
            loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
            revision: Optional[str] = None,
            streaming: Optional[bool] = None,
            **kwargs,
    ) -> "BaseDataset":
        """
        Load dataset from HuggingFace Hub.

        Args:
            repo_id: HuggingFace dataset repository ID
            store: Store instance
            split: Dataset split (e.g., "train", "validation")
            loading_strategy: Loading strategy (MEMORY, DYNAMIC_LOAD, or ITERABLE_ONLY)
            revision: Optional git revision/branch/tag
            streaming: Optional override for streaming (if None, uses loading_strategy)
            **kwargs: Additional arguments passed to load_dataset
        """
        # Determine if we should use streaming for HuggingFace load_dataset
        # Only ITERABLE_ONLY needs streaming=True for load_dataset
        use_streaming = streaming if streaming is not None else (loading_strategy == LoadingStrategy.ITERABLE_ONLY)

        ds = load_dataset(
            path=repo_id,
            split=split,
            revision=revision,
            streaming=use_streaming,
            **kwargs,
        )

        return cls(ds, store=store, loading_strategy=loading_strategy)

    @classmethod
    def from_csv(
            cls,
            source: Union[str, Path],
            store: Store,
            *,
            loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
            text_field: str = "text",
            delimiter: str = ",",
            **kwargs,
    ) -> "BaseDataset":
        """
        Load dataset from CSV file.

        Args:
            source: Path to CSV file
            store: Store instance
            loading_strategy: Loading strategy
            text_field: Name of the column containing text
            delimiter: CSV delimiter (default: comma)
            **kwargs: Additional arguments passed to load_dataset
        """
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"CSV file not found: {source}")

        if loading_strategy == LoadingStrategy.ITERABLE_ONLY:
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

        return cls(ds, store=store, loading_strategy=loading_strategy)

    @classmethod
    def from_json(
            cls,
            source: Union[str, Path],
            store: Store,
            *,
            loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
            text_field: str = "text",
            **kwargs,
    ) -> "BaseDataset":
        """
        Load dataset from JSON or JSONL file.

        Args:
            source: Path to JSON or JSONL file
            store: Store instance
            loading_strategy: Loading strategy
            text_field: Name of the field containing text (for JSON objects)
            **kwargs: Additional arguments passed to load_dataset
        """
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"JSON file not found: {source}")

        if loading_strategy == LoadingStrategy.ITERABLE_ONLY:
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

        return cls(ds, store=store, loading_strategy=loading_strategy)
