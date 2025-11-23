from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, List, Optional, Union

from datasets import Dataset, IterableDataset, load_dataset, load_from_disk

from amber.datasets.loading_strategy import IndexLike, LoadingStrategy
from amber.store.store import Store


class BaseDataset(ABC):
    """
    Abstract base class for datasets with support for multiple sources,
    loading strategies, and Store integration.

    Loading Strategies:
    - MEMORY: Load entire dataset into memory (fastest random access, highest memory usage)
    - DYNAMIC_LOAD: Save to disk, read dynamically via memory-mapped Arrow files
      (supports len/getitem, lower memory usage)
    - ITERABLE_ONLY: True streaming mode using IterableDataset
      (lowest memory, no len/getitem support)
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

        Raises:
            ValueError: If store is None, loading_strategy is invalid, or dataset operations fail
            OSError: If file system operations fail
        """
        self._validate_initialization_params(store, loading_strategy)

        self._store = store
        self._loading_strategy = loading_strategy
        self._dataset_dir: Path = Path(store.base_path) / store.dataset_prefix

        is_iterable_input = isinstance(ds, IterableDataset)

        if loading_strategy == LoadingStrategy.MEMORY:
            # MEMORY: Convert to Dataset if needed, save to disk, load fully into memory
            self._is_iterable = False
            if is_iterable_input:
                ds = Dataset.from_generator(lambda: iter(ds))
            self._ds = self._save_and_load_dataset(ds, use_memory_mapping=False)
        elif loading_strategy == LoadingStrategy.DYNAMIC_LOAD:
            # DYNAMIC_LOAD: Save to disk, use memory-mapped Arrow files (supports len/getitem)
            self._is_iterable = False
            if is_iterable_input:
                ds = Dataset.from_generator(lambda: iter(ds))
            self._ds = self._save_and_load_dataset(ds, use_memory_mapping=True)
        elif loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            # ITERABLE_ONLY: Convert to IterableDataset, don't save to disk (no len/getitem)
            if not is_iterable_input:
                ds = ds.to_iterable_dataset()
            self._is_iterable = True
            self._ds = ds
            # Don't save to disk for iterable-only mode
        else:
            raise ValueError(
                f"Unknown loading strategy: {loading_strategy}. Must be one of: {[s.value for s in LoadingStrategy]}"
            )

    def _validate_initialization_params(self, store: Store, loading_strategy: LoadingStrategy) -> None:
        """Validate initialization parameters.

        Args:
            store: Store instance to validate
            loading_strategy: Loading strategy to validate

        Raises:
            ValueError: If store is None or loading_strategy is invalid
        """
        if store is None:
            raise ValueError("store cannot be None")

        if not isinstance(loading_strategy, LoadingStrategy):
            raise ValueError(f"loading_strategy must be a LoadingStrategy enum value, got: {type(loading_strategy)}")

    def _has_valid_dataset_dir(self) -> bool:
        """Check if dataset directory path is valid (non-empty base_path).

        Returns:
            True if base_path is not empty, False otherwise
        """
        return bool(self._store.base_path and str(self._store.base_path).strip())

    def _save_and_load_dataset(self, ds: Dataset, use_memory_mapping: bool = True) -> Dataset:
        """Save dataset to disk and load it back (with optional memory mapping).

        Args:
            ds: Dataset to save and load
            use_memory_mapping: Whether to use memory mapping (True for DYNAMIC_LOAD)

        Returns:
            Loaded dataset

        Raises:
            OSError: If file system operations fail
            RuntimeError: If dataset operations fail
        """
        if self._has_valid_dataset_dir():
            try:
                self._dataset_dir.mkdir(parents=True, exist_ok=True)
                ds.save_to_disk(str(self._dataset_dir))
                return load_from_disk(str(self._dataset_dir))
            except OSError as e:
                raise OSError(f"Failed to save/load dataset at {self._dataset_dir}. Error: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to process dataset at {self._dataset_dir}. Error: {e}") from e
        else:
            return ds

    def get_batch(self, start: int, batch_size: int) -> List[Any]:
        """
        Get a contiguous batch of items.

        Args:
            start: Starting index
            batch_size: Number of items to retrieve

        Returns:
            List of items

        Raises:
            NotImplementedError: If loading_strategy is ITERABLE_ONLY
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

        Args:
            n: Number of items to retrieve (default: 5)

        Returns:
            List of first n items
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

        Raises:
            NotImplementedError: If loading_strategy is ITERABLE_ONLY
        """
        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            raise NotImplementedError(
                "sample() not supported for ITERABLE_ONLY datasets. Use iter_items() and sample manually."
            )

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

    @abstractmethod
    def extract_texts_from_batch(self, batch: List[Any]) -> List[str]:
        """Extract text strings from a batch.

        Args:
            batch: A batch as returned by iter_batches()

        Returns:
            List of text strings ready for model inference
        """
        pass

    @abstractmethod
    def get_all_texts(self) -> List[str]:
        """Get all texts from the dataset.

        Returns:
            List of all text strings in the dataset

        Raises:
            NotImplementedError: If not supported for streaming datasets
        """
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

        Returns:
            BaseDataset instance

        Raises:
            ValueError: If repo_id is empty or store is None
            RuntimeError: If dataset loading fails
        """
        if not repo_id or not isinstance(repo_id, str) or not repo_id.strip():
            raise ValueError(f"repo_id must be a non-empty string, got: {repo_id!r}")

        if store is None:
            raise ValueError("store cannot be None")

        # Determine if we should use streaming for HuggingFace load_dataset
        # Only ITERABLE_ONLY needs streaming=True for load_dataset
        use_streaming = streaming if streaming is not None else (loading_strategy == LoadingStrategy.ITERABLE_ONLY)

        try:
            ds = load_dataset(
                path=repo_id,
                split=split,
                revision=revision,
                streaming=use_streaming,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset from HuggingFace Hub: repo_id={repo_id!r}, "
                f"split={split!r}, revision={revision!r}. Error: {e}"
            ) from e

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

        Returns:
            BaseDataset instance

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If store is None or source is invalid
            RuntimeError: If dataset loading fails
        """
        if store is None:
            raise ValueError("store cannot be None")

        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"CSV file not found: {source}")

        if not p.is_file():
            raise ValueError(f"Source must be a file, got: {source}")

        try:
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
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV dataset from {source}. Error: {e}") from e

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

        Returns:
            BaseDataset instance

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            ValueError: If store is None or source is invalid
            RuntimeError: If dataset loading fails
        """
        if store is None:
            raise ValueError("store cannot be None")

        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"JSON file not found: {source}")

        if not p.is_file():
            raise ValueError(f"Source must be a file, got: {source}")

        try:
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
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON dataset from {source}. Error: {e}") from e

        return cls(ds, store=store, loading_strategy=loading_strategy)
