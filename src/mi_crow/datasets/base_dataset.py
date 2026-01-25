from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from datasets import Dataset, IterableDataset, load_dataset, load_from_disk

from mi_crow.datasets.loading_strategy import IndexLike, LoadingStrategy
from mi_crow.store.store import Store


class BaseDataset(ABC):
    """
    Abstract base class for datasets with support for multiple sources,
    loading strategies, and Store integration.

    Loading Strategies:
    - MEMORY: Load entire dataset into memory (fastest random access, highest memory usage)
    - DISK: Save to disk, read dynamically via memory-mapped Arrow files
      (supports len/getitem, lower memory usage)
    - STREAMING: True streaming mode using IterableDataset
      (lowest memory, no len/getitem support, no stratification and limit support)
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
            loading_strategy: How to load data (MEMORY, DISK, or STREAMING)

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
        elif loading_strategy == LoadingStrategy.DISK:
            # DISK: Save to disk, use memory-mapped Arrow files (supports len/getitem)
            self._is_iterable = False
            if is_iterable_input:
                ds = Dataset.from_generator(lambda: iter(ds))
            self._ds = self._save_and_load_dataset(ds, use_memory_mapping=True)
        elif loading_strategy == LoadingStrategy.STREAMING:
            # STREAMING: Convert to IterableDataset, don't save to disk (no len/getitem)
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
            use_memory_mapping: Whether to use memory mapping (True for DISK)

        Returns:
            Loaded dataset

        Raises:
            OSError: If file system operations fail
            RuntimeError: If dataset operations fail
        """
        if len(ds) == 0:
            return ds
        
        if self._has_valid_dataset_dir():
            try:
                self._dataset_dir.mkdir(parents=True, exist_ok=True)
                ds.save_to_disk(str(self._dataset_dir))
                return load_from_disk(str(self._dataset_dir), keep_in_memory=not use_memory_mapping)
            except OSError as e:
                raise OSError(f"Failed to save/load dataset at {self._dataset_dir}. Error: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to process dataset at {self._dataset_dir}. Error: {e}") from e
        else:
            return ds

    @classmethod
    def _postprocess_non_streaming_dataset(
        cls,
        ds: Dataset,
        *,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        stratify_by: Optional[str] = None,
        stratify_seed: Optional[int] = None,
        drop_na_columns: Optional[List[str]] = None,
    ) -> Dataset:
        """Apply filters, stratified sampling, and limits to an in-memory dataset."""

        if drop_na_columns:
            ds = cls._drop_na(ds, drop_na_columns)

        if filters:
            ds = cls._apply_filters(ds, filters)

        limit_applied = False
        if stratify_by:
            sample_size = limit if limit is not None else len(ds)
            if sample_size is not None and sample_size <= 0:
                raise ValueError(f"limit must be > 0 when stratifying, got: {sample_size}")
            ds = cls._stratified_sample(
                ds,
                stratify_by=stratify_by,
                sample_size=sample_size,
                seed=stratify_seed,
            )
            limit_applied = True

        if limit is not None and not limit_applied:
            if limit <= 0:
                raise ValueError(f"limit must be > 0, got: {limit}")
            ds = ds.select(range(min(limit, len(ds))))

        return ds

    @staticmethod
    def _drop_na(ds: Dataset, columns: List[str]) -> Dataset:
        """Drop rows where any of the specified columns are None or empty string."""

        def _is_valid(example: Dict[str, Any]) -> bool:
            for col in columns:
                val = example.get(col)
                if val is None:
                    return False
                if isinstance(val, str) and not val.strip():
                    return False
            return True

        return ds.filter(_is_valid)

    @staticmethod
    def _apply_filters(ds: Dataset, filters: Dict[str, Any]) -> Dataset:
        """Apply exact-match filters to a Dataset."""

        def _predicate(example: Dict[str, Any]) -> bool:
            return all(example.get(key) == value for key, value in filters.items())

        return ds.filter(_predicate)

    @staticmethod
    def _stratified_sample(  # noqa: C901
        ds: Dataset,
        *,
        stratify_by: str,
        sample_size: Optional[int],
        seed: Optional[int],
    ) -> Dataset:
        """Return a stratified sample of the dataset with the requested size."""

        if stratify_by not in ds.column_names:
            raise ValueError(f"Column '{stratify_by}' not found in dataset columns: {ds.column_names}")

        total_rows = len(ds)
        if total_rows == 0:
            return ds

        if sample_size is None:
            sample_size = total_rows

        sample_size = min(sample_size, total_rows)
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than 0 for stratification")

        column_values = ds[stratify_by]
        label_to_indices: Dict[Any, List[int]] = defaultdict(list)
        for idx, label in enumerate(column_values):
            label_to_indices[label].append(idx)

        label_counts = {label: len(indices) for label, indices in label_to_indices.items()}
        allocations: Dict[Any, int] = {}
        fractional_parts: List[tuple[float, int, Any]] = []

        allocated_total = 0
        for order, (label, count) in enumerate(label_counts.items()):
            exact_allocation = (count / total_rows) * sample_size
            base_allocation = min(count, int(math.floor(exact_allocation)))
            allocations[label] = base_allocation
            allocated_total += base_allocation
            fractional_parts.append((exact_allocation - base_allocation, order, label))

        remaining = sample_size - allocated_total
        fractional_parts.sort(key=lambda item: (-item[0], item[1]))
        for _, _, label in fractional_parts:
            if remaining <= 0:
                break
            available = label_counts[label] - allocations[label]
            if available <= 0:
                continue
            take = min(available, remaining)
            allocations[label] += take
            remaining -= take

        rng = random.Random(seed)
        selected_indices: List[int] = []
        for label, count in allocations.items():
            if count <= 0:
                continue
            indices = label_to_indices[label]
            if count >= len(indices):
                chosen = list(indices)
            else:
                chosen = rng.sample(indices, count)
            selected_indices.extend(chosen)

        rng.shuffle(selected_indices)
        return ds.select(selected_indices)

    @staticmethod
    def _load_csv_source(
        source: Union[str, Path],
        *,
        delimiter: str,
        streaming: bool,
        **kwargs,
    ) -> Dataset | IterableDataset:
        """Load a CSV dataset from disk using HuggingFace datasets."""

        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"CSV file not found: {source}")
        if not p.is_file():
            raise ValueError(f"Source must be a file, got: {source}")

        try:
            return load_dataset(
                "csv",
                data_files=str(p),
                split="train",
                delimiter=delimiter,
                streaming=streaming,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV dataset from {source}. Error: {e}") from e

    @staticmethod
    def _load_json_source(
        source: Union[str, Path],
        *,
        streaming: bool,
        **kwargs,
    ) -> Dataset | IterableDataset:
        """Load a JSON/JSONL dataset from disk using HuggingFace datasets."""

        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"JSON file not found: {source}")
        if not p.is_file():
            raise ValueError(f"Source must be a file, got: {source}")

        try:
            return load_dataset(
                "json",
                data_files=str(p),
                split="train",
                streaming=streaming,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON dataset from {source}. Error: {e}") from e

    def get_batch(self, start: int, batch_size: int) -> List[Any]:
        """
        Get a contiguous batch of items.

        Args:
            start: Starting index
            batch_size: Number of items to retrieve

        Returns:
            List of items

        Raises:
            NotImplementedError: If loading_strategy is STREAMING
        """
        if self._loading_strategy == LoadingStrategy.STREAMING:
            raise NotImplementedError("get_batch not supported for STREAMING datasets. Use iter_batches instead.")
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
        if self._loading_strategy == LoadingStrategy.STREAMING:
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

        Works for MEMORY and DISK strategies only.

        Args:
            n: Number of items to sample

        Returns:
            List of n randomly sampled items

        Raises:
            NotImplementedError: If loading_strategy is STREAMING
        """
        if self._loading_strategy == LoadingStrategy.STREAMING:
            raise NotImplementedError(
                "sample() not supported for STREAMING datasets. Use iter_items() and sample manually."
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
        """Whether this dataset is streaming (DISK or STREAMING)."""
        return self._loading_strategy in (LoadingStrategy.DISK, LoadingStrategy.STREAMING)

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
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        stratify_by: Optional[str] = None,
        stratify_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "BaseDataset":
        """
        Load dataset from HuggingFace Hub.

        Args:
            repo_id: HuggingFace dataset repository ID
            store: Store instance
            split: Dataset split (e.g., "train", "validation")
            loading_strategy: Loading strategy (MEMORY, DISK, or STREAMING)
            revision: Optional git revision/branch/tag
            streaming: Optional override for streaming (if None, uses loading_strategy)
            filters: Optional dict of column->value pairs used for exact-match filtering
            limit: Optional maximum number of rows to keep (applied after filtering/stratification)
            stratify_by: Optional column to use for stratified sampling (non-streaming only)
            stratify_seed: Optional RNG seed for deterministic stratification
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
        use_streaming = streaming if streaming is not None else (loading_strategy == LoadingStrategy.STREAMING)

        if stratify_by and loading_strategy == LoadingStrategy.STREAMING:
            raise NotImplementedError("Stratification is not supported for STREAMING datasets.")

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

        if use_streaming:
            if filters or limit or stratify_by:
                raise NotImplementedError(
                    "filters, limit, and stratification are not supported when streaming datasets. "
                    "Choose MEMORY or DISK loading strategy instead."
                )
        else:
            ds = cls._postprocess_non_streaming_dataset(
                ds,
                filters=filters,
                limit=limit,
                stratify_by=stratify_by,
                stratify_seed=stratify_seed,
            )

        return cls(ds, store=store, loading_strategy=loading_strategy)

    @classmethod
    def from_disk(
        cls,
        store: Store,
        *,
        loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
        **kwargs: Any,
    ) -> "BaseDataset":
        """
        Load dataset from already-saved Arrow files on disk.

        Use this when you've previously saved a dataset and want to reload it
        without re-downloading from HuggingFace or re-applying transformations.

        Args:
            store: Store instance pointing to where the dataset was saved
                   (dataset will be loaded from store.base_path/store.dataset_prefix/)
            loading_strategy: Loading strategy (MEMORY or DISK only, not STREAMING)
            **kwargs: Additional arguments (for subclass compatibility)

        Returns:
            BaseDataset instance loaded from disk

        Raises:
            ValueError: If store is None or loading_strategy is STREAMING
            FileNotFoundError: If dataset directory doesn't exist
            RuntimeError: If dataset loading fails

        Example:
            # First: save dataset
            dataset_store = LocalStore("store/my_dataset")
            dataset = ClassificationDataset.from_huggingface(..., store=dataset_store)
            # Dataset saved to: store/my_dataset/datasets/*.arrow

            # Later: reload from disk
            dataset_store = LocalStore("store/my_dataset")
            dataset = ClassificationDataset.from_disk(store=dataset_store)
        """
        if store is None:
            raise ValueError("store cannot be None")

        if loading_strategy == LoadingStrategy.STREAMING:
            raise ValueError("STREAMING loading strategy not supported for from_disk(). Use MEMORY or DISK.")

        dataset_dir = Path(store.base_path) / store.dataset_prefix

        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_dir}. "
                f"Make sure you've previously saved a dataset to this store location."
            )

        # Verify it's a valid Arrow dataset directory
        arrow_files = list(dataset_dir.glob("*.arrow"))
        if not arrow_files:
            raise FileNotFoundError(
                f"No Arrow files found in {dataset_dir}. Directory exists but doesn't contain a valid dataset."
            )

        try:
            use_memory_mapping = loading_strategy == LoadingStrategy.DISK
            ds = load_from_disk(str(dataset_dir), keep_in_memory=not use_memory_mapping)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {dataset_dir}. Error: {e}") from e

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
        stratify_by: Optional[str] = None,
        stratify_seed: Optional[int] = None,
        drop_na_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "BaseDataset":
        """
        Load dataset from CSV file.

        Args:
            source: Path to CSV file
            store: Store instance
            loading_strategy: Loading strategy
            text_field: Name of the column containing text
            delimiter: CSV delimiter (default: comma)
            stratify_by: Optional column used for stratified sampling (non-streaming only)
            stratify_seed: Optional RNG seed for stratified sampling
            drop_na_columns: Optional list of columns to check for None/empty values
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

        use_streaming = loading_strategy == LoadingStrategy.STREAMING
        if (stratify_by or drop_na_columns) and use_streaming:
            raise NotImplementedError("Stratification and drop_na are not supported for STREAMING datasets.")

        ds = cls._load_csv_source(
            source,
            delimiter=delimiter,
            streaming=use_streaming,
            **kwargs,
        )

        if not use_streaming and (stratify_by or drop_na_columns):
            ds = cls._postprocess_non_streaming_dataset(
                ds,
                stratify_by=stratify_by,
                stratify_seed=stratify_seed,
                drop_na_columns=drop_na_columns,
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
        stratify_by: Optional[str] = None,
        stratify_seed: Optional[int] = None,
        drop_na_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "BaseDataset":
        """
        Load dataset from JSON or JSONL file.

        Args:
            source: Path to JSON or JSONL file
            store: Store instance
            loading_strategy: Loading strategy
            text_field: Name of the field containing text (for JSON objects)
            stratify_by: Optional column used for stratified sampling (non-streaming only)
            stratify_seed: Optional RNG seed for stratified sampling
            drop_na_columns: Optional list of columns to check for None/empty values
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

        use_streaming = loading_strategy == LoadingStrategy.STREAMING
        if (stratify_by or drop_na_columns) and use_streaming:
            raise NotImplementedError("Stratification and drop_na are not supported for STREAMING datasets.")

        ds = cls._load_json_source(
            source,
            streaming=use_streaming,
            **kwargs,
        )

        if not use_streaming and (stratify_by or drop_na_columns):
            ds = cls._postprocess_non_streaming_dataset(
                ds,
                stratify_by=stratify_by,
                stratify_seed=stratify_seed,
                drop_na_columns=drop_na_columns,
            )

        return cls(ds, store=store, loading_strategy=loading_strategy)
