from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from datasets import Dataset, IterableDataset, load_dataset, load_from_disk

from mi_crow.datasets.base_dataset import BaseDataset
from mi_crow.datasets.loading_strategy import IndexLike, LoadingStrategy
from mi_crow.store.store import Store


class TextDataset(BaseDataset):
    """
    Text-only dataset with support for multiple sources and loading strategies.
    Each item is a string (text snippet).
    """

    def __init__(
        self,
        ds: Dataset | IterableDataset,
        store: Store,
        loading_strategy: LoadingStrategy = LoadingStrategy.DISK,
        text_field: str = "text",
    ):
        """
        Initialize text dataset.

        Args:
            ds: HuggingFace Dataset or IterableDataset
            store: Store instance
            loading_strategy: Loading strategy
            text_field: Name of the column containing text

        Raises:
            ValueError: If text_field is empty or not found in dataset
        """
        self._validate_text_field(text_field)

        # Validate and prepare dataset
        is_iterable = isinstance(ds, IterableDataset)
        if not is_iterable:
            if text_field not in ds.column_names:
                raise ValueError(f"Dataset must have a '{text_field}' column; got columns: {ds.column_names}")
            # Keep only text column for memory efficiency
            columns_to_remove = [c for c in ds.column_names if c != text_field]
            if columns_to_remove:
                ds = ds.remove_columns(columns_to_remove)
            if text_field != "text":
                ds = ds.rename_column(text_field, "text")
            ds.set_format("python", columns=["text"])

        self._text_field = text_field
        super().__init__(ds, store=store, loading_strategy=loading_strategy)

    def _validate_text_field(self, text_field: str) -> None:
        """Validate text_field parameter.

        Args:
            text_field: Text field name to validate

        Raises:
            ValueError: If text_field is empty or not a string
        """
        if not text_field or not isinstance(text_field, str) or not text_field.strip():
            raise ValueError(f"text_field must be a non-empty string, got: {text_field!r}")

    def _extract_text_from_row(self, row: Dict[str, Any]) -> Optional[str]:
        """Extract text from a dataset row.

        Args:
            row: Dataset row dictionary

        Returns:
            Text string from the row

        Raises:
            ValueError: If text field is not found in row
        """
        if self._text_field in row:
            text = row[self._text_field]
        elif "text" in row:
            text = row["text"]
        else:
            raise ValueError(
                f"Text field '{self._text_field}' or 'text' not found in dataset row. "
                f"Available fields: {list(row.keys())}"
            )
        return text

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Raises:
            NotImplementedError: If loading_strategy is STREAMING
        """
        if self._loading_strategy == LoadingStrategy.STREAMING:
            raise NotImplementedError("len() not supported for STREAMING datasets")
        return self._ds.num_rows

    def __getitem__(self, idx: IndexLike) -> Union[Optional[str], List[Optional[str]]]:
        """
        Get text item(s) by index.

        Args:
            idx: Index (int), slice, or sequence of indices

        Returns:
            Single text string or list of text strings

        Raises:
            NotImplementedError: If loading_strategy is STREAMING
            IndexError: If index is out of bounds
            ValueError: If dataset is empty
        """
        if self._loading_strategy == LoadingStrategy.STREAMING:
            raise NotImplementedError("Indexing not supported for STREAMING datasets. Use iter_items or iter_batches.")

        dataset_len = len(self)
        if dataset_len == 0:
            raise ValueError("Cannot index into empty dataset")

        if isinstance(idx, int):
            if idx < 0:
                idx = dataset_len + idx
            if idx < 0 or idx >= dataset_len:
                raise IndexError(f"Index {idx} out of bounds for dataset of length {dataset_len}")
            return self._ds[idx]["text"]

        if isinstance(idx, slice):
            start, stop, step = idx.indices(dataset_len)
            if step != 1:
                indices = list(range(start, stop, step))
                out = self._ds.select(indices)["text"]
            else:
                out = self._ds.select(range(start, stop))["text"]
            return list(out)

        if isinstance(idx, Sequence):
            # Validate all indices are in bounds
            invalid_indices = [i for i in idx if not (0 <= i < dataset_len)]
            if invalid_indices:
                raise IndexError(f"Indices out of bounds: {invalid_indices} (dataset length: {dataset_len})")
            out = self._ds.select(list(idx))["text"]
            return list(out)

        raise TypeError(f"Invalid index type: {type(idx)}")

    def iter_items(self) -> Iterator[Optional[str]]:
        """
        Iterate over text items one by one.

        Yields:
            Text strings from the dataset

        Raises:
            ValueError: If text field is not found in any row
        """
        for row in self._ds:
            yield self._extract_text_from_row(row)

    def iter_batches(self, batch_size: int) -> Iterator[List[Optional[str]]]:
        """
        Iterate over text items in batches.

        Args:
            batch_size: Number of items per batch

        Yields:
            Lists of text strings (batches)

        Raises:
            ValueError: If batch_size <= 0 or text field is not found in any row
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got: {batch_size}")

        if self._loading_strategy == LoadingStrategy.STREAMING:
            batch = []
            for row in self._ds:
                batch.append(self._extract_text_from_row(row))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        else:
            for batch in self._ds.iter(batch_size=batch_size):
                yield list(batch["text"])

    def extract_texts_from_batch(self, batch: List[Optional[str]]) -> List[Optional[str]]:
        """Extract text strings from a batch.

        For TextDataset, batch items are already strings, so return as-is.

        Args:
            batch: List of text strings

        Returns:
            List of text strings (same as input)
        """
        return batch

    def get_all_texts(self) -> List[Optional[str]]:
        """Get all texts from the dataset.

        Returns:
            List of all text strings

        Raises:
            NotImplementedError: If loading_strategy is STREAMING
        """
        if self._loading_strategy == LoadingStrategy.STREAMING:
            return list(self.iter_items())
        return list(self._ds["text"])

    def random_sample(self, n: int, seed: Optional[int] = None) -> "TextDataset":
        """Create a new TextDataset with n randomly sampled items.

        Args:
            n: Number of items to sample
            seed: Optional random seed for reproducibility

        Returns:
            New TextDataset instance with sampled items

        Raises:
            NotImplementedError: If loading_strategy is STREAMING
            ValueError: If n <= 0
        """
        if self._loading_strategy == LoadingStrategy.STREAMING:
            raise NotImplementedError(
                "random_sample() not supported for STREAMING datasets. Use iter_items() and sample manually."
            )
        
        if n <= 0:
            raise ValueError(f"n must be > 0, got: {n}")
        
        dataset_len = len(self)
        if n >= dataset_len:
            if seed is not None:
                random.seed(seed)
            indices = list(range(dataset_len))
            random.shuffle(indices)
            sampled_ds = self._ds.select(indices)
        else:
            if seed is not None:
                random.seed(seed)
            indices = random.sample(range(dataset_len), n)
            sampled_ds = self._ds.select(indices)
        
        return TextDataset(
            sampled_ds,
            store=self._store,
            loading_strategy=self._loading_strategy,
            text_field=self._text_field,
        )

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str,
        store: Store,
        *,
        split: str = "train",
        loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
        revision: Optional[str] = None,
        text_field: str = "text",
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        stratify_by: Optional[str] = None,
        stratify_seed: Optional[int] = None,
        streaming: Optional[bool] = None,
        drop_na: bool = False,
        **kwargs: Any,
    ) -> "TextDataset":
        """
        Load text dataset from HuggingFace Hub.

        Args:
            repo_id: HuggingFace dataset repository ID
            store: Store instance
            split: Dataset split
            loading_strategy: Loading strategy
            revision: Optional git revision
            text_field: Name of the column containing text
            filters: Optional filters to apply (dict of column: value)
            limit: Optional limit on number of rows
            stratify_by: Optional column used for stratified sampling (non-streaming only)
            stratify_seed: Optional RNG seed for deterministic stratification
            streaming: Optional override for streaming
            drop_na: Whether to drop rows with None/empty text
            **kwargs: Additional arguments for load_dataset

        Returns:
            TextDataset instance

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If dataset loading fails
        """
        use_streaming = streaming if streaming is not None else (loading_strategy == LoadingStrategy.STREAMING)

        if (stratify_by or drop_na) and use_streaming:
            raise NotImplementedError(
                "Stratification and drop_na are not supported for streaming datasets. Use MEMORY or DISK."
            )

        try:
            ds = load_dataset(
                path=repo_id,
                split=split,
                revision=revision,
                streaming=use_streaming,
                **kwargs,
            )

            if use_streaming:
                if filters or limit:
                    raise NotImplementedError(
                        "filters and limit are not supported when streaming datasets. Choose MEMORY or DISK."
                    )
            else:
                drop_na_columns = [text_field] if drop_na else None
                ds = cls._postprocess_non_streaming_dataset(
                    ds,
                    filters=filters,
                    limit=limit,
                    stratify_by=stratify_by,
                    stratify_seed=stratify_seed,
                    drop_na_columns=drop_na_columns,
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load text dataset from HuggingFace Hub: "
                f"repo_id={repo_id!r}, split={split!r}, text_field={text_field!r}. "
                f"Error: {e}"
            ) from e

        return cls(ds, store=store, loading_strategy=loading_strategy, text_field=text_field)

    @classmethod
    def from_disk(
        cls,
        store: Store,
        *,
        loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
        text_field: str = "text",
    ) -> "TextDataset":
        """
        Load text dataset from already-saved Arrow files on disk.

        Use this when you've previously saved a dataset and want to reload it
        without re-downloading from HuggingFace or re-applying transformations.

        Args:
            store: Store instance pointing to where the dataset was saved
            loading_strategy: Loading strategy (MEMORY or DISK only)
            text_field: Name of the column containing text

        Returns:
            TextDataset instance loaded from disk

        Raises:
            FileNotFoundError: If dataset directory doesn't exist or contains no Arrow files

        Example:
            # First: save dataset
            dataset_store = LocalStore("store/my_texts")
            dataset = TextDataset.from_huggingface(
                "wikipedia",
                store=dataset_store,
                limit=1000
            )
            # Dataset saved to: store/my_texts/datasets/*.arrow

            # Later: reload from disk
            dataset_store = LocalStore("store/my_texts")
            dataset = TextDataset.from_disk(store=dataset_store)
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

        # Create TextDataset with the loaded dataset and field name
        return cls(
            ds,
            store=store,
            loading_strategy=loading_strategy,
            text_field=text_field,
        )

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
        drop_na: bool = False,
        **kwargs: Any,
    ) -> "TextDataset":
        """
        Load text dataset from CSV file.

        Args:
            source: Path to CSV file
            store: Store instance
            loading_strategy: Loading strategy
            text_field: Name of the column containing text
            delimiter: CSV delimiter (default: comma)
            stratify_by: Optional column to use for stratified sampling
            stratify_seed: Optional RNG seed for stratified sampling
            drop_na: Whether to drop rows with None/empty text
            **kwargs: Additional arguments for load_dataset

        Returns:
            TextDataset instance

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            RuntimeError: If dataset loading fails
        """
        if store is None:
            raise ValueError("store cannot be None")

        use_streaming = loading_strategy == LoadingStrategy.STREAMING
        if (stratify_by or drop_na) and use_streaming:
            raise NotImplementedError("Stratification and drop_na are not supported for STREAMING datasets.")

        # Load CSV using parent's static method
        ds = cls._load_csv_source(
            source,
            delimiter=delimiter,
            streaming=use_streaming,
            **kwargs,
        )

        # Apply postprocessing if not streaming
        if not use_streaming and (stratify_by or drop_na):
            drop_na_columns = [text_field] if drop_na else None
            ds = cls._postprocess_non_streaming_dataset(
                ds,
                stratify_by=stratify_by,
                stratify_seed=stratify_seed,
                drop_na_columns=drop_na_columns,
            )

        return cls(
            ds,
            store=store,
            loading_strategy=loading_strategy,
            text_field=text_field,
        )

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
        drop_na: bool = False,
        **kwargs: Any,
    ) -> "TextDataset":
        """
        Load text dataset from JSON/JSONL file.

        Args:
            source: Path to JSON or JSONL file
            store: Store instance
            loading_strategy: Loading strategy
            text_field: Name of the field containing text
            stratify_by: Optional column to use for stratified sampling
            stratify_seed: Optional RNG seed for stratified sampling
            drop_na: Whether to drop rows with None/empty text
            **kwargs: Additional arguments for load_dataset

        Returns:
            TextDataset instance

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            RuntimeError: If dataset loading fails
        """
        if store is None:
            raise ValueError("store cannot be None")

        use_streaming = loading_strategy == LoadingStrategy.STREAMING
        if (stratify_by or drop_na) and use_streaming:
            raise NotImplementedError("Stratification and drop_na are not supported for STREAMING datasets.")

        # Load JSON using parent's static method
        ds = cls._load_json_source(
            source,
            streaming=use_streaming,
            **kwargs,
        )

        # Apply postprocessing if not streaming
        if not use_streaming and (stratify_by or drop_na):
            drop_na_columns = [text_field] if drop_na else None
            ds = cls._postprocess_non_streaming_dataset(
                ds,
                stratify_by=stratify_by,
                stratify_seed=stratify_seed,
                drop_na_columns=drop_na_columns,
            )

        return cls(
            ds,
            store=store,
            loading_strategy=loading_strategy,
            text_field=text_field,
        )

    @classmethod
    def from_local(
        cls,
        source: Union[str, Path],
        store: Store,
        *,
        loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
        text_field: str = "text",
        recursive: bool = True,
    ) -> "TextDataset":
        """
        Load from a local directory or file(s).

        Supported:
          - Directory of .txt files (each file becomes one example)
          - JSONL/JSON/CSV/TSV files with a text column

        Args:
            source: Path to directory or file
            store: Store instance
            loading_strategy: Loading strategy
            text_field: Name of the column/field containing text
            recursive: Whether to recursively search directories for .txt files

        Returns:
            TextDataset instance

        Raises:
            FileNotFoundError: If source path doesn't exist
            ValueError: If source is invalid or unsupported file type
            RuntimeError: If file operations fail
        """
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Source path does not exist: {source}")

        if p.is_dir():
            txts: List[str] = []
            pattern = "**/*.txt" if recursive else "*.txt"
            try:
                for fp in sorted(p.glob(pattern)):
                    txts.append(fp.read_text(encoding="utf-8", errors="ignore"))
            except OSError as e:
                raise RuntimeError(f"Failed to read text files from directory {source}. Error: {e}") from e

            if not txts:
                raise ValueError(f"No .txt files found in directory: {source} (recursive={recursive})")

            ds = Dataset.from_dict({"text": txts})
        else:
            suffix = p.suffix.lower()
            if suffix in {".jsonl", ".json"}:
                return cls.from_json(
                    source,
                    store=store,
                    loading_strategy=loading_strategy,
                    text_field=text_field,
                )
            elif suffix in {".csv"}:
                return cls.from_csv(
                    source,
                    store=store,
                    loading_strategy=loading_strategy,
                    text_field=text_field,
                )
            elif suffix in {".tsv"}:
                return cls.from_csv(
                    source,
                    store=store,
                    loading_strategy=loading_strategy,
                    text_field=text_field,
                    delimiter="\t",
                )
            else:
                raise ValueError(
                    f"Unsupported file type: {suffix} for source: {source}. "
                    f"Use directory of .txt, or JSON/JSONL/CSV/TSV."
                )

        return cls(ds, store=store, loading_strategy=loading_strategy, text_field=text_field)
