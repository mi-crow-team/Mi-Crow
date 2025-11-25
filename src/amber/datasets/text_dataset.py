from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from datasets import Dataset, IterableDataset, load_dataset

from amber.datasets.base_dataset import BaseDataset
from amber.datasets.loading_strategy import IndexLike, LoadingStrategy
from amber.store.store import Store


class TextDataset(BaseDataset):
    """
    Text-only dataset with support for multiple sources and loading strategies.
    Each item is a string (text snippet).
    """

    def __init__(
        self,
        ds: Dataset | IterableDataset,
        store: Store,
        loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
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

    def _extract_text_from_row(self, row: Dict[str, Any]) -> str:
        """Extract text from a dataset row.

        Args:
            row: Dataset row dictionary

        Returns:
            Text string from the row

        Raises:
            ValueError: If text field is not found in row
        """
        text = row.get(self._text_field) or row.get("text")
        if text is None:
            raise ValueError(
                f"Text field '{self._text_field}' or 'text' not found in dataset row. "
                f"Available fields: {list(row.keys())}"
            )
        return text

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Raises:
            NotImplementedError: If loading_strategy is ITERABLE_ONLY
        """
        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            raise NotImplementedError("len() not supported for ITERABLE_ONLY datasets")
        return self._ds.num_rows

    def __getitem__(self, idx: IndexLike) -> Union[str, List[str]]:
        """
        Get text item(s) by index.

        Args:
            idx: Index (int), slice, or sequence of indices

        Returns:
            Single text string or list of text strings

        Raises:
            NotImplementedError: If loading_strategy is ITERABLE_ONLY
            IndexError: If index is out of bounds
            ValueError: If dataset is empty
        """
        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            raise NotImplementedError(
                "Indexing not supported for ITERABLE_ONLY datasets. Use iter_items or iter_batches."
            )

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

    def iter_items(self) -> Iterator[str]:
        """
        Iterate over text items one by one.

        Yields:
            Text strings from the dataset

        Raises:
            ValueError: If text field is not found in any row
        """
        for row in self._ds:
            yield self._extract_text_from_row(row)

    def iter_batches(self, batch_size: int) -> Iterator[List[str]]:
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

        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
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

    def extract_texts_from_batch(self, batch: List[str]) -> List[str]:
        """Extract text strings from a batch.

        For TextDataset, batch items are already strings, so return as-is.

        Args:
            batch: List of text strings

        Returns:
            List of text strings (same as input)
        """
        return batch

    def get_all_texts(self) -> List[str]:
        """Get all texts from the dataset.

        Returns:
            List of all text strings

        Raises:
            NotImplementedError: If loading_strategy is ITERABLE_ONLY
        """
        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            return list(self.iter_items())
        return list(self._ds["text"])

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
        streaming: Optional[bool] = None,
        **kwargs,
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
            streaming: Optional override for streaming
            **kwargs: Additional arguments for load_dataset

        Returns:
            TextDataset instance

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If dataset loading fails
        """
        use_streaming = streaming if streaming is not None else (loading_strategy == LoadingStrategy.ITERABLE_ONLY)

        try:
            ds = load_dataset(
                path=repo_id,
                split=split,
                revision=revision,
                streaming=use_streaming,
                **kwargs,
            )

            if not use_streaming:
                if filters:

                    def _pred(example):
                        return all(example.get(k) == v for k, v in filters.items())

                    ds = ds.filter(_pred)

                if limit is not None:
                    if limit <= 0:
                        raise ValueError(f"limit must be > 0, got: {limit}")
                    ds = ds.select(range(min(limit, len(ds))))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load text dataset from HuggingFace Hub: "
                f"repo_id={repo_id!r}, split={split!r}, text_field={text_field!r}. "
                f"Error: {e}"
            ) from e

        return cls(ds, store=store, loading_strategy=loading_strategy, text_field=text_field)

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
    ) -> "TextDataset":
        """
        Load text dataset from CSV file.

        Args:
            source: Path to CSV file
            store: Store instance
            loading_strategy: Loading strategy
            text_field: Name of the column containing text
            delimiter: CSV delimiter (default: comma)
            **kwargs: Additional arguments for load_dataset

        Returns:
            TextDataset instance

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            RuntimeError: If dataset loading fails
        """
        dataset = super().from_csv(
            source,
            store=store,
            loading_strategy=loading_strategy,
            text_field=text_field,
            delimiter=delimiter,
            **kwargs,
        )
        return cls(
            dataset._ds,
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
        **kwargs,
    ) -> "TextDataset":
        """
        Load text dataset from JSON/JSONL file.

        Args:
            source: Path to JSON or JSONL file
            store: Store instance
            loading_strategy: Loading strategy
            text_field: Name of the field containing text
            **kwargs: Additional arguments for load_dataset

        Returns:
            TextDataset instance

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            RuntimeError: If dataset loading fails
        """
        dataset = super().from_json(
            source,
            store=store,
            loading_strategy=loading_strategy,
            text_field=text_field,
            **kwargs,
        )
        # Re-initialize with text_field
        return cls(
            dataset._ds,
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
