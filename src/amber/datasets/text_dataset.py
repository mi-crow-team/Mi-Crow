from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Sequence, Union, Optional, Dict, Any

from datasets import Dataset, load_dataset, IterableDataset

from amber.store.store import Store
from amber.datasets.base_dataset import BaseDataset
from amber.datasets.loading_strategy import LoadingStrategy, IndexLike


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
        """
        # Validate and prepare dataset
        is_iterable = isinstance(ds, IterableDataset)
        if not is_iterable:
            if text_field not in ds.column_names:
                raise ValueError(f"Dataset must have a '{text_field}' column; got {ds.column_names}")
            # Keep only text column for memory efficiency
            columns_to_remove = [c for c in ds.column_names if c != text_field]
            if columns_to_remove:
                ds = ds.remove_columns(columns_to_remove)
            if text_field != "text":
                ds = ds.rename_column(text_field, "text")
            ds.set_format("python", columns=["text"])

        self._text_field = text_field
        super().__init__(ds, store=store, loading_strategy=loading_strategy)

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            raise NotImplementedError("len() not supported for ITERABLE_ONLY datasets")
        return self._ds.num_rows

    def __getitem__(self, idx: IndexLike) -> Union[str, List[str]]:
        """Get text item(s) by index."""
        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            raise NotImplementedError("Indexing not supported for ITERABLE_ONLY datasets. Use iter_items or iter_batches.")

        if isinstance(idx, int):
            return self._ds[idx]["text"]
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                indices = list(range(start, stop, step))
                out = self._ds.select(indices)["text"]
            else:
                out = self._ds.select(range(start, stop))["text"]
            return list(out)
        if isinstance(idx, Sequence):
            out = self._ds.select(list(idx))["text"]
            return list(out)

    def iter_items(self) -> Iterator[str]:
        """Iterate over text items one by one."""
        for row in self._ds:
            text = row.get(self._text_field) or row.get("text")
            if text is None:
                raise ValueError(f"Text field '{self._text_field}' or 'text' not found in dataset row")
            yield text

    def iter_batches(self, batch_size: int) -> Iterator[List[str]]:
        """Iterate over text items in batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            batch = []
            for row in self._ds:
                text = row.get(self._text_field) or row.get("text")
                if text is None:
                    raise ValueError(f"Text field '{self._text_field}' or 'text' not found in dataset row")
                batch.append(text)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        else:
            for batch in self._ds.iter(batch_size=batch_size):
                yield list(batch["text"])

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
        """
        use_streaming = streaming if streaming is not None else (loading_strategy == LoadingStrategy.ITERABLE_ONLY)

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
                ds = ds.select(range(min(limit, len(ds))))

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
        """Load text dataset from CSV file."""
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
        """Load text dataset from JSON/JSONL file."""
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
        """
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(source)

        if p.is_dir():
            txts: List[str] = []
            pattern = "**/*.txt" if recursive else "*.txt"
            for fp in sorted(p.glob(pattern)):
                txts.append(fp.read_text(encoding="utf-8", errors="ignore"))
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
                    f"Unsupported file type: {suffix}. "
                    f"Use directory of .txt, or JSON/JSONL/CSV/TSV."
                )

        return cls(ds, store=store, loading_strategy=loading_strategy, text_field=text_field)
