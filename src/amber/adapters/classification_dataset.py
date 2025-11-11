from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Sequence, Union, Optional, Dict, Any

from datasets import Dataset, load_dataset, IterableDataset

from amber.store.store import Store
from amber.adapters.base_dataset import BaseDataset
from amber.adapters.loading_strategy import LoadingStrategy, IndexLike


class ClassificationDataset(BaseDataset):
    """
    Classification dataset with text and category/label columns.
    Each item is a tuple of (text, category) or a dict with 'text' and 'category' keys.
    """

    def __init__(
            self,
            ds: Dataset | IterableDataset,
            store: Store,
            loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
            text_field: str = "text",
            category_field: str = "category",
    ):
        """
        Initialize classification dataset.

        Args:
            ds: HuggingFace Dataset or IterableDataset
            store: Store instance
            loading_strategy: Loading strategy
            text_field: Name of the column containing text
            category_field: Name of the column containing category/label
        """
        # Validate dataset
        is_iterable = isinstance(ds, IterableDataset)
        if not is_iterable:
            if text_field not in ds.column_names:
                raise ValueError(f"Dataset must have a '{text_field}' column; got {ds.column_names}")
            if category_field not in ds.column_names:
                raise ValueError(f"Dataset must have a '{category_field}' column; got {ds.column_names}")
            ds.set_format("python", columns=[text_field, category_field])

        self._text_field = text_field
        self._category_field = category_field
        super().__init__(ds, store=store, loading_strategy=loading_strategy)

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        if self._is_iterable:
            raise NotImplementedError("len() not supported for streaming datasets")
        return self._ds.num_rows

    def __getitem__(self, idx: IndexLike) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get item(s) by index. Returns dict with 'text' and 'category' keys."""
        if self._is_iterable:
            raise NotImplementedError("Indexing not supported for streaming datasets. Use iter_items or iter_batches.")

        if isinstance(idx, int):
            row = self._ds[idx]
            return {"text": row[self._text_field], "category": row[self._category_field]}
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                indices = list(range(start, stop, step))
                selected = self._ds.select(indices)
            else:
                selected = self._ds.select(range(start, stop))
            return [
                {"text": row[self._text_field], "category": row[self._category_field]}
                for row in selected
            ]
        if isinstance(idx, Sequence):
            selected = self._ds.select(list(idx))
            return [
                {"text": row[self._text_field], "category": row[self._category_field]}
                for row in selected
            ]
        raise TypeError(f"Invalid index type: {type(idx)}")

    def iter_items(self) -> Iterator[Dict[str, Any]]:
        """Iterate over items one by one. Yields dict with 'text' and 'category' keys."""
        for row in self._ds:
            text = row.get(self._text_field) or row.get("text")
            category = row.get(self._category_field) or row.get("category")
            if text is None:
                raise ValueError(f"Text field '{self._text_field}' or 'text' not found in dataset row")
            if category is None:
                raise ValueError(f"Category field '{self._category_field}' or 'category' not found in dataset row")
            yield {"text": text, "category": category}

    def iter_batches(self, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
        """Iterate over items in batches. Each batch is a list of dicts with 'text' and 'category' keys."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if self._is_iterable:
            batch = []
            for row in self._ds:
                text = row.get(self._text_field) or row.get("text")
                category = row.get(self._category_field) or row.get("category")
                if text is None:
                    raise ValueError(f"Text field '{self._text_field}' or 'text' not found in dataset row")
                if category is None:
                    raise ValueError(f"Category field '{self._category_field}' or 'category' not found in dataset row")
                batch.append({"text": text, "category": category})
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        else:
            for batch in self._ds.iter(batch_size=batch_size):
                batch_list = [
                    {"text": row[self._text_field], "category": row[self._category_field]}
                    for row in batch
                ]
                yield batch_list

    def get_categories(self) -> List[Any]:
        """Get list of unique categories in the dataset, excluding None values."""
        if self._is_iterable:
            categories = set()
            for item in self.iter_items():
                cat = item["category"]
                if cat is not None:
                    categories.add(cat)
            return sorted(list(categories))
        categories = [cat for cat in set(self._ds[self._category_field]) if cat is not None]
        return sorted(categories)

    def get_texts(self) -> List[str]:
        """Get all texts as a list."""
        if self._is_iterable:
            return [item["text"] for item in self.iter_items()]
        return list(self._ds[self._text_field])

    def get_categories_for_texts(self, texts: List[str]) -> List[Any]:
        """Get categories for given texts (if texts match dataset texts)."""
        if self._is_iterable:
            raise NotImplementedError("get_categories_for_texts not supported for streaming datasets")
        text_to_category = {
            row[self._text_field]: row[self._category_field]
            for row in self._ds
        }
        return [text_to_category.get(text) for text in texts]

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
            category_field: str = "category",
            filters: Optional[Dict[str, Any]] = None,
            limit: Optional[int] = None,
            streaming: Optional[bool] = None,
            **kwargs,
    ) -> "ClassificationDataset":
        """Load classification dataset from HuggingFace Hub."""
        use_streaming = streaming if streaming is not None else (loading_strategy == LoadingStrategy.STREAM)

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

        return cls(
            ds,
            store=store,
            loading_strategy=loading_strategy,
            text_field=text_field,
            category_field=category_field,
        )

    @classmethod
    def from_csv(
            cls,
            source: Union[str, Path],
            store: Store,
            *,
            loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
            text_field: str = "text",
            category_field: str = "category",
            delimiter: str = ",",
            **kwargs,
    ) -> "ClassificationDataset":
        """Load classification dataset from CSV file."""
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
            category_field=category_field,
        )

    @classmethod
    def from_json(
            cls,
            source: Union[str, Path],
            store: Store,
            *,
            loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
            text_field: str = "text",
            category_field: str = "category",
            **kwargs,
    ) -> "ClassificationDataset":
        """Load classification dataset from JSON/JSONL file."""
        dataset = super().from_json(
            source,
            store=store,
            loading_strategy=loading_strategy,
            text_field=text_field,
            **kwargs,
        )
        return cls(
            dataset._ds,
            store=store,
            loading_strategy=loading_strategy,
            text_field=text_field,
            category_field=category_field,
        )
