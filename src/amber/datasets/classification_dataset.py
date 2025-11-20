from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Sequence, Union, Optional, Dict, Any

from datasets import Dataset, load_dataset, IterableDataset

from amber.store.store import Store
from amber.datasets.base_dataset import BaseDataset
from amber.datasets.loading_strategy import LoadingStrategy, IndexLike


class ClassificationDataset(BaseDataset):
    """
    Classification dataset with text and category/label columns.
    Each item is a dict with 'text' and label column(s) as keys.
    Supports single or multiple label columns.
    """

    def __init__(
            self,
            ds: Dataset | IterableDataset,
            store: Store,
            loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
            text_field: str = "text",
            category_field: Union[str, List[str]] = "category",
    ):
        """
        Initialize classification dataset.

        Args:
            ds: HuggingFace Dataset or IterableDataset
            store: Store instance
            loading_strategy: Loading strategy
            text_field: Name of the column containing text
            category_field: Name(s) of the column(s) containing category/label.
                          Can be a single string or a list of strings for multiple labels.
        """
        # Normalize category_field to list
        if isinstance(category_field, str):
            self._category_fields = [category_field]
        else:
            self._category_fields = list(category_field)
        
        # Validate dataset
        is_iterable = isinstance(ds, IterableDataset)
        if not is_iterable:
            if text_field not in ds.column_names:
                raise ValueError(f"Dataset must have a '{text_field}' column; got {ds.column_names}")
            for cat_field in self._category_fields:
                if cat_field not in ds.column_names:
                    raise ValueError(f"Dataset must have a '{cat_field}' column; got {ds.column_names}")
            # Set format with all required columns
            format_columns = [text_field] + self._category_fields
            ds.set_format("python", columns=format_columns)

        self._text_field = text_field
        self._category_field = category_field  # Keep original for backward compatibility
        super().__init__(ds, store=store, loading_strategy=loading_strategy)

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            raise NotImplementedError("len() not supported for ITERABLE_ONLY datasets")
        return self._ds.num_rows

    def __getitem__(self, idx: IndexLike) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get item(s) by index. Returns dict with 'text' and label column(s) as keys.
        
        For single label: {"text": "...", "category": "..."}
        For multiple labels: {"text": "...", "label1": "...", "label2": "..."}
        """
        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            raise NotImplementedError("Indexing not supported for ITERABLE_ONLY datasets. Use iter_items or iter_batches.")

        def _make_item(row):
            item = {"text": row[self._text_field]}
            for cat_field in self._category_fields:
                item[cat_field] = row[cat_field]
            return item

        if isinstance(idx, int):
            row = self._ds[idx]
            return _make_item(row)
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                indices = list(range(start, stop, step))
                selected = self._ds.select(indices)
            else:
                selected = self._ds.select(range(start, stop))
            return [_make_item(row) for row in selected]
        if isinstance(idx, Sequence):
            selected = self._ds.select(list(idx))
            return [_make_item(row) for row in selected]
        raise TypeError(f"Invalid index type: {type(idx)}")

    def iter_items(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over items one by one. Yields dict with 'text' and label column(s) as keys.
        
        For single label: {"text": "...", "category": "..."}
        For multiple labels: {"text": "...", "label1": "...", "label2": "..."}
        """
        for row in self._ds:
            text = row.get(self._text_field) or row.get("text")
            if text is None:
                raise ValueError(f"Text field '{self._text_field}' or 'text' not found in dataset row")
            
            item = {"text": text}
            for cat_field in self._category_fields:
                category = row.get(cat_field)
                if category is None:
                    raise ValueError(f"Category field '{cat_field}' not found in dataset row")
                item[cat_field] = category
            yield item

    def iter_batches(self, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
        """
        Iterate over items in batches. Each batch is a list of dicts with 'text' and label column(s) as keys.
        
        For single label: [{"text": "...", "category": "..."}, ...]
        For multiple labels: [{"text": "...", "label1": "...", "label2": "..."}, ...]
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            batch = []
            for row in self._ds:
                text = row.get(self._text_field) or row.get("text")
                if text is None:
                    raise ValueError(f"Text field '{self._text_field}' or 'text' not found in dataset row")
                
                item = {"text": text}
                for cat_field in self._category_fields:
                    category = row.get(cat_field)
                    if category is None:
                        raise ValueError(f"Category field '{cat_field}' not found in dataset row")
                    item[cat_field] = category
                
                batch.append(item)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        else:
            # Use select to get batches with proper format
            for i in range(0, len(self), batch_size):
                end = min(i + batch_size, len(self))
                batch_list = self[i:end]
                yield batch_list

    def get_categories(self) -> Union[List[Any], Dict[str, List[Any]]]:
        """
        Get unique categories in the dataset, excluding None values.
        
        Returns:
            - For single label column: List of unique category values
            - For multiple label columns: Dict mapping column name to list of unique categories
        """
        if len(self._category_fields) == 1:
            # Single label: return list for backward compatibility
            cat_field = self._category_fields[0]
            if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
                categories = set()
                for item in self.iter_items():
                    cat = item[cat_field]
                    if cat is not None:
                        categories.add(cat)
                return sorted(list(categories))
            categories = [cat for cat in set(self._ds[cat_field]) if cat is not None]
            return sorted(categories)
        else:
            # Multiple labels: return dict
            result = {}
            if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
                # Collect categories from all items
                category_sets = {field: set() for field in self._category_fields}
                for item in self.iter_items():
                    for field in self._category_fields:
                        cat = item[field]
                        if cat is not None:
                            category_sets[field].add(cat)
                for field in self._category_fields:
                    result[field] = sorted(list(category_sets[field]))
            else:
                # Use direct column access
                for field in self._category_fields:
                    categories = [cat for cat in set(self._ds[field]) if cat is not None]
                    result[field] = sorted(categories)
            return result

    def get_texts(self) -> List[str]:
        """Get all texts as a list."""
        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            return [item["text"] for item in self.iter_items()]
        return list(self._ds[self._text_field])

    def get_categories_for_texts(self, texts: List[str]) -> Union[List[Any], List[Dict[str, Any]]]:
        """
        Get categories for given texts (if texts match dataset texts).
        
        Returns:
            - For single label column: List of category values (one per text)
            - For multiple label columns: List of dicts with label columns as keys
        """
        if self._loading_strategy == LoadingStrategy.ITERABLE_ONLY:
            raise NotImplementedError("get_categories_for_texts not supported for ITERABLE_ONLY datasets")
        
        if len(self._category_fields) == 1:
            # Single label: return list for backward compatibility
            cat_field = self._category_fields[0]
            text_to_category = {
                row[self._text_field]: row[cat_field]
                for row in self._ds
            }
            return [text_to_category.get(text) for text in texts]
        else:
            # Multiple labels: return list of dicts
            text_to_categories = {
                row[self._text_field]: {field: row[field] for field in self._category_fields}
                for row in self._ds
            }
            return [text_to_categories.get(text) for text in texts]

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
            category_field: Union[str, List[str]] = "category",
            filters: Optional[Dict[str, Any]] = None,
            limit: Optional[int] = None,
            streaming: Optional[bool] = None,
            **kwargs,
    ) -> "ClassificationDataset":
        """Load classification dataset from HuggingFace Hub."""
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
            category_field: Union[str, List[str]] = "category",
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
            category_field: Union[str, List[str]] = "category",
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
