from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from datasets import Dataset, IterableDataset, load_dataset, load_from_disk

from mi_crow.datasets.base_dataset import BaseDataset
from mi_crow.datasets.loading_strategy import IndexLike, LoadingStrategy
from mi_crow.store.store import Store


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

        Raises:
            ValueError: If text_field or category_field is empty, or fields not found in dataset
        """
        self._validate_text_field(text_field)

        # Normalize category_field to list
        if isinstance(category_field, str):
            self._category_fields = [category_field]
        else:
            self._category_fields = list(category_field)

        self._validate_category_fields(self._category_fields)

        # Validate dataset
        is_iterable = isinstance(ds, IterableDataset)
        if not is_iterable:
            if text_field not in ds.column_names:
                raise ValueError(f"Dataset must have a '{text_field}' column; got columns: {ds.column_names}")
            for cat_field in self._category_fields:
                if cat_field not in ds.column_names:
                    raise ValueError(f"Dataset must have a '{cat_field}' column; got columns: {ds.column_names}")
            # Set format with all required columns
            format_columns = [text_field] + self._category_fields
            ds.set_format("python", columns=format_columns)

        self._text_field = text_field
        self._category_field = category_field  # Keep original for backward compatibility
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

    def _validate_category_fields(self, category_fields: List[str]) -> None:
        """Validate category_fields parameter.

        Args:
            category_fields: List of category field names to validate

        Raises:
            ValueError: If category_fields is empty or contains invalid values
        """
        if not category_fields:
            raise ValueError("category_field cannot be empty")

        for cat_field in category_fields:
            if not cat_field or not isinstance(cat_field, str) or not cat_field.strip():
                raise ValueError(f"All category fields must be non-empty strings, got invalid field: {cat_field!r}")

    def _extract_item_from_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Extract item (text + categories) from a dataset row.

        Args:
            row: Dataset row dictionary

        Returns:
            Dictionary with 'text' and category fields as keys

        Raises:
            ValueError: If required fields are not found in row
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

        item = {"text": text}
        for cat_field in self._category_fields:
            if cat_field not in row:
                raise ValueError(
                    f"Category field '{cat_field}' not found in dataset row. Available fields: {list(row.keys())}"
                )
            category = row.get(cat_field)  # Potentially None
            item[cat_field] = category

        return item

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Raises:
            NotImplementedError: If loading_strategy is STREAMING
        """
        if self._loading_strategy == LoadingStrategy.STREAMING:
            raise NotImplementedError("len() not supported for STREAMING datasets")
        return self._ds.num_rows

    def __getitem__(self, idx: IndexLike) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get item(s) by index. Returns dict with 'text' and label column(s) as keys.

        For single label: {"text": "...", "category": "..."}
        For multiple labels: {"text": "...", "label1": "...", "label2": "..."}

        Args:
            idx: Index (int), slice, or sequence of indices

        Returns:
            Single item dict or list of item dicts

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
            row = self._ds[idx]
            return self._extract_item_from_row(row)

        if isinstance(idx, slice):
            start, stop, step = idx.indices(dataset_len)
            if step != 1:
                indices = list(range(start, stop, step))
                selected = self._ds.select(indices)
            else:
                selected = self._ds.select(range(start, stop))
            return [self._extract_item_from_row(row) for row in selected]

        if isinstance(idx, Sequence):
            # Validate all indices are in bounds
            invalid_indices = [i for i in idx if not (0 <= i < dataset_len)]
            if invalid_indices:
                raise IndexError(f"Indices out of bounds: {invalid_indices} (dataset length: {dataset_len})")
            selected = self._ds.select(list(idx))
            return [self._extract_item_from_row(row) for row in selected]

        raise TypeError(f"Invalid index type: {type(idx)}")

    def iter_items(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over items one by one. Yields dict with 'text' and label column(s) as keys.

        For single label: {"text": "...", "category_column_1": "..."}
        For multiple labels: {"text": "...", "category_column_1": "...", "category_column_2": "..."}

        Yields:
            Item dictionaries with text and category fields

        Raises:
            ValueError: If required fields are not found in any row
        """
        for row in self._ds:
            yield self._extract_item_from_row(row)

    def iter_batches(self, batch_size: int) -> Iterator[List[Dict[str, Any]]]:
        """
        Iterate over items in batches. Each batch is a list of dicts with 'text' and label column(s) as keys.

        For single label: [{"text": "...", "category_column_1": "..."}, ...]
        For multiple labels: [{"text": "...", "category_column_1": "...", "category_column_2": "..."}, ...]

        Args:
            batch_size: Number of items per batch

        Yields:
            Lists of item dictionaries (batches)

        Raises:
            ValueError: If batch_size <= 0 or required fields are not found in any row
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got: {batch_size}")

        if self._loading_strategy == LoadingStrategy.STREAMING:
            batch = []
            for row in self._ds:
                batch.append(self._extract_item_from_row(row))
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

    def get_categories(self) -> Union[List[Any], Dict[str, List[Any]]]:  # noqa: C901
        """
        Get unique categories in the dataset, excluding None values.

        Returns:
            - For single label column: List of unique category values
            - For multiple label columns: Dict mapping column name to list of unique categories

        Raises:
            NotImplementedError: If loading_strategy is STREAMING and dataset is large
        """
        if len(self._category_fields) == 1:
            # Single label: return list for backward compatibility
            cat_field = self._category_fields[0]
            if self._loading_strategy == LoadingStrategy.STREAMING:
                categories = set()
                for item in self.iter_items():
                    cat = item[cat_field]
                    if cat is not None:
                        categories.add(cat)
                return sorted(list(categories))  # noqa: C414
            categories = [cat for cat in set(self._ds[cat_field]) if cat is not None]
            return sorted(categories)
        else:
            # Multiple labels: return dict
            result = {}
            if self._loading_strategy == LoadingStrategy.STREAMING:
                # Collect categories from all items
                category_sets = {field: set() for field in self._category_fields}
                for item in self.iter_items():
                    for field in self._category_fields:
                        cat = item[field]
                        if cat is not None:
                            category_sets[field].add(cat)
                for field in self._category_fields:
                    result[field] = sorted(list(category_sets[field]))  # noqa: C414
            else:
                # Use direct column access
                for field in self._category_fields:
                    categories = [cat for cat in set(self._ds[field]) if cat is not None]
                    result[field] = sorted(categories)
            return result

    def extract_texts_from_batch(self, batch: List[Dict[str, Any]]) -> List[Optional[str]]:
        """Extract text strings from a batch of classification items.

        Args:
            batch: List of dicts with 'text' and category fields

        Returns:
            List of text strings from the batch

        Raises:
            ValueError: If 'text' key is not found in any batch item
        """
        texts = []
        for item in batch:
            if "text" not in item:
                raise ValueError(f"'text' key not found in batch item. Available keys: {list(item.keys())}")
            texts.append(item["text"])
        return texts

    def get_all_texts(self) -> List[Optional[str]]:
        """Get all texts from the dataset.

        Returns:
            List of all text strings

        Raises:
            NotImplementedError: If loading_strategy is STREAMING and dataset is very large
        """
        if self._loading_strategy == LoadingStrategy.STREAMING:
            return [item["text"] for item in self.iter_items()]
        return list(self._ds[self._text_field])

    def get_categories_for_texts(self, texts: List[Optional[str]]) -> Union[List[Any], List[Dict[str, Any]]]:
        """
        Get categories for given texts (if texts match dataset texts).

        Args:
            texts: List of text strings to look up

        Returns:
            - For single label column: List of category values (one per text)
            - For multiple label columns: List of dicts with label columns as keys

        Raises:
            NotImplementedError: If loading_strategy is STREAMING
            ValueError: If texts list is empty
        """
        if self._loading_strategy == LoadingStrategy.STREAMING:
            raise NotImplementedError("get_categories_for_texts not supported for STREAMING datasets")

        if not texts:
            raise ValueError("texts list cannot be empty")

        if len(self._category_fields) == 1:
            # Single label: return list for backward compatibility
            cat_field = self._category_fields[0]
            text_to_category = {row[self._text_field]: row[cat_field] for row in self._ds}
            return [text_to_category.get(text) for text in texts]
        else:
            # Multiple labels: return list of dicts
            text_to_categories = {
                row[self._text_field]: {field: row[field] for field in self._category_fields} for row in self._ds
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
        stratify_by: Optional[str] = None,
        stratify_seed: Optional[int] = None,
        streaming: Optional[bool] = None,
        drop_na: bool = False,
        **kwargs: Any,
    ) -> "ClassificationDataset":
        """
        Load classification dataset from HuggingFace Hub.

        Args:
            repo_id: HuggingFace dataset repository ID
            store: Store instance
            split: Dataset split
            loading_strategy: Loading strategy
            revision: Optional git revision
            text_field: Name of the column containing text
            category_field: Name(s) of the column(s) containing category/label
            filters: Optional filters to apply (dict of column: value)
            limit: Optional limit on number of rows
            stratify_by: Optional column used for stratified sampling (non-streaming only)
            stratify_seed: Optional RNG seed for stratified sampling
            streaming: Optional override for streaming
            drop_na: Whether to drop rows with None/empty text or categories
            **kwargs: Additional arguments for load_dataset

        Returns:
            ClassificationDataset instance

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
                drop_na_columns = None
                if drop_na:
                    cat_fields = [category_field] if isinstance(category_field, str) else category_field
                    drop_na_columns = [text_field] + list(cat_fields)

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
                f"Failed to load classification dataset from HuggingFace Hub: "
                f"repo_id={repo_id!r}, split={split!r}, text_field={text_field!r}, "
                f"category_field={category_field!r}. Error: {e}"
            ) from e

        return cls(
            ds,
            store=store,
            loading_strategy=loading_strategy,
            text_field=text_field,
            category_field=category_field,
        )

    @classmethod
    def from_disk(
        cls,
        store: Store,
        *,
        loading_strategy: LoadingStrategy = LoadingStrategy.MEMORY,
        text_field: str = "text",
        category_field: Union[str, List[str]] = "category",
    ) -> "ClassificationDataset":
        """
        Load classification dataset from already-saved Arrow files on disk.

        Use this when you've previously saved a dataset and want to reload it
        without re-downloading from HuggingFace or re-applying transformations.

        Args:
            store: Store instance pointing to where the dataset was saved
            loading_strategy: Loading strategy (MEMORY or DISK only)
            text_field: Name of the column containing text
            category_field: Name(s) of the column(s) containing category/label

        Returns:
            ClassificationDataset instance loaded from disk

        Raises:
            FileNotFoundError: If dataset directory doesn't exist or contains no Arrow files
            ValueError: If required fields are not in the loaded dataset

        Example:
            # First: save dataset
            dataset_store = LocalStore("store/wgmix_test")
            dataset = ClassificationDataset.from_huggingface(
                "allenai/wildguardmix",
                store=dataset_store,
                limit=100
            )
            # Dataset saved to: store/wgmix_test/datasets/*.arrow

            # Later: reload from disk
            dataset_store = LocalStore("store/wgmix_test")
            dataset = ClassificationDataset.from_disk(
                store=dataset_store,
                text_field="prompt",
                category_field="prompt_harm_label"
            )
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

        # Create ClassificationDataset with the loaded dataset and field names
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
        stratify_by: Optional[str] = None,
        stratify_seed: Optional[int] = None,
        drop_na: bool = False,
        **kwargs: Any,
    ) -> "ClassificationDataset":
        """
        Load classification dataset from CSV file.

        Args:
            source: Path to CSV file
            store: Store instance
            loading_strategy: Loading strategy
            text_field: Name of the column containing text
            category_field: Name(s) of the column(s) containing category/label
            delimiter: CSV delimiter (default: comma)
            stratify_by: Optional column used for stratified sampling
            stratify_seed: Optional RNG seed for stratified sampling
            drop_na: Whether to drop rows with None/empty text or categories
            **kwargs: Additional arguments for load_dataset

        Returns:
            ClassificationDataset instance

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
            drop_na_columns = None
            if drop_na:
                cat_fields = [category_field] if isinstance(category_field, str) else category_field
                drop_na_columns = [text_field] + list(cat_fields)

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
        stratify_by: Optional[str] = None,
        stratify_seed: Optional[int] = None,
        drop_na: bool = False,
        **kwargs: Any,
    ) -> "ClassificationDataset":
        """
        Load classification dataset from JSON/JSONL file.

        Args:
            source: Path to JSON or JSONL file
            store: Store instance
            loading_strategy: Loading strategy
            text_field: Name of the field containing text
            category_field: Name(s) of the field(s) containing category/label
            stratify_by: Optional column used for stratified sampling
            stratify_seed: Optional RNG seed for stratified sampling
            drop_na: Whether to drop rows with None/empty text or categories
            **kwargs: Additional arguments for load_dataset

        Returns:
            ClassificationDataset instance

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
            drop_na_columns = None
            if drop_na:
                cat_fields = [category_field] if isinstance(category_field, str) else category_field
                drop_na_columns = [text_field] + list(cat_fields)

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
            category_field=category_field,
        )
