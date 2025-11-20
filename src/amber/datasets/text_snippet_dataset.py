from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Sequence, Union, Optional

from datasets import Dataset, load_dataset, load_from_disk

IndexLike = Union[int, slice, Sequence[int]]


class TextSnippetDataset:
    """
    Text-only dataset with fast random access (Arrow memory-mapped)
    and convenient batch/stream APIs.

    Each item is a string (text snippet).
    """

    def __init__(self, ds: Dataset, dataset_dir: Union[str, Path, object]):
        if "text" not in ds.column_names:
            raise ValueError(f"Dataset must have a 'text' column; got {ds.column_names}")
        self.ds: Dataset = ds.remove_columns([c for c in ds.column_names if c != "text"])
        # Allow passing a LocalStore-like object with a base_path attribute
        base = dataset_dir
        if not isinstance(dataset_dir, (str, Path)) and hasattr(dataset_dir, "base_path"):
            base = getattr(dataset_dir, "base_path")
        self.dataset_dir = Path(base)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.ds.save_to_disk(str(self.dataset_dir))
        self.ds = load_from_disk(str(self.dataset_dir))
        self.ds.set_format("python", columns=["text"])

    # --- Constructors ---

    @classmethod
    def from_huggingface(
            cls,
            repo_id: str,
            *,
            split: str = "train",
            dataset_dir: Union[str, Path] = "./snippet_cache/hf",
            revision: Optional[str] = None,
            text_field: str = "text",
            filters: Optional[dict] = None,
            limit: Optional[int] = None,
    ) -> "TextSnippetDataset":
        """
        Load from Hugging Face hub and persist to disk (Arrow).

        - repo_id: e.g. "openwebtext" or "roneneldan/TinyStories"
        - split:   e.g. "train"
        - revision: branch/tag/commit if you need a specific version
        - text_field: which column holds the text
        - filters: dict like {"lang": "pl"} applied via .filter
        - limit: take first N rows for a small subset
        """
        ds = load_dataset(
            path=repo_id,
            split=split,
            revision=revision,
        )
        if text_field != "text":
            if text_field not in ds.column_names:
                raise ValueError(f"text_field '{text_field}' not in columns: {ds.column_names}")
            ds = ds.rename_column(text_field, "text")

        if filters:
            def _pred(example):
                return all(example.get(k) == v for k, v in filters.items())

            ds = ds.filter(_pred)

        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        return cls(ds, dataset_dir)

    @classmethod
    def from_local(
            cls,
            source: Union[str, Path],
            *,
            dataset_dir: Union[str, Path] = "./snippet_cache/local",
            text_field: str = "text",
            recursive: bool = True,
    ) -> "TextSnippetDataset":
        """
        Load from a local directory or file(s) and persist to disk.

        Supported:
          - Directory of .txt files (each file becomes one example)
          - JSONL/JSON/CSV/TSV files with a text column

        Args:
          source: path to directory or file
          :param text_field: the column to use when loading structured files
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
                ds = load_dataset("json", data_files=str(p), split="train")
            elif suffix in {".csv"}:
                ds = load_dataset("csv", data_files=str(p), split="train")
            elif suffix in {".tsv"}:
                ds = load_dataset("csv", data_files=str(p), split="train", delimiter="\t")
            else:
                raise ValueError(
                    f"Unsupported file type: {suffix}. "
                    f"Use directory of .txt, or JSON/JSONL/CSV/TSV."
                )
            if text_field != "text":
                if text_field not in ds.column_names:
                    raise ValueError(f"text_field '{text_field}' not in columns: {ds.column_names}")
                ds = ds.rename_column(text_field, "text")

        return cls(ds, dataset_dir)

    # --- Pythonic language_model API ---

    def __len__(self) -> int:
        return self.ds.num_rows

    def __getitem__(self, idx: IndexLike) -> Union[str, List[str]]:
        if isinstance(idx, int):
            return self.ds[idx]["text"]
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                indices = list(range(start, stop, step))
                out = self.ds.select(indices)["text"]
            else:
                out = self.ds.select(range(start, stop))["text"]
            return list(out)
        if isinstance(idx, Sequence):
            out = self.ds.select(list(idx))["text"]
            return list(out)

    # --- Convenience: batches & streaming ---

    def get_batch(self, start: int, batch_size: int) -> List[str]:
        """
        Fast contiguous batch read using Arrow offsets (select(range(...))).
        """
        if batch_size <= 0:
            return []
        end = min(start + batch_size, len(self))
        if start >= end:
            return []
        return list(self.ds.select(range(start, end))["text"])

    def get_batch_by_indices(self, indices: Sequence[int]) -> List[str]:
        """
        Fast non-contiguous batch read using .select(list(indices)).
        """
        if not indices:
            return []
        return list(self.ds.select(list(indices))["text"])

    def iter_items(self) -> Iterator[str]:
        """
        Stream items one by one (zero-copy iter over Arrow batches).
        """
        for row in self.ds:
            yield row["text"]

    def iter_batches(self, batch_size: int) -> Iterator[List[str]]:
        """
        Stream items in batches; each batch is a plain list[str].
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        for batch in self.ds.iter(batch_size=batch_size):
            yield list(batch["text"])

    # --- Utilities ---

    def head(self, n: int = 5) -> List[str]:
        return list(self.ds.select(range(min(n, len(self))))["text"])
