from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence
import json

from amber.store import Store


@dataclass
class Concept:
    name: str
    score: float


class ConceptDictionary:
    def __init__(
            self,
            n_size: int,
            store: Store | None = None,
            max_concepts: int | None = None
    ) -> None:
        self.n_size = n_size
        self.max_concepts = max_concepts
        self.concepts_map: Dict[int, List[Concept]] = {}
        self.store = store
        self._directory: Path | None = None

    def set_directory(self, directory: Path | str) -> None:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        self._directory = p

    def add(self, index: int, name: str, score: float) -> None:
        if not (0 <= index < self.n_size):
            raise IndexError(f"index {index} out of bounds for n_size={self.n_size}")
        lst = self.concepts_map.setdefault(index, [])
        lst.append(Concept(name=name, score=score))
        if self.max_concepts is not None and len(lst) > self.max_concepts:
            # Keep the top-k by score
            lst.sort(key=lambda c: c.score, reverse=True)
            del lst[self.max_concepts:]

    def get(self, index: int) -> List[Concept]:
        if not (0 <= index < self.n_size):
            raise IndexError(f"index {index} out of bounds for n_size={self.n_size}")
        return list(self.concepts_map.get(index, []))

    def get_many(self, indices: Sequence[int]) -> Dict[int, List[Concept]]:
        return {i: self.get(i) for i in indices}

    def save(self, directory: Path | str | None = None) -> Path:
        if directory is not None:
            self.set_directory(directory)
        if self._directory is None:
            raise ValueError("No directory set. Call save(directory=...) or set_directory() first.")
        path = self._directory / "concepts.json"
        serializable = {str(k): [asdict(c) for c in v] for k, v in self.concepts_map.items()}
        meta = {
            "n_size": self.n_size,
            "max_concepts": self.max_concepts,
            "concepts": serializable,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return path

    def load(self, directory: Path | str | None = None) -> None:
        if directory is not None:
            self.set_directory(directory)
        if self._directory is None:
            raise ValueError("No directory set. Call load(directory=...) or set_directory() first.")
        path = self._directory / "concepts.json"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        self.n_size = int(meta.get("n_size", self.n_size))
        self.max_concepts = meta.get("max_concepts", self.max_concepts)
        concepts = meta.get("concepts", {})
        self.concepts_map = {
            int(k): [Concept(**c) for c in v]
            for k, v in concepts.items()
        }

    @classmethod
    def from_directory(cls, directory: Path | str) -> "ConceptDictionary":
        p = Path(directory)
        if not p.exists():
            raise FileNotFoundError(p)
        meta_path = p / "concepts.json"
        if not meta_path.exists():
            # Create empty dictionary with best guess of n_size (0)
            inst = cls(n_size=0)
            inst.set_directory(p)
            return inst
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        n_size = int(meta.get("n_size", 0))
        max_concepts = meta.get("max_concepts")
        inst = cls(n_size=n_size, max_concepts=max_concepts)
        inst.set_directory(p)
        concepts = meta.get("concepts", {})
        inst.concepts_map = {int(k): [Concept(**c) for c in v] for k, v in concepts.items()}
        return inst
