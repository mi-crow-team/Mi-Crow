from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, TYPE_CHECKING
import json
import csv

from amber.store import Store

if TYPE_CHECKING:
    from amber.mechanistic.autoencoder.concepts.concept_models import NeuronText


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
    def from_csv(
            cls,
            csv_filepath: Path | str,
            n_size: int,
            store: Store | None = None,
            max_concepts: int | None = None
    ) -> "ConceptDictionary":
        csv_path = Path(csv_filepath)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        concept_dict = cls(n_size=n_size, store=store, max_concepts=max_concepts)

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                neuron_idx = int(row["neuron_idx"])
                concept_name = row["concept_name"]
                score = float(row["score"])
                concept_dict.add(neuron_idx, concept_name, score)

        return concept_dict

    @classmethod
    def from_json(
            cls,
            json_filepath: Path | str,
            n_size: int,
            store: Store | None = None,
            max_concepts: int | None = None
    ) -> "ConceptDictionary":
        json_path = Path(json_filepath)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        concept_dict = cls(n_size=n_size, store=store, max_concepts=max_concepts)

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for neuron_idx_str, concepts in data.items():
            neuron_idx = int(neuron_idx_str)
            # Handle case where concepts is not a list
            if not isinstance(concepts, list):
                continue
            for concept in concepts:
                if not isinstance(concept, dict):
                    continue
                concept_name = concept["name"]
                score = float(concept["score"])
                concept_dict.add(neuron_idx, concept_name, score)

        return concept_dict

    @classmethod
    def from_llm(
            cls,
            neuron_texts: list[list["NeuronText"]],
            n_size: int,
            store: Store | None = None,
            max_concepts: int | None = None,
            llm_provider: str | None = None
    ) -> "ConceptDictionary":
        concept_dict = cls(n_size=n_size, store=store, max_concepts=max_concepts)

        for neuron_idx, texts in enumerate(neuron_texts):
            if not texts:
                continue

            # Extract texts and their specific activated tokens
            texts_with_tokens = []
            for nt in texts:
                texts_with_tokens.append({
                    "text": nt.text,
                    "score": nt.score,
                    "token_str": nt.token_str,
                    "token_idx": nt.token_idx
                })

            # Generate concept names using LLM
            concept_names = cls._generate_concept_names_llm(texts_with_tokens, llm_provider)

            # Add concepts to dictionary
            for concept_name, score in concept_names:
                concept_dict.add(neuron_idx, concept_name, score)

        return concept_dict

    @staticmethod
    def _generate_concept_names_llm(texts_with_tokens: list[dict], llm_provider: str | None = None) -> list[
        tuple[str, float]]:
        raise NotImplementedError(
            "LLM provider not configured. Please implement _generate_concept_names_llm "
            "method with your preferred LLM provider (OpenAI, Anthropic, etc.)"
        )
