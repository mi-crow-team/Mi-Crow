from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Sequence, TYPE_CHECKING, Optional
import json
import csv

from amber.store.store import Store

if TYPE_CHECKING:
    pass


@dataclass
class Concept:
    name: str
    score: float


class ConceptDictionary:
    def __init__(
            self,
            n_size: int,
            store: Store | None = None
    ) -> None:
        self.n_size = n_size
        self.concepts_map: Dict[int, Concept] = {}
        self.store = store
        self._directory: Path | None = None

    def set_directory(self, directory: Path | str) -> None:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        self._directory = p

    def add(self, index: int, name: str, score: float) -> None:
        """
        Add a concept to the dictionary.
        
        Args:
            index: Neuron index (0 to n_size-1)
            name: Concept name
            score: Concept score
            
        Raises:
            IndexError: If index is out of bounds
            TypeError: If name is not a string
            ValueError: If name is empty
        """
        if not isinstance(index, int):
            raise TypeError(f"index must be int, got {type(index)}")
        
        if not (0 <= index < self.n_size):
            raise IndexError(f"index {index} out of bounds for n_size={self.n_size}")
        
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got {type(name)}")
        
        if not name.strip():
            raise ValueError("name cannot be empty or whitespace")
        
        if not isinstance(score, (int, float)):
            raise TypeError(f"score must be numeric, got {type(score)}")
        
        # Only allow 1 concept per neuron - replace if exists
        self.concepts_map[index] = Concept(name=name, score=float(score))

    def get(self, index: int) -> Optional[Concept]:
        if not (0 <= index < self.n_size):
            raise IndexError(f"index {index} out of bounds for n_size={self.n_size}")
        return self.concepts_map.get(index)

    def get_many(self, indices: Sequence[int]) -> Dict[int, Optional[Concept]]:
        return {i: self.get(i) for i in indices}

    def save(self, directory: Path | str | None = None) -> Path:
        if directory is not None:
            self.set_directory(directory)
        if self._directory is None:
            raise ValueError("No directory set. Call save(directory=...) or set_directory() first.")
        path = self._directory / "concepts.json"
        serializable = {str(k): asdict(v) for k, v in self.concepts_map.items()}
        meta = {
            "n_size": self.n_size,
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
        concepts = meta.get("concepts", {})
        # Handle both old format (list) and new format (single dict)
        self.concepts_map = {}
        for k, v in concepts.items():
            if isinstance(v, list):
                # Old format: take first concept if list
                if v:
                    self.concepts_map[int(k)] = Concept(**v[0])
            else:
                # New format: single concept dict
                self.concepts_map[int(k)] = Concept(**v)

    @classmethod
    def from_csv(
            cls,
            csv_filepath: Path | str,
            n_size: int,
            store: Store | None = None
    ) -> "ConceptDictionary":
        """
        Load ConceptDictionary from CSV file.
        
        Args:
            csv_filepath: Path to CSV file
            n_size: Number of neurons/concepts
            store: Optional Store instance
            
        Returns:
            ConceptDictionary instance
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
        """
        csv_path = Path(csv_filepath)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        concept_dict = cls(n_size=n_size, store=store)

        # Track best concept per neuron (highest score)
        neuron_concepts: Dict[int, tuple[str, float]] = {}

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                neuron_idx = int(row["neuron_idx"])
                concept_name = row["concept_name"]
                score = float(row["score"])

                # Keep only the concept with highest score per neuron
                if neuron_idx not in neuron_concepts or score > neuron_concepts[neuron_idx][1]:
                    neuron_concepts[neuron_idx] = (concept_name, score)

        # Add the best concept for each neuron
        cls._load_concepts_from_data(concept_dict, neuron_concepts)

        return concept_dict

    @classmethod
    def from_json(
            cls,
            json_filepath: Path | str,
            n_size: int,
            store: Store | None = None
    ) -> "ConceptDictionary":
        """
        Load ConceptDictionary from JSON file.
        
        Args:
            json_filepath: Path to JSON file
            n_size: Number of neurons/concepts
            store: Optional Store instance
            
        Returns:
            ConceptDictionary instance
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
        """
        json_path = Path(json_filepath)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        concept_dict = cls(n_size=n_size, store=store)

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract concepts from JSON data
        neuron_concepts: Dict[int, tuple[str, float]] = {}
        for neuron_idx_str, concepts in data.items():
            neuron_idx = int(neuron_idx_str)
            concept_name, score = cls._extract_concept_from_json_entry(concepts)
            if concept_name is not None:
                neuron_concepts[neuron_idx] = (concept_name, score)

        # Add concepts to dictionary
        cls._load_concepts_from_data(concept_dict, neuron_concepts)

        return concept_dict

    @staticmethod
    def _extract_concept_from_json_entry(concepts: Any) -> tuple[str | None, float]:
        """
        Extract concept name and score from JSON entry (handles both old and new formats).
        
        Args:
            concepts: JSON entry (list or dict)
            
        Returns:
            Tuple of (concept_name, score) or (None, 0.0) if invalid
        """
        if isinstance(concepts, list):
            # Old format: take the concept with highest score
            best_concept = None
            best_score = float('-inf')
            for concept in concepts:
                if not isinstance(concept, dict):
                    continue
                score = float(concept.get("score", 0.0))
                if score > best_score:
                    best_score = score
                    best_concept = concept

            if best_concept is not None:
                return (best_concept["name"], best_score)
            return (None, 0.0)
        elif isinstance(concepts, dict):
            # New format: single concept dict
            concept_name = concepts["name"]
            score = float(concepts["score"])
            return (concept_name, score)
        else:
            return (None, 0.0)

    @staticmethod
    def _load_concepts_from_data(
            concept_dict: "ConceptDictionary",
            neuron_concepts: Dict[int, tuple[str, float]]
    ) -> None:
        """
        Load concepts from data dictionary into ConceptDictionary.
        
        Args:
            concept_dict: ConceptDictionary instance to populate
            neuron_concepts: Dictionary mapping neuron_idx to (concept_name, score) tuples
        """
        for neuron_idx, (concept_name, score) in neuron_concepts.items():
            concept_dict.add(neuron_idx, concept_name, score)

    @classmethod
    def from_llm(
            cls,
            neuron_texts: list[list["NeuronText"]],
            n_size: int,
            store: Store | None = None,
            llm_provider: str | None = None
    ) -> "ConceptDictionary":
        concept_dict = cls(n_size=n_size, store=store)

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

            # Add only the best concept (highest score) to dictionary
            if concept_names:
                # Sort by score descending and take the first one
                concept_names_sorted = sorted(concept_names, key=lambda x: x[1], reverse=True)
                concept_name, score = concept_names_sorted[0]
                concept_dict.add(neuron_idx, concept_name, score)

        return concept_dict

    @staticmethod
    def _generate_concept_names_llm(texts_with_tokens: list[dict], llm_provider: str | None = None) -> list[
        tuple[str, float]]:
        raise NotImplementedError(
            "LLM provider not configured. Please implement _generate_concept_names_llm "
            "method with your preferred LLM provider (OpenAI, Anthropic, etc.)"
        )
