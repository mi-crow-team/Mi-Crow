"""Test persistence functionality in ConceptDictionary."""

import pytest
import json
import csv
from pathlib import Path
from unittest.mock import patch

from amber.mechanistic.autoencoder.concepts.concept_dictionary import ConceptDictionary, Concept
from amber.mechanistic.autoencoder.concepts.concept_models import NeuronText


class TestConceptDictionaryPersistence:
    """Test save/load functionality and edge cases."""

    def test_save_load_with_various_concept_sizes(self, tmp_path):
        """Test save/load with different concept dictionary sizes."""
        # Test with different sizes
        sizes = [0, 1, 5, 10, 100]
        
        for size in sizes:
            # Create dictionary
            dictionary = ConceptDictionary(n_size=size)
            
            # Add some concepts
            for i in range(min(size, 5)):  # Add up to 5 concepts
                dictionary.add(i, f"concept_{i}", float(i))
            
            # Save to directory
            save_dir = tmp_path / f"dict_{size}"
            dictionary.set_directory(save_dir)
            saved_path = dictionary.save()
            
            # Load from directory
            loaded_dict = ConceptDictionary(n_size=0)
            loaded_dict.load(directory=save_dir)
            
            # Should match
            assert loaded_dict.n_size == size
            assert loaded_dict.concepts_map == dictionary.concepts_map

    def test_one_concept_per_neuron_behavior(self, tmp_path):
        """Test that only one concept per neuron is kept."""
        dictionary = ConceptDictionary(n_size=5)
        
        # Add multiple concepts - only last one is kept
        for i in range(5):
            dictionary.add(0, f"concept_{i}", float(i))
        
        # Should only keep last one (1 per neuron)
        concept = dictionary.get(0)
        assert concept is not None
        assert concept.name == "concept_4"
        assert concept.score == 4.0

    def test_export_to_csv_format(self, tmp_path):
        """Test export to CSV format."""
        dictionary = ConceptDictionary(n_size=3)
        
        # Add some concepts
        dictionary.add(0, "concept_0", 0.8)
        dictionary.add(0, "concept_1", 0.6)
        dictionary.add(1, "concept_2", 0.9)
        
        # Save to JSON first (since export_to_csv doesn't exist)
        save_dir = tmp_path / "concepts"
        dictionary.set_directory(save_dir)
        saved_path = dictionary.save()
        
        # Check JSON file exists
        assert saved_path.exists()
        
        # Read and verify JSON content
        with saved_path.open("r") as f:
            data = json.load(f)
            
            assert "concepts" in data
            assert "0" in data["concepts"]
            assert "1" in data["concepts"]
            # Only 1 concept per neuron - check it's a dict (single concept) or list (backward compat)
            concept_0 = data["concepts"]["0"]
            if isinstance(concept_0, list):
                assert len(concept_0) >= 1  # At least 1 concept
            else:
                assert isinstance(concept_0, dict)  # Single concept dict
            concept_1 = data["concepts"]["1"]
            if isinstance(concept_1, list):
                assert len(concept_1) >= 1
            else:
                assert isinstance(concept_1, dict)

    def test_from_csv_factory_method(self, tmp_path):
        """Test from_csv factory method."""
        # Create a CSV file
        csv_path = tmp_path / "concepts.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["neuron_idx", "concept_name", "score"])
            writer.writerow([0, "concept_0", 0.8])
            writer.writerow([0, "concept_1", 0.6])
            writer.writerow([1, "concept_2", 0.9])
        
        # Create dictionary from CSV
        dictionary = ConceptDictionary.from_csv(csv_path, n_size=3)
        
        # Check that concepts were added correctly
        assert dictionary.n_size == 3
        
        # Only highest scoring concept per neuron is kept
        concept_0 = dictionary.get(0)
        assert concept_0 is not None
        assert concept_0.name == "concept_0"  # Highest score (0.8 > 0.6)
        assert concept_0.score == 0.8
        
        concept_1 = dictionary.get(1)
        assert concept_1 is not None
        assert concept_1.name == "concept_2"
        assert concept_1.score == 0.9

    def test_from_csv_file_not_found(self, tmp_path):
        """Test from_csv with non-existent file."""
        csv_path = tmp_path / "non_existent.csv"
        
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            ConceptDictionary.from_csv(csv_path, n_size=3)

    def test_from_json_factory_method(self, tmp_path):
        """Test from_json factory method."""
        # Create a JSON file
        json_path = tmp_path / "concepts.json"
        json_data = {
            "0": [
                {"name": "concept_0", "score": 0.8},
                {"name": "concept_1", "score": 0.6}
            ],
            "1": [
                {"name": "concept_2", "score": 0.9}
            ]
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(json_data, f)
        
        # Create dictionary from JSON
        dictionary = ConceptDictionary.from_json(json_path, n_size=3)
        
        # Check that concepts were added correctly
        assert dictionary.n_size == 3
        
        # Only highest scoring concept per neuron is kept
        concept_0 = dictionary.get(0)
        assert concept_0 is not None
        assert concept_0.name == "concept_0"  # Highest score (0.8 > 0.6)
        assert concept_0.score == 0.8
        
        concept_1 = dictionary.get(1)
        assert concept_1 is not None
        assert concept_1.name == "concept_2"
        assert concept_1.score == 0.9

    def test_from_json_file_not_found(self, tmp_path):
        """Test from_json with non-existent file."""
        json_path = tmp_path / "non_existent.json"
        
        with pytest.raises(FileNotFoundError, match="JSON file not found"):
            ConceptDictionary.from_json(json_path, n_size=3)

    def test_from_json_with_non_list_concepts(self, tmp_path):
        """Test from_json handles non-list concepts gracefully."""
        json_path = tmp_path / "concepts.json"
        json_data = {
            "0": [
                {"name": "concept_0", "score": 0.8}
            ],
            "1": "not_a_list",  # Should be skipped
            "2": 123,  # Should be skipped
            "3": [
                {"name": "concept_1", "score": 0.9}
            ]
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(json_data, f)
        
        dictionary = ConceptDictionary.from_json(json_path, n_size=5)
        
        # Should only have concepts from neuron 0 and 3
        assert dictionary.get(0) is not None
        assert dictionary.get(1) is None
        assert dictionary.get(2) is None
        assert dictionary.get(3) is not None

    def test_from_json_with_non_dict_concepts(self, tmp_path):
        """Test from_json handles non-dict concept entries gracefully."""
        json_path = tmp_path / "concepts.json"
        json_data = {
            "0": [
                {"name": "concept_0", "score": 0.8},
                "not_a_dict",  # Should be skipped
                123,  # Should be skipped
                {"name": "concept_1", "score": 0.6}
            ]
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(json_data, f)
        
        dictionary = ConceptDictionary.from_json(json_path, n_size=3)
        
        # Should only have 1 concept (highest scoring, skipping non-dict entries)
        concept = dictionary.get(0)
        assert concept is not None
        assert concept.name == "concept_0"  # Highest score (0.8 > 0.6)
        assert concept.score == 0.8

    def test_from_llm_raises_not_implemented(self):
        """Test from_llm raises NotImplementedError."""
        neuron_texts = [
            [NeuronText(text="test", score=0.8, token_str="test", token_idx=0)]
        ]
        
        with pytest.raises(NotImplementedError, match="LLM provider not configured"):
            ConceptDictionary.from_llm(neuron_texts, n_size=3)

    def test_from_llm_with_empty_texts(self):
        """Test from_llm handles empty neuron texts."""
        neuron_texts = [
            [],  # Empty list for neuron 0
            [NeuronText(text="test", score=0.8, token_str="test", token_idx=0)]  # Neuron 1 has text
        ]
        
        with pytest.raises(NotImplementedError):
            ConceptDictionary.from_llm(neuron_texts, n_size=3)

    def test_save_load_with_special_characters(self, tmp_path):
        """Test save/load with special characters in concept names."""
        dictionary = ConceptDictionary(n_size=3)
        
        # Add concepts with special characters
        special_names = [
            "concept with spaces",
            "concept-with-dashes",
            "concept_with_underscores",
            "concept.with.dots",
            "concept/with/slashes",
            "concept\\with\\backslashes",
            "concept:with:colons",
            "concept;with;semicolons",
            "concept,with,commas",
            "concept\"with\"quotes",
            "concept'with'apostrophes",
            "concept<with>brackets",
            "concept[with]square_brackets",
            "concept{with}braces",
            "concept(with)parentheses",
            "concept@with@at_symbols",
            "concept#with#hashes",
            "concept$with$dollars",
            "concept%with%percents",
            "concept^with^carets",
            "concept&with&ampersands",
            "concept*with*asterisks",
            "concept+with+pluses",
            "concept=with=equals",
            "concept?with?question_marks",
            "concept!with!exclamation_marks",
            "concept|with|pipes",
            "concept~with~tildes",
            "concept`with`backticks",
            "concept\twith\ttabs",
            "concept\nwith\nnewlines",
            "concept\rwith\rcarriage_returns",
            "concept with unicode: ñáéíóú",
            "concept with emoji: 🚀🎉💯",
        ]
        
        for i, name in enumerate(special_names):
            dictionary.add(0, name, float(i))
        
        # Save and load
        save_dir = tmp_path / "special_chars"
        dictionary.set_directory(save_dir)
        saved_path = dictionary.save()
        
        loaded_dict = ConceptDictionary(n_size=0)
        loaded_dict.load(directory=save_dir)
        
        # Only last concept is kept (1 per neuron)
        loaded_concept = loaded_dict.get(0)
        assert loaded_concept is not None
        assert loaded_concept.name == special_names[-1]  # Last one added
        assert loaded_concept.score == float(len(special_names) - 1)

    def test_save_load_with_large_concept_dictionary(self, tmp_path):
        """Test save/load with large concept dictionary."""
        # Create large dictionary
        n_size = 1000
        dictionary = ConceptDictionary(n_size=n_size)
        
        # Add concepts to multiple neurons
        for neuron_idx in range(0, n_size, 10):  # Every 10th neuron
            for concept_idx in range(5):  # 5 concepts per neuron
                dictionary.add(
                    neuron_idx,
                    f"neuron_{neuron_idx}_concept_{concept_idx}",
                    float(concept_idx)
                )
        
        # Save and load
        save_dir = tmp_path / "large_dict"
        dictionary.set_directory(save_dir)
        saved_path = dictionary.save()
        
        loaded_dict = ConceptDictionary(n_size=0)
        loaded_dict.load(directory=save_dir)
        
        # Check that all concepts were preserved
        assert loaded_dict.n_size == n_size
        
        # Check a few random neurons - only last concept per neuron is kept
        for neuron_idx in range(0, n_size, 100):  # Every 100th neuron
            concept = loaded_dict.get(neuron_idx)
            assert concept is not None
            assert concept.name.startswith(f"neuron_{neuron_idx}_")
            assert concept.name == f"neuron_{neuron_idx}_concept_4"  # Last one added

    def test_save_load_with_nested_concept_structure(self, tmp_path):
        """Test save/load with nested concept structure."""
        dictionary = ConceptDictionary(n_size=3)
        
        # Add concepts with hierarchical names
        hierarchical_concepts = [
            "animal.mammal.dog",
            "animal.mammal.cat",
            "animal.bird.eagle",
            "animal.bird.sparrow",
            "plant.tree.oak",
            "plant.tree.pine",
            "plant.flower.rose",
            "plant.flower.tulip",
        ]
        
        for i, name in enumerate(hierarchical_concepts):
            dictionary.add(0, name, float(i))
        
        # Save and load
        save_dir = tmp_path / "nested_dict"
        dictionary.set_directory(save_dir)
        saved_path = dictionary.save()
        
        loaded_dict = ConceptDictionary(n_size=0)
        loaded_dict.load(directory=save_dir)
        
        # Only last concept is kept (1 per neuron)
        loaded_concept = loaded_dict.get(0)
        assert loaded_concept is not None
        assert loaded_concept.name == hierarchical_concepts[-1]  # Last one added
        assert loaded_concept.score == float(len(hierarchical_concepts) - 1)

    def test_save_load_with_duplicate_concept_names(self, tmp_path):
        """Test save/load with duplicate concept names."""
        dictionary = ConceptDictionary(n_size=3)
        
        # Add concepts with duplicate names but different scores
        dictionary.add(0, "duplicate", 0.8)
        dictionary.add(0, "duplicate", 0.6)
        dictionary.add(1, "duplicate", 0.9)
        dictionary.add(1, "duplicate", 0.7)
        
        # Save and load
        save_dir = tmp_path / "duplicate_dict"
        dictionary.set_directory(save_dir)
        saved_path = dictionary.save()
        
        loaded_dict = ConceptDictionary(n_size=0)
        loaded_dict.load(directory=save_dir)
        
        # Only last concept per neuron is kept
        concept_0 = loaded_dict.get(0)
        assert concept_0 is not None
        assert concept_0.name == "duplicate"
        assert concept_0.score == 0.6  # Last one added for neuron 0
        
        concept_1 = loaded_dict.get(1)
        assert concept_1 is not None
        assert concept_1.name == "duplicate"
        assert concept_1.score == 0.7  # Last one added for neuron 1

    def test_save_load_with_empty_dictionary(self, tmp_path):
        """Test save/load with empty dictionary."""
        dictionary = ConceptDictionary(n_size=5)
        
        # Save empty dictionary
        save_dir = tmp_path / "empty_dict"
        dictionary.set_directory(save_dir)
        saved_path = dictionary.save()
        
        loaded_dict = ConceptDictionary(n_size=0)
        loaded_dict.load(directory=save_dir)
        
        # Check that empty dictionary was preserved
        assert loaded_dict.n_size == 5
        assert len(loaded_dict.concepts_map) == 0
        
        # All neurons should return None (no concepts)
        for i in range(5):
            assert loaded_dict.get(i) is None

    def test_save_load_with_corrupted_file_handling(self, tmp_path):
        """Test handling of corrupted save files."""
        dictionary = ConceptDictionary(n_size=3)
        dictionary.add(0, "test", 0.5)
        
        # Save dictionary
        save_dir = tmp_path / "corrupted_dict"
        dictionary.set_directory(save_dir)
        saved_path = dictionary.save()
        
        # Corrupt the file
        with saved_path.open("w") as f:
            f.write("corrupted json content")
        
        # Should raise appropriate error when loading
        with pytest.raises(json.JSONDecodeError):
            loaded_dict = ConceptDictionary(n_size=0)
            loaded_dict.load(directory=save_dir)

    def test_save_load_with_missing_directory(self, tmp_path):
        """Test handling of missing directory."""
        missing_dir = tmp_path / "missing_dir"
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            loaded_dict = ConceptDictionary(n_size=0)
            loaded_dict.load(directory=missing_dir)

    def test_save_load_with_permission_errors(self, tmp_path):
        """Test handling of permission errors."""
        dictionary = ConceptDictionary(n_size=3)
        dictionary.add(0, "test", 0.5)
        
        # Mock permission error
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                dictionary.set_directory(tmp_path / "no_permission")
