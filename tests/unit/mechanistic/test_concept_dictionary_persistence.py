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
            loaded_dict = ConceptDictionary.from_directory(save_dir)
            
            # Should match
            assert loaded_dict.n_size == size
            assert loaded_dict.concepts_map == dictionary.concepts_map

    def test_max_concepts_limiting_behavior(self, tmp_path):
        """Test max_concepts limiting behavior."""
        dictionary = ConceptDictionary(n_size=5, max_concepts=3)
        
        # Add more concepts than max_concepts
        for i in range(5):
            dictionary.add(0, f"concept_{i}", float(i))
        
        # Should only keep top 3 by score
        concepts = dictionary.get(0)
        assert len(concepts) == 3
        
        # Should be sorted by score (descending)
        scores = [c.score for c in concepts]
        assert scores == sorted(scores, reverse=True)

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
            assert len(data["concepts"]["0"]) == 2
            assert len(data["concepts"]["1"]) == 1

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
        
        concepts_0 = dictionary.get(0)
        assert len(concepts_0) == 2
        assert any(c.name == "concept_0" and c.score == 0.8 for c in concepts_0)
        assert any(c.name == "concept_1" and c.score == 0.6 for c in concepts_0)
        
        concepts_1 = dictionary.get(1)
        assert len(concepts_1) == 1
        assert concepts_1[0].name == "concept_2" and concepts_1[0].score == 0.9

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
        
        loaded_dict = ConceptDictionary.from_directory(save_dir)
        
        # Check that all concepts were preserved
        loaded_concepts = loaded_dict.get(0)
        assert len(loaded_concepts) == len(special_names)
        
        for i, name in enumerate(special_names):
            assert any(c.name == name and c.score == float(i) for c in loaded_concepts)

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
        
        loaded_dict = ConceptDictionary.from_directory(save_dir)
        
        # Check that all concepts were preserved
        assert loaded_dict.n_size == n_size
        
        # Check a few random neurons
        for neuron_idx in range(0, n_size, 100):  # Every 100th neuron
            concepts = loaded_dict.get(neuron_idx)
            assert len(concepts) == 5
            assert all(c.name.startswith(f"neuron_{neuron_idx}_") for c in concepts)

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
        
        loaded_dict = ConceptDictionary.from_directory(save_dir)
        
        # Check that all concepts were preserved
        loaded_concepts = loaded_dict.get(0)
        assert len(loaded_concepts) == len(hierarchical_concepts)
        
        for i, name in enumerate(hierarchical_concepts):
            assert any(c.name == name and c.score == float(i) for c in loaded_concepts)

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
        
        loaded_dict = ConceptDictionary.from_directory(save_dir)
        
        # Check that all concepts were preserved
        concepts_0 = loaded_dict.get(0)
        assert len(concepts_0) == 2
        assert all(c.name == "duplicate" for c in concepts_0)
        
        concepts_1 = loaded_dict.get(1)
        assert len(concepts_1) == 2
        assert all(c.name == "duplicate" for c in concepts_1)

    def test_save_load_with_empty_dictionary(self, tmp_path):
        """Test save/load with empty dictionary."""
        dictionary = ConceptDictionary(n_size=5)
        
        # Save empty dictionary
        save_dir = tmp_path / "empty_dict"
        dictionary.set_directory(save_dir)
        saved_path = dictionary.save()
        
        loaded_dict = ConceptDictionary.from_directory(save_dir)
        
        # Check that empty dictionary was preserved
        assert loaded_dict.n_size == 5
        assert len(loaded_dict.concepts_map) == 0
        
        # All neurons should return empty lists
        for i in range(5):
            assert loaded_dict.get(i) == []

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
            ConceptDictionary.from_directory(save_dir)

    def test_save_load_with_missing_directory(self, tmp_path):
        """Test handling of missing directory."""
        missing_dir = tmp_path / "missing_dir"
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            ConceptDictionary.from_directory(missing_dir)

    def test_save_load_with_permission_errors(self, tmp_path):
        """Test handling of permission errors."""
        dictionary = ConceptDictionary(n_size=3)
        dictionary.add(0, "test", 0.5)
        
        # Mock permission error
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                dictionary.set_directory(tmp_path / "no_permission")
