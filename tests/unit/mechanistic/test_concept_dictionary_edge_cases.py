"""Test edge cases and error handling in ConceptDictionary."""

import pytest
from unittest.mock import patch

from amber.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary, Concept
from amber.mechanistic.sae.concepts.concept_models import NeuronText


class TestConceptDictionaryEdgeCases:
    """Test edge cases and error handling functionality."""

    def test_out_of_bounds_index_access(self):
        """Test out-of-bounds index access raises IndexError."""
        dictionary = ConceptDictionary(n_size=5)
        
        # Test negative index
        with pytest.raises(IndexError, match="index -1 out of bounds"):
            dictionary.add(-1, "test", 0.5)
        
        with pytest.raises(IndexError, match="index -1 out of bounds"):
            dictionary.get(-1)
        
        # Test index >= n_size
        with pytest.raises(IndexError, match="index 5 out of bounds"):
            dictionary.add(5, "test", 0.5)
        
        with pytest.raises(IndexError, match="index 5 out of bounds"):
            dictionary.get(5)
        
        with pytest.raises(IndexError, match="index 10 out of bounds"):
            dictionary.add(10, "test", 0.5)

    def test_duplicate_concept_names(self):
        """Test handling of duplicate concept names - only last one is kept."""
        dictionary = ConceptDictionary(n_size=3)
        
        # Add concepts with same name to same neuron
        dictionary.add(0, "duplicate", 0.8)
        dictionary.add(0, "duplicate", 0.6)
        dictionary.add(0, "duplicate", 0.9)
        
        # Only 1 concept per neuron - last one added is kept
        concept = dictionary.get(0)
        assert concept is not None
        assert concept.name == "duplicate"
        assert concept.score == 0.9  # Last one added

    def test_concept_replacement(self):
        """Test that adding a new concept replaces the old one."""
        dictionary = ConceptDictionary(n_size=3)
        
        # Add concepts with different scores
        scores = [0.1, 0.9, 0.3, 0.7, 0.5]
        for i, score in enumerate(scores):
            dictionary.add(0, f"concept_{i}", score)
        
        # Only 1 concept per neuron - last one added is kept
        concept = dictionary.get(0)
        assert concept is not None
        assert concept.name == "concept_4"
        assert concept.score == 0.5  # Last one added

    def test_empty_dictionary_operations(self):
        """Test operations on empty dictionary."""
        dictionary = ConceptDictionary(n_size=5)
        
        # All neurons should return None (no concept)
        for i in range(5):
            assert dictionary.get(i) is None
        
        # get_many should return None values
        many_concepts = dictionary.get_many([0, 1, 2])
        assert many_concepts == {0: None, 1: None, 2: None}
        
        # Should handle empty operations gracefully
        assert len(dictionary.concepts_map) == 0

    def test_concept_replacement_edge_cases(self):
        """Test concept replacement behavior."""
        dictionary = ConceptDictionary(n_size=3)
        
        # Add first concept
        dictionary.add(0, "first", 0.5)
        concept = dictionary.get(0)
        assert concept is not None
        assert concept.name == "first"
        assert concept.score == 0.5
        
        # Replace with higher score
        dictionary.add(0, "second", 0.8)
        concept = dictionary.get(0)
        assert concept is not None
        assert concept.name == "second"
        assert concept.score == 0.8
        
        # Replace with lower score (still replaces)
        dictionary.add(0, "third", 0.3)
        concept = dictionary.get(0)
        assert concept is not None
        assert concept.name == "third"
        assert concept.score == 0.3

    def test_concept_with_extreme_scores(self):
        """Test concepts with extreme score values."""
        dictionary = ConceptDictionary(n_size=3)
        
        # Test with extreme scores
        extreme_scores = [
            float('inf'),
            float('-inf'),
            0.0,
            1e-10,
            1e10,
            -1e-10,
            -1e10,
        ]
        
        for i, score in enumerate(extreme_scores):
            dictionary.add(0, f"concept_{i}", score)
        
        # Only last concept is kept (1 per neuron)
        concept = dictionary.get(0)
        assert concept is not None
        assert concept.name == "concept_6"
        assert concept.score == -1e10  # Last one added

    def test_concept_with_special_characters_in_names(self):
        """Test concepts with special characters in names."""
        dictionary = ConceptDictionary(n_size=3)
        
        # Test with various special characters
        special_names = [
            "",  # Empty string
            " ",  # Space
            "\t",  # Tab
            "\n",  # Newline
            "\r",  # Carriage return
            "concept with spaces",
            "concept\twith\ttabs",
            "concept\nwith\nnewlines",
            "concept\rwith\rcarriage_returns",
            "concept with unicode: ñáéíóú",
            "concept with emoji: 🚀🎉💯",
            "concept with symbols: !@#$%^&*()",
            "concept with quotes: \"double\" and 'single'",
            "concept with brackets: [{}]()",
            "concept with slashes: /\\",
            "concept with backslashes: \\\\",
            "concept with dots: ...",
            "concept with commas: ,,,,,",
            "concept with semicolons: ;;;;",
            "concept with colons: ::::",
        ]
        
        for i, name in enumerate(special_names):
            dictionary.add(0, name, float(i))
        
        # Only last concept is kept (1 per neuron)
        concept = dictionary.get(0)
        assert concept is not None
        assert concept.name == special_names[-1]  # Last one added
        assert concept.score == float(len(special_names) - 1)

    def test_concept_with_very_long_names(self):
        """Test concepts with very long names."""
        dictionary = ConceptDictionary(n_size=3)
        
        # Test with very long names
        long_names = [
            "a" * 1000,  # 1000 characters
            "b" * 10000,  # 10000 characters
            "c" * 100000,  # 100000 characters
        ]
        
        for i, name in enumerate(long_names):
            dictionary.add(0, name, float(i))
        
        # Only last concept is kept (1 per neuron)
        concept = dictionary.get(0)
        assert concept is not None
        assert concept.name == long_names[-1]  # Last one added
        assert concept.score == float(len(long_names) - 1)

    def test_concept_with_none_values(self):
        """Test concepts with None values."""
        dictionary = ConceptDictionary(n_size=3)
        
        # Test with None name (should work, no validation)
        dictionary.add(0, None, 0.5)
        concept = dictionary.get(0)
        assert concept is not None
        assert concept.name is None
        assert concept.score == 0.5
        
        # Test with None score (should work, no validation) - replaces previous
        dictionary.add(0, "test", None)
        concept = dictionary.get(0)
        assert concept is not None
        assert concept.name == "test"
        assert concept.score is None

    def test_concept_with_invalid_types(self):
        """Test concepts with invalid types (no validation, should work)."""
        dictionary = ConceptDictionary(n_size=3)
        
        # Test with invalid name types (should work, no validation)
        dictionary.add(0, 123, 0.5)
        concept = dictionary.get(0)
        assert concept is not None
        assert concept.name == 123
        assert concept.score == 0.5
        
        dictionary.add(0, [], 0.6)
        concept = dictionary.get(0)
        assert concept is not None
        assert concept.name == []
        assert concept.score == 0.6
        
        dictionary.add(0, {}, 0.7)
        concept = dictionary.get(0)
        assert concept is not None
        assert concept.name == {}
        assert concept.score == 0.7
        
        # Test with invalid score types (should work, no validation)
        dictionary.add(1, "test", "invalid")
        concept = dictionary.get(1)
        assert concept is not None
        assert concept.name == "test"
        assert concept.score == "invalid"

    def test_concept_dictionary_with_zero_size(self):
        """Test concept dictionary with zero size."""
        dictionary = ConceptDictionary(n_size=0)
        
        # Should handle zero size
        assert dictionary.n_size == 0
        assert len(dictionary.concepts_map) == 0
        
        # Should raise error when trying to add concepts
        with pytest.raises(IndexError, match="index 0 out of bounds"):
            dictionary.add(0, "test", 0.5)

    def test_concept_dictionary_with_very_large_size(self):
        """Test concept dictionary with very large size."""
        # Test with large size
        large_size = 1000000
        dictionary = ConceptDictionary(n_size=large_size)
        
        # Should handle large size
        assert dictionary.n_size == large_size
        
        # Should be able to add concepts at the end
        dictionary.add(large_size - 1, "test", 0.5)
        concept = dictionary.get(large_size - 1)
        assert concept is not None
        assert concept.name == "test"
        assert concept.score == 0.5

    def test_concept_dictionary_with_negative_size(self):
        """Test concept dictionary with negative size (no validation)."""
        # Should work with negative size (no validation)
        dictionary = ConceptDictionary(n_size=-1)
        assert dictionary.n_size == -1
        
        # Adding concepts should fail due to bounds check in add method
        with pytest.raises(IndexError):
            dictionary.add(0, "test", 0.5)

    def test_concept_dictionary_replacement_behavior(self):
        """Test that adding a concept always replaces the previous one."""
        dictionary = ConceptDictionary(n_size=5)
        
        # Add multiple concepts - only last one is kept
        for i in range(100):
            dictionary.add(0, f"concept_{i}", float(i))
        
        # Should only store the last concept
        concept = dictionary.get(0)
        assert concept is not None
        assert concept.name == "concept_99"
        assert concept.score == 99.0
