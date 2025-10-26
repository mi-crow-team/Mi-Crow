"""Test edge cases and error handling in ConceptDictionary."""

import pytest
from unittest.mock import patch

from amber.mechanistic.autoencoder.concepts.concept_dictionary import ConceptDictionary, Concept
from amber.mechanistic.autoencoder.concepts.concept_models import NeuronText


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
        """Test handling of duplicate concept names."""
        dictionary = ConceptDictionary(n_size=3)
        
        # Add concepts with same name to same neuron
        dictionary.add(0, "duplicate", 0.8)
        dictionary.add(0, "duplicate", 0.6)
        dictionary.add(0, "duplicate", 0.9)
        
        # Should all be stored (not necessarily sorted unless max_concepts is set)
        concepts = dictionary.get(0)
        assert len(concepts) == 3
        assert all(c.name == "duplicate" for c in concepts)
        
        # Concepts are stored in order of addition, not sorted by score
        scores = [c.score for c in concepts]
        assert scores == [0.8, 0.6, 0.9]  # Order of addition

    def test_concept_sorting_by_score(self):
        """Test that concepts are sorted by score when max_concepts is set."""
        dictionary = ConceptDictionary(n_size=3, max_concepts=3)
        
        # Add concepts with different scores
        scores = [0.1, 0.9, 0.3, 0.7, 0.5]
        for i, score in enumerate(scores):
            dictionary.add(0, f"concept_{i}", score)
        
        # Should be sorted by score (descending) due to max_concepts limiting
        concepts = dictionary.get(0)
        assert len(concepts) == 3  # Limited by max_concepts
        
        concept_scores = [c.score for c in concepts]
        assert concept_scores == sorted(scores, reverse=True)[:3]  # Top 3 scores

    def test_empty_dictionary_operations(self):
        """Test operations on empty dictionary."""
        dictionary = ConceptDictionary(n_size=5)
        
        # All neurons should return empty lists
        for i in range(5):
            assert dictionary.get(i) == []
        
        # get_many should return empty dicts
        many_concepts = dictionary.get_many([0, 1, 2])
        assert many_concepts == {0: [], 1: [], 2: []}
        
        # Should handle empty operations gracefully
        assert len(dictionary.concepts_map) == 0

    def test_max_concepts_edge_cases(self):
        """Test max_concepts with edge cases."""
        # Test max_concepts = 0
        dictionary = ConceptDictionary(n_size=3, max_concepts=0)
        dictionary.add(0, "test", 0.5)
        
        # Should not store any concepts
        assert dictionary.get(0) == []
        
        # Test max_concepts = 1
        dictionary = ConceptDictionary(n_size=3, max_concepts=1)
        dictionary.add(0, "first", 0.5)
        dictionary.add(0, "second", 0.8)
        dictionary.add(0, "third", 0.3)
        
        # Should only keep the highest scoring concept
        concepts = dictionary.get(0)
        assert len(concepts) == 1
        assert concepts[0].name == "second"
        assert concepts[0].score == 0.8

    def test_concept_with_extreme_scores(self):
        """Test concepts with extreme score values."""
        dictionary = ConceptDictionary(n_size=3, max_concepts=3)
        
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
        
        # Should handle extreme scores (only top 3 due to max_concepts)
        concepts = dictionary.get(0)
        assert len(concepts) == 3  # Limited by max_concepts
        
        # Should be sorted correctly (inf > finite > -inf)
        concept_scores = [c.score for c in concepts]
        assert concept_scores[0] == float('inf')  # Highest
        assert concept_scores[-1] == 1e-10  # Third highest

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
        
        # Should handle all special characters
        concepts = dictionary.get(0)
        assert len(concepts) == len(special_names)
        
        # Check that all names were preserved
        concept_names = [c.name for c in concepts]
        for name in special_names:
            assert name in concept_names

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
        
        # Should handle very long names
        concepts = dictionary.get(0)
        assert len(concepts) == len(long_names)
        
        # Check that all names were preserved
        concept_names = [c.name for c in concepts]
        for name in long_names:
            assert name in concept_names

    def test_concept_with_none_values(self):
        """Test concepts with None values."""
        dictionary = ConceptDictionary(n_size=3)
        
        # Test with None name (should work, no validation)
        dictionary.add(0, None, 0.5)
        concepts = dictionary.get(0)
        assert len(concepts) == 1
        assert concepts[0].name is None
        assert concepts[0].score == 0.5
        
        # Test with None score (should work, no validation)
        dictionary.add(0, "test", None)
        concepts = dictionary.get(0)
        assert len(concepts) == 2
        assert concepts[1].name == "test"
        assert concepts[1].score is None

    def test_concept_with_invalid_types(self):
        """Test concepts with invalid types (no validation, should work)."""
        dictionary = ConceptDictionary(n_size=3)
        
        # Test with invalid name types (should work, no validation)
        dictionary.add(0, 123, 0.5)
        concepts = dictionary.get(0)
        assert len(concepts) == 1
        assert concepts[0].name == 123
        assert concepts[0].score == 0.5
        
        dictionary.add(0, [], 0.6)
        concepts = dictionary.get(0)
        assert len(concepts) == 2
        assert concepts[1].name == []
        assert concepts[1].score == 0.6
        
        dictionary.add(0, {}, 0.7)
        concepts = dictionary.get(0)
        assert len(concepts) == 3
        assert concepts[2].name == {}
        assert concepts[2].score == 0.7
        
        # Test with invalid score types (should work, no validation)
        dictionary.add(1, "test", "invalid")
        concepts = dictionary.get(1)
        assert len(concepts) == 1
        assert concepts[0].name == "test"
        assert concepts[0].score == "invalid"

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
        concepts = dictionary.get(large_size - 1)
        assert len(concepts) == 1
        assert concepts[0].name == "test"
        assert concepts[0].score == 0.5

    def test_concept_dictionary_with_negative_size(self):
        """Test concept dictionary with negative size (no validation)."""
        # Should work with negative size (no validation)
        dictionary = ConceptDictionary(n_size=-1)
        assert dictionary.n_size == -1
        
        # Adding concepts should fail due to bounds check in add method
        with pytest.raises(IndexError):
            dictionary.add(0, "test", 0.5)

    def test_concept_dictionary_with_negative_max_concepts(self):
        """Test concept dictionary with negative max_concepts (no validation)."""
        # Should work with negative max_concepts (no validation)
        dictionary = ConceptDictionary(n_size=5, max_concepts=-1)
        assert dictionary.max_concepts == -1
        
        # Adding concepts should work but be deleted due to negative max_concepts
        dictionary.add(0, "test", 0.5)
        concepts = dictionary.get(0)
        assert len(concepts) == 0  # All concepts deleted due to negative max_concepts

    def test_concept_dictionary_with_none_max_concepts(self):
        """Test concept dictionary with None max_concepts."""
        dictionary = ConceptDictionary(n_size=5, max_concepts=None)
        
        # Should handle None max_concepts (unlimited)
        for i in range(100):
            dictionary.add(0, f"concept_{i}", float(i))
        
        # Should store all concepts
        concepts = dictionary.get(0)
        assert len(concepts) == 100

    def test_concept_dictionary_with_float_max_concepts(self):
        """Test concept dictionary with float max_concepts (no validation)."""
        # Should work with float max_concepts (no validation)
        dictionary = ConceptDictionary(n_size=5, max_concepts=3.5)
        assert dictionary.max_concepts == 3.5
        
        # Adding concepts should work normally
        dictionary.add(0, "test", 0.5)
        concepts = dictionary.get(0)
        assert len(concepts) == 1

    def test_concept_dictionary_with_string_max_concepts(self):
        """Test concept dictionary with string max_concepts (causes TypeError)."""
        # Should work with string max_concepts (no validation)
        dictionary = ConceptDictionary(n_size=5, max_concepts="3")
        assert dictionary.max_concepts == "3"
        
        # Adding concepts should fail due to type comparison error
        with pytest.raises(TypeError):
            dictionary.add(0, "test", 0.5)
