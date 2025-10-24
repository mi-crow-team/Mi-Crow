"""
Simplified end-to-end tests for the concept naming workflow.

These tests focus on the core functionality without complex model training.
"""

import tempfile
import json
import csv
from pathlib import Path
import pytest

from amber.mechanistic.autoencoder.concepts.concept_dictionary import ConceptDictionary
from amber.mechanistic.autoencoder.concepts.concept_models import NeuronText
from amber.mechanistic.autoencoder.concepts.autoencoder_concepts import AutoencoderConcepts


class TestSimpleConceptWorkflow:
    """Simplified end-to-end tests for concept naming workflow."""

    def test_concept_dictionary_builders(self):
        """Test ConceptDictionary builder methods (from_csv, from_json, from_llm)."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Test 1: CSV builder
            csv_path = tmp_path / "concepts.csv"
            with csv_path.open("w") as f:
                f.write("neuron_idx,concept_name,score\n")
                f.write("0,animal,0.9\n")
                f.write("0,creature,0.8\n")
                f.write("1,action,0.9\n")
                f.write("1,movement,0.7\n")
                f.write("2,nature,0.9\n")
            
            cd_csv = ConceptDictionary.from_csv(csv_path, n_size=3)
            
            # Verify CSV loading
            assert len(cd_csv.get(0)) == 2  # animal, creature
            assert cd_csv.get(0)[0].name == "animal"
            assert cd_csv.get(0)[0].score == 0.9
            assert cd_csv.get(0)[1].name == "creature"
            assert cd_csv.get(0)[1].score == 0.8
            
            assert len(cd_csv.get(1)) == 2  # action, movement
            assert cd_csv.get(1)[0].name == "action"
            assert cd_csv.get(1)[0].score == 0.9
            
            assert len(cd_csv.get(2)) == 1  # nature
            assert cd_csv.get(2)[0].name == "nature"
            assert cd_csv.get(2)[0].score == 0.9
            
            print("✅ ConceptDictionary.from_csv() works correctly")
            
            # Test 2: JSON builder
            json_path = tmp_path / "concepts.json"
            json_data = {
                "0": [{"name": "color", "score": 0.9}, {"name": "hue", "score": 0.8}],
                "1": [{"name": "emotion", "score": 0.9}, {"name": "feeling", "score": 0.7}],
                "2": [{"name": "object", "score": 0.8}]
            }
            
            with json_path.open("w") as f:
                json.dump(json_data, f)
            
            cd_json = ConceptDictionary.from_json(json_path, n_size=3)
            
            # Verify JSON loading
            assert len(cd_json.get(0)) == 2  # color, hue
            assert cd_json.get(0)[0].name == "color"
            assert cd_json.get(0)[0].score == 0.9
            assert cd_json.get(0)[1].name == "hue"
            assert cd_json.get(0)[1].score == 0.8
            
            assert len(cd_json.get(1)) == 2  # emotion, feeling
            assert cd_json.get(1)[0].name == "emotion"
            assert cd_json.get(1)[0].score == 0.9
            
            assert len(cd_json.get(2)) == 1  # object
            assert cd_json.get(2)[0].name == "object"
            assert cd_json.get(2)[0].score == 0.8
            
            print("✅ ConceptDictionary.from_json() works correctly")
            
            # Test 3: LLM builder (should raise NotImplementedError)
            try:
                neuron_texts = [
                    [NeuronText(score=0.8, text="test text 1", token_idx=0, token_str="test")],
                    [NeuronText(score=0.7, text="test text 2", token_idx=1, token_str="text")],
                    [NeuronText(score=0.6, text="test text 3", token_idx=0, token_str="test")]
                ]
                ConceptDictionary.from_llm(neuron_texts, n_size=3)
                assert False, "Should have raised NotImplementedError"
            except NotImplementedError as e:
                assert "LLM provider not configured" in str(e)
                print("✅ ConceptDictionary.from_llm() properly raises NotImplementedError")

    def test_autoencoder_concepts_integration(self):
        """Test AutoencoderConcepts integration with concept loading."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Create AutoencoderConcepts
            concepts = AutoencoderConcepts(n_size=3)
            
            # Test CSV loading
            csv_path = tmp_path / "concepts.csv"
            with csv_path.open("w") as f:
                f.write("neuron_idx,concept_name,score\n")
                f.write("0,animal,0.9\n")
                f.write("1,action,0.8\n")
                f.write("2,nature,0.7\n")
            
            concepts.load_concepts_from_csv(csv_path)
            
            # Verify concepts were loaded
            assert concepts.dictionary is not None
            assert len(concepts.dictionary.get(0)) == 1
            assert concepts.dictionary.get(0)[0].name == "animal"
            assert concepts.dictionary.get(0)[0].score == 0.9
            
            assert len(concepts.dictionary.get(1)) == 1
            assert concepts.dictionary.get(1)[0].name == "action"
            assert concepts.dictionary.get(1)[0].score == 0.8
            
            assert len(concepts.dictionary.get(2)) == 1
            assert concepts.dictionary.get(2)[0].name == "nature"
            assert concepts.dictionary.get(2)[0].score == 0.7
            
            print("✅ AutoencoderConcepts.load_concepts_from_csv() works")
            
            # Test JSON loading
            json_path = tmp_path / "concepts.json"
            json_data = {
                "0": [{"name": "color", "score": 0.9}],
                "1": [{"name": "emotion", "score": 0.8}],
                "2": [{"name": "object", "score": 0.7}]
            }
            
            with json_path.open("w") as f:
                json.dump(json_data, f)
            
            concepts.load_concepts_from_json(json_path)
            
            # Verify JSON concepts were loaded (should replace previous ones)
            assert len(concepts.dictionary.get(0)) == 1
            assert concepts.dictionary.get(0)[0].name == "color"
            assert concepts.dictionary.get(0)[0].score == 0.9
            
            assert len(concepts.dictionary.get(1)) == 1
            assert concepts.dictionary.get(1)[0].name == "emotion"
            assert concepts.dictionary.get(1)[0].score == 0.8
            
            assert len(concepts.dictionary.get(2)) == 1
            assert concepts.dictionary.get(2)[0].name == "object"
            assert concepts.dictionary.get(2)[0].score == 0.7
            
            print("✅ AutoencoderConcepts.load_concepts_from_json() works")
            
            # Test LLM generation (should raise NotImplementedError)
            try:
                concepts.generate_concepts_with_llm("openai")
                assert False, "Should have raised ValueError (no text tracker)"
            except ValueError as e:
                assert "No text tracker available" in str(e)
                print("✅ AutoencoderConcepts.generate_concepts_with_llm() properly raises ValueError")

    def test_export_functionality(self):
        """Test export functionality for neuron-to-texts mapping."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Create AutoencoderConcepts with mock data
            concepts = AutoencoderConcepts(n_size=3)
            
            # Mock the top_texts_tracker with test data
            class MockTracker:
                def get_all(self):
                    return [
                        [NeuronText(score=0.9, text="The cat sat on the mat", token_idx=1, token_str="cat")],
                        [NeuronText(score=0.8, text="Dogs are loyal animals", token_idx=0, token_str="Dogs")],
                        [NeuronText(score=0.7, text="Birds can fly in the sky", token_idx=0, token_str="Birds")]
                    ]
            
            concepts.top_texts_tracker = MockTracker()
            
            # Test JSON export
            json_path = tmp_path / "neuron_texts.json"
            concepts.export_top_texts_to_json(json_path)
            
            assert json_path.exists()
            
            with json_path.open("r") as f:
                data = json.load(f)
            
            # Verify JSON structure
            assert "0" in data
            assert "1" in data
            assert "2" in data
            
            # Check first neuron data
            neuron_0_data = data["0"]
            assert len(neuron_0_data) == 1
            assert neuron_0_data[0]["text"] == "The cat sat on the mat"
            assert neuron_0_data[0]["score"] == 0.9
            assert neuron_0_data[0]["token_str"] == "cat"
            assert neuron_0_data[0]["token_idx"] == 1
            
            # Check second neuron data
            neuron_1_data = data["1"]
            assert len(neuron_1_data) == 1
            assert neuron_1_data[0]["text"] == "Dogs are loyal animals"
            assert neuron_1_data[0]["score"] == 0.8
            assert neuron_1_data[0]["token_str"] == "Dogs"
            assert neuron_1_data[0]["token_idx"] == 0
            
            print("✅ JSON export works correctly")
            
            # Test CSV export
            csv_path = tmp_path / "neuron_texts.csv"
            concepts.export_top_texts_to_csv(csv_path)
            
            assert csv_path.exists()
            
            with csv_path.open("r") as f:
                lines = f.readlines()
            
            # Verify CSV structure
            assert len(lines) == 4  # Header + 3 data rows
            header = lines[0].strip()
            assert "neuron_idx,text,score,token_str,token_idx" in header
            
            # Check data rows
            data_rows = [line.strip() for line in lines[1:]]
            assert len(data_rows) == 3
            
            # Verify first row
            first_row = data_rows[0].split(",")
            assert first_row[0] == "0"  # neuron_idx
            assert "The cat sat on the mat" in first_row[1]  # text
            assert first_row[2] == "0.9"  # score
            assert first_row[3] == "cat"  # token_str
            assert first_row[4] == "1"  # token_idx
            
            print("✅ CSV export works correctly")

    def test_neuron_text_model(self):
        """Test the enhanced NeuronText model with token information."""
        
        # Test creating NeuronText with token information
        nt = NeuronText(
            score=0.8,
            text="The quick brown fox",
            token_idx=2,
            token_str="brown"
        )
        
        assert nt.score == 0.8
        assert nt.text == "The quick brown fox"
        assert nt.token_idx == 2
        assert nt.token_str == "brown"
        
        # Test that all fields are accessible
        assert hasattr(nt, 'score')
        assert hasattr(nt, 'text')
        assert hasattr(nt, 'token_idx')
        assert hasattr(nt, 'token_str')
        
        print("✅ Enhanced NeuronText model works correctly")

    def test_concept_dictionary_edge_cases(self):
        """Test edge cases for ConceptDictionary builders."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Test empty CSV
            empty_csv_path = tmp_path / "empty.csv"
            with empty_csv_path.open("w") as f:
                f.write("neuron_idx,concept_name,score\n")
            
            cd_empty = ConceptDictionary.from_csv(empty_csv_path, n_size=3)
            assert len(cd_empty.get(0)) == 0
            assert len(cd_empty.get(1)) == 0
            assert len(cd_empty.get(2)) == 0
            
            print("✅ Empty CSV handling works")
            
            # Test CSV with missing columns
            try:
                bad_csv_path = tmp_path / "bad.csv"
                with bad_csv_path.open("w") as f:
                    f.write("neuron_idx,concept_name\n")  # Missing score column
                    f.write("0,animal\n")
                
                ConceptDictionary.from_csv(bad_csv_path, n_size=3)
                assert False, "Should have raised KeyError"
            except KeyError:
                print("✅ CSV with missing columns properly raises KeyError")
            
            # Test JSON with invalid structure (should be handled gracefully)
            bad_json_path = tmp_path / "bad.json"
            with bad_json_path.open("w") as f:
                json.dump({"0": "invalid_structure"}, f)
            
            # This should not raise an error, but should handle gracefully
            cd_bad = ConceptDictionary.from_json(bad_json_path, n_size=3)
            assert len(cd_bad.get(0)) == 0  # Should be empty due to invalid structure
            print("✅ JSON with invalid structure handled gracefully")
            
            # Test file not found
            try:
                ConceptDictionary.from_csv("nonexistent.csv", n_size=3)
                assert False, "Should have raised FileNotFoundError"
            except FileNotFoundError:
                print("✅ FileNotFoundError properly raised for missing files")


if __name__ == "__main__":
    # Run the tests
    test_instance = TestSimpleConceptWorkflow()
    
    print("🧪 Running simplified end-to-end concept naming workflow tests...")
    
    try:
        test_instance.test_concept_dictionary_builders()
        test_instance.test_autoencoder_concepts_integration()
        test_instance.test_export_functionality()
        test_instance.test_neuron_text_model()
        test_instance.test_concept_dictionary_edge_cases()
        
        print("\n🎉 All simplified end-to-end tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
