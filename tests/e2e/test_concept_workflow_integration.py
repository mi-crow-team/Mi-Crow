"""
Integration test for the complete concept naming workflow.

This test focuses on the core functionality without complex dependencies.
"""

import tempfile
import json
import csv
from pathlib import Path

from amber.mechanistic.autoencoder.concepts.concept_dictionary import ConceptDictionary
from amber.mechanistic.autoencoder.concepts.concept_models import NeuronText
from amber.mechanistic.autoencoder.concepts.autoencoder_concepts import AutoencoderConcepts


class TestConceptWorkflowIntegration:
    """Integration test for concept naming workflow."""

    def test_complete_concept_workflow_integration(self):
        """Test the complete concept naming workflow integration."""
        
        print("🚀 Starting concept workflow integration test...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Step 1: Simulate collected neuron texts (as if from SAE inference)
            print("📊 Step 1: Simulating collected neuron texts...")
            
            mock_neuron_texts = [
                # Neuron 0: Animal concepts
                [
                    NeuronText(score=0.95, text="The cat sat on the mat", token_idx=1, token_str="cat"),
                    NeuronText(score=0.90, text="Dogs are loyal animals", token_idx=0, token_str="Dogs"),
                    NeuronText(score=0.85, text="Birds can fly in the sky", token_idx=0, token_str="Birds")
                ],
                # Neuron 1: Action concepts
                [
                    NeuronText(score=0.92, text="The quick brown fox jumps", token_idx=4, token_str="jumps"),
                    NeuronText(score=0.88, text="She runs very fast", token_idx=1, token_str="runs"),
                    NeuronText(score=0.83, text="He walks to the store", token_idx=1, token_str="walks")
                ],
                # Neuron 2: Nature concepts
                [
                    NeuronText(score=0.94, text="The ocean is deep and blue", token_idx=1, token_str="ocean"),
                    NeuronText(score=0.89, text="Mountains are tall and majestic", token_idx=0, token_str="Mountains"),
                    NeuronText(score=0.84, text="Trees grow in the forest", token_idx=0, token_str="Trees")
                ]
            ]
            
            print(f"✅ Simulated texts for {len(mock_neuron_texts)} neurons")
            
            # Step 2: Test ConceptDictionary builder methods
            print("🏗️ Step 2: Testing ConceptDictionary builder methods...")
            
            # Test CSV builder
            csv_path = tmp_path / "concepts.csv"
            with csv_path.open("w") as f:
                f.write("neuron_idx,concept_name,score\n")
                f.write("0,animal,0.95\n")
                f.write("0,creature,0.90\n")
                f.write("1,action,0.92\n")
                f.write("1,movement,0.88\n")
                f.write("2,nature,0.94\n")
                f.write("2,environment,0.89\n")
            
            cd_csv = ConceptDictionary.from_csv(csv_path, n_size=3)
            self._verify_concept_dictionary(cd_csv, "CSV")
            print("✅ ConceptDictionary.from_csv() works")
            
            # Test JSON builder
            json_path = tmp_path / "concepts.json"
            json_data = {
                "0": [{"name": "animal", "score": 0.95}, {"name": "creature", "score": 0.90}],
                "1": [{"name": "action", "score": 0.92}, {"name": "movement", "score": 0.88}],
                "2": [{"name": "nature", "score": 0.94}, {"name": "environment", "score": 0.89}]
            }
            
            with json_path.open("w") as f:
                json.dump(json_data, f)
            
            cd_json = ConceptDictionary.from_json(json_path, n_size=3)
            self._verify_concept_dictionary(cd_json, "JSON")
            print("✅ ConceptDictionary.from_json() works")
            
            # Test LLM builder (should raise NotImplementedError)
            try:
                ConceptDictionary.from_llm(mock_neuron_texts, n_size=3)
                assert False, "Should have raised NotImplementedError"
            except NotImplementedError as e:
                assert "LLM provider not configured" in str(e)
                print("✅ ConceptDictionary.from_llm() properly raises NotImplementedError")
            
            # Step 3: Test AutoencoderConcepts integration
            print("🔗 Step 3: Testing AutoencoderConcepts integration...")
            
            concepts = AutoencoderConcepts(n_size=3)
            
            # Mock the text tracker
            class MockTracker:
                def get_all(self):
                    return mock_neuron_texts
            
            concepts.top_texts_tracker = MockTracker()
            
            # Test CSV loading
            concepts.load_concepts_from_csv(csv_path)
            self._verify_autoencoder_concepts(concepts, "CSV")
            print("✅ AutoencoderConcepts.load_concepts_from_csv() works")
            
            # Test JSON loading
            concepts.load_concepts_from_json(json_path)
            self._verify_autoencoder_concepts(concepts, "JSON")
            print("✅ AutoencoderConcepts.load_concepts_from_json() works")
            
            # Test LLM generation (should raise NotImplementedError)
            try:
                concepts.generate_concepts_with_llm("openai")
                assert False, "Should have raised NotImplementedError"
            except NotImplementedError as e:
                assert "LLM provider not configured" in str(e)
                print("✅ AutoencoderConcepts.generate_concepts_with_llm() properly raises NotImplementedError")
            
            # Step 4: Test export functionality
            print("📤 Step 4: Testing export functionality...")
            
            # Export to JSON
            export_json_path = tmp_path / "neuron_texts.json"
            concepts.export_top_texts_to_json(export_json_path)
            self._verify_json_export(export_json_path)
            print("✅ JSON export works")
            
            # Export to CSV
            export_csv_path = tmp_path / "neuron_texts.csv"
            concepts.export_top_texts_to_csv(export_csv_path)
            self._verify_csv_export(export_csv_path)
            print("✅ CSV export works")
            
            # Step 5: Test edge cases
            print("🔍 Step 5: Testing edge cases...")
            self._test_edge_cases(tmp_path)
            print("✅ Edge cases handled correctly")
            
            print("\n🎉 Complete concept workflow integration test passed!")
            print("📋 Summary:")
            print("   - ConceptDictionary builder methods work correctly")
            print("   - AutoencoderConcepts integration works correctly")
            print("   - Export functionality works correctly")
            print("   - Edge cases are handled gracefully")
            print("   - LLM integration properly raises NotImplementedError")

    def _verify_concept_dictionary(self, cd: ConceptDictionary, source: str):
        """Verify ConceptDictionary contains expected concepts."""
        
        # Check neuron 0
        concepts_0 = cd.get(0)
        assert len(concepts_0) == 2, f"Expected 2 concepts for neuron 0 from {source}"
        assert concepts_0[0].name == "animal"
        assert concepts_0[0].score == 0.95
        assert concepts_0[1].name == "creature"
        assert concepts_0[1].score == 0.90
        
        # Check neuron 1
        concepts_1 = cd.get(1)
        assert len(concepts_1) == 2, f"Expected 2 concepts for neuron 1 from {source}"
        assert concepts_1[0].name == "action"
        assert concepts_1[0].score == 0.92
        assert concepts_1[1].name == "movement"
        assert concepts_1[1].score == 0.88
        
        # Check neuron 2
        concepts_2 = cd.get(2)
        assert len(concepts_2) == 2, f"Expected 2 concepts for neuron 2 from {source}"
        assert concepts_2[0].name == "nature"
        assert concepts_2[0].score == 0.94
        assert concepts_2[1].name == "environment"
        assert concepts_2[1].score == 0.89

    def _verify_autoencoder_concepts(self, concepts: AutoencoderConcepts, source: str):
        """Verify AutoencoderConcepts has loaded concepts correctly."""
        
        assert concepts.dictionary is not None, f"Dictionary should be loaded from {source}"
        
        # Check a few key concepts
        concepts_0 = concepts.dictionary.get(0)
        assert len(concepts_0) == 2, f"Expected 2 concepts for neuron 0 from {source}"
        assert concepts_0[0].name == "animal"
        assert concepts_0[0].score == 0.95

    def _verify_json_export(self, json_path: Path):
        """Verify JSON export structure."""
        
        with json_path.open("r") as f:
            data = json.load(f)
        
        assert len(data) == 3, f"Expected 3 neurons, got {len(data)}"
        
        # Check neuron 0
        neuron_0 = data["0"]
        assert len(neuron_0) == 3, f"Expected 3 texts for neuron 0, got {len(neuron_0)}"
        assert neuron_0[0]["text"] == "The cat sat on the mat"
        assert neuron_0[0]["token_str"] == "cat"
        assert neuron_0[0]["token_idx"] == 1
        
        # Check neuron 1
        neuron_1 = data["1"]
        assert len(neuron_1) == 3, f"Expected 3 texts for neuron 1, got {len(neuron_1)}"
        assert neuron_1[0]["text"] == "The quick brown fox jumps"
        assert neuron_1[0]["token_str"] == "jumps"
        assert neuron_1[0]["token_idx"] == 4

    def _verify_csv_export(self, csv_path: Path):
        """Verify CSV export structure."""
        
        with csv_path.open("r") as f:
            lines = f.readlines()
        
        assert len(lines) == 10, f"Expected 10 lines (header + 9 data), got {len(lines)}"
        
        # Check header
        header = lines[0].strip()
        assert "neuron_idx,text,score,token_str,token_idx" in header
        
        # Check data rows
        data_lines = [line.strip() for line in lines[1:]]
        assert len(data_lines) == 9, f"Expected 9 data rows, got {len(data_lines)}"

    def _test_edge_cases(self, tmp_path: Path):
        """Test edge cases."""
        
        # Test empty CSV
        empty_csv_path = tmp_path / "empty.csv"
        with empty_csv_path.open("w") as f:
            f.write("neuron_idx,concept_name,score\n")
        
        cd_empty = ConceptDictionary.from_csv(empty_csv_path, n_size=3)
        assert len(cd_empty.get(0)) == 0
        assert len(cd_empty.get(1)) == 0
        assert len(cd_empty.get(2)) == 0
        
        # Test CSV with missing columns
        try:
            bad_csv_path = tmp_path / "bad.csv"
            with bad_csv_path.open("w") as f:
                f.write("neuron_idx,concept_name\n")  # Missing score column
                f.write("0,animal\n")
            
            ConceptDictionary.from_csv(bad_csv_path, n_size=3)
            assert False, "Should have raised KeyError"
        except KeyError:
            pass  # Expected
        
        # Test JSON with invalid structure
        bad_json_path = tmp_path / "bad.json"
        with bad_json_path.open("w") as f:
            json.dump({"0": "invalid_structure"}, f)
        
        cd_bad = ConceptDictionary.from_json(bad_json_path, n_size=3)
        assert len(cd_bad.get(0)) == 0  # Should be empty due to invalid structure
        
        # Test file not found
        try:
            ConceptDictionary.from_csv("nonexistent.csv", n_size=3)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected


if __name__ == "__main__":
    # Run the integration test
    test_instance = TestConceptWorkflowIntegration()
    
    try:
        test_instance.test_complete_concept_workflow_integration()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
