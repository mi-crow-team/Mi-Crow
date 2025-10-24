"""
Realistic end-to-end test demonstrating the complete concept naming workflow.

This test simulates a realistic scenario where:
1. We have a trained SAE model
2. We collect top activating texts with token information
3. We load concept names from external sources
4. We export the results for analysis
"""

import tempfile
import json
import csv
from pathlib import Path
import pytest

from amber.mechanistic.autoencoder.concepts.concept_dictionary import ConceptDictionary
from amber.mechanistic.autoencoder.concepts.concept_models import NeuronText
from amber.mechanistic.autoencoder.concepts.autoencoder_concepts import AutoencoderConcepts


class TestRealisticConceptWorkflow:
    """Realistic end-to-end test for concept naming workflow."""

    def test_complete_concept_naming_pipeline(self):
        """Test the complete pipeline from data collection to concept naming."""
        
        print("🚀 Starting realistic concept naming workflow test...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Step 1: Simulate collected top activating texts (as if from SAE inference)
            print("📊 Step 1: Simulating collected top activating texts...")
            
            # Create mock neuron texts with realistic data
            mock_neuron_texts = [
                # Neuron 0: Animal-related concepts
                [
                    NeuronText(score=0.95, text="The cat sat on the mat", token_idx=1, token_str="cat"),
                    NeuronText(score=0.90, text="Dogs are loyal animals", token_idx=0, token_str="Dogs"),
                    NeuronText(score=0.85, text="Birds can fly in the sky", token_idx=0, token_str="Birds")
                ],
                # Neuron 1: Action-related concepts  
                [
                    NeuronText(score=0.92, text="The quick brown fox jumps", token_idx=4, token_str="jumps"),
                    NeuronText(score=0.88, text="She runs very fast", token_idx=1, token_str="runs"),
                    NeuronText(score=0.83, text="He walks to the store", token_idx=1, token_str="walks")
                ],
                # Neuron 2: Nature-related concepts
                [
                    NeuronText(score=0.94, text="The ocean is deep and blue", token_idx=1, token_str="ocean"),
                    NeuronText(score=0.89, text="Mountains are tall and majestic", token_idx=0, token_str="Mountains"),
                    NeuronText(score=0.84, text="Trees grow in the forest", token_idx=0, token_str="Trees")
                ],
                # Neuron 3: Emotion-related concepts
                [
                    NeuronText(score=0.91, text="She felt happy and joyful", token_idx=2, token_str="happy"),
                    NeuronText(score=0.86, text="He was sad and depressed", token_idx=2, token_str="sad"),
                    NeuronText(score=0.81, text="They were excited about the trip", token_idx=3, token_str="excited")
                ]
            ]
            
            print(f"✅ Collected texts for {len(mock_neuron_texts)} neurons")
            
            # Step 2: Export neuron-to-texts mapping
            print("📤 Step 2: Exporting neuron-to-texts mapping...")
            
            # Create AutoencoderConcepts and mock the tracker
            concepts = AutoencoderConcepts(n_size=4)
            
            class MockTracker:
                def get_all(self):
                    return mock_neuron_texts
            
            concepts.top_texts_tracker = MockTracker()
            
            # Export to JSON
            json_path = tmp_path / "neuron_texts.json"
            concepts.export_top_texts_to_json(json_path)
            print(f"✅ Exported to JSON: {json_path}")
            
            # Export to CSV
            csv_path = tmp_path / "neuron_texts.csv"
            concepts.export_top_texts_to_csv(csv_path)
            print(f"✅ Exported to CSV: {csv_path}")
            
            # Verify exports
            self._verify_exports(json_path, csv_path)
            
            # Step 3: Create concept mappings (simulating manual annotation)
            print("🏷️ Step 3: Creating concept mappings...")
            
            # Create CSV concept mapping
            csv_concepts_path = tmp_path / "concept_mappings.csv"
            with csv_concepts_path.open("w") as f:
                f.write("neuron_idx,concept_name,score\n")
                f.write("0,animal,0.95\n")
                f.write("0,creature,0.90\n")
                f.write("1,action,0.92\n")
                f.write("1,movement,0.88\n")
                f.write("2,nature,0.94\n")
                f.write("2,environment,0.89\n")
                f.write("3,emotion,0.91\n")
                f.write("3,feeling,0.86\n")
            
            print(f"✅ Created concept mappings CSV: {csv_concepts_path}")
            
            # Create JSON concept mapping
            json_concepts_path = tmp_path / "concept_mappings.json"
            json_concepts_data = {
                "0": [{"name": "animal", "score": 0.95}, {"name": "creature", "score": 0.90}],
                "1": [{"name": "action", "score": 0.92}, {"name": "movement", "score": 0.88}],
                "2": [{"name": "nature", "score": 0.94}, {"name": "environment", "score": 0.89}],
                "3": [{"name": "emotion", "score": 0.91}, {"name": "feeling", "score": 0.86}]
            }
            
            with json_concepts_path.open("w") as f:
                json.dump(json_concepts_data, f)
            
            print(f"✅ Created concept mappings JSON: {json_concepts_path}")
            
            # Step 4: Load concepts into ConceptDictionary
            print("📚 Step 4: Loading concepts into ConceptDictionary...")
            
            # Test CSV loading
            cd_csv = ConceptDictionary.from_csv(csv_concepts_path, n_size=4)
            self._verify_concept_dictionary(cd_csv, "CSV")
            
            # Test JSON loading
            cd_json = ConceptDictionary.from_json(json_concepts_path, n_size=4)
            self._verify_concept_dictionary(cd_json, "JSON")
            
            # Step 5: Test AutoencoderConcepts integration
            print("🔗 Step 5: Testing AutoencoderConcepts integration...")
            
            # Load concepts through AutoencoderConcepts
            concepts.load_concepts_from_csv(csv_concepts_path)
            self._verify_autoencoder_concepts(concepts, "CSV")
            
            concepts.load_concepts_from_json(json_concepts_path)
            self._verify_autoencoder_concepts(concepts, "JSON")
            
            # Step 6: Test LLM concept generation (stub)
            print("🤖 Step 6: Testing LLM concept generation stub...")
            
            # Test with mock tracker
            concepts.top_texts_tracker = MockTracker()
            try:
                concepts.generate_concepts_with_llm("openai")
                assert False, "Should have raised NotImplementedError"
            except NotImplementedError as e:
                assert "LLM provider not configured" in str(e)
                print("✅ LLM concept generation properly raises NotImplementedError")
            
            print("\n🎉 Complete realistic concept naming workflow test passed!")
            print("📋 Summary:")
            print(f"   - Collected texts for {len(mock_neuron_texts)} neurons")
            print(f"   - Exported neuron-to-texts mapping to JSON and CSV")
            print(f"   - Created concept mappings for {len(mock_neuron_texts)} neurons")
            print(f"   - Successfully loaded concepts via CSV and JSON")
            print(f"   - Verified AutoencoderConcepts integration")
            print(f"   - Tested LLM concept generation stub")

    def _verify_exports(self, json_path: Path, csv_path: Path):
        """Verify that the exports contain the expected data."""
        
        # Verify JSON export
        with json_path.open("r") as f:
            json_data = json.load(f)
        
        assert len(json_data) == 4, f"Expected 4 neurons, got {len(json_data)}"
        
        # Check neuron 0 (animal concepts)
        neuron_0 = json_data["0"]
        assert len(neuron_0) == 3, f"Expected 3 texts for neuron 0, got {len(neuron_0)}"
        assert neuron_0[0]["text"] == "The cat sat on the mat"
        assert neuron_0[0]["token_str"] == "cat"
        assert neuron_0[0]["token_idx"] == 1
        
        # Check neuron 1 (action concepts)
        neuron_1 = json_data["1"]
        assert len(neuron_1) == 3, f"Expected 3 texts for neuron 1, got {len(neuron_1)}"
        assert neuron_1[0]["text"] == "The quick brown fox jumps"
        assert neuron_1[0]["token_str"] == "jumps"
        assert neuron_1[0]["token_idx"] == 4
        
        print("✅ JSON export verification passed")
        
        # Verify CSV export
        with csv_path.open("r") as f:
            lines = f.readlines()
        
        assert len(lines) == 13, f"Expected 13 lines (header + 12 data), got {len(lines)}"
        
        # Check header
        header = lines[0].strip()
        assert "neuron_idx,text,score,token_str,token_idx" in header
        
        # Check data rows
        data_lines = [line.strip() for line in lines[1:]]
        assert len(data_lines) == 12, f"Expected 12 data rows, got {len(data_lines)}"
        
        print("✅ CSV export verification passed")

    def _verify_concept_dictionary(self, cd: ConceptDictionary, source: str):
        """Verify that the ConceptDictionary contains the expected concepts."""
        
        # Check neuron 0 (animal concepts)
        concepts_0 = cd.get(0)
        assert len(concepts_0) == 2, f"Expected 2 concepts for neuron 0 from {source}"
        assert concepts_0[0].name == "animal"
        assert concepts_0[0].score == 0.95
        assert concepts_0[1].name == "creature"
        assert concepts_0[1].score == 0.90
        
        # Check neuron 1 (action concepts)
        concepts_1 = cd.get(1)
        assert len(concepts_1) == 2, f"Expected 2 concepts for neuron 1 from {source}"
        assert concepts_1[0].name == "action"
        assert concepts_1[0].score == 0.92
        assert concepts_1[1].name == "movement"
        assert concepts_1[1].score == 0.88
        
        # Check neuron 2 (nature concepts)
        concepts_2 = cd.get(2)
        assert len(concepts_2) == 2, f"Expected 2 concepts for neuron 2 from {source}"
        assert concepts_2[0].name == "nature"
        assert concepts_2[0].score == 0.94
        assert concepts_2[1].name == "environment"
        assert concepts_2[1].score == 0.89
        
        # Check neuron 3 (emotion concepts)
        concepts_3 = cd.get(3)
        assert len(concepts_3) == 2, f"Expected 2 concepts for neuron 3 from {source}"
        assert concepts_3[0].name == "emotion"
        assert concepts_3[0].score == 0.91
        assert concepts_3[1].name == "feeling"
        assert concepts_3[1].score == 0.86
        
        print(f"✅ ConceptDictionary verification passed for {source}")

    def _verify_autoencoder_concepts(self, concepts: AutoencoderConcepts, source: str):
        """Verify that AutoencoderConcepts has loaded the concepts correctly."""
        
        assert concepts.dictionary is not None, f"Dictionary should be loaded from {source}"
        
        # Check a few key concepts
        concepts_0 = concepts.dictionary.get(0)
        assert len(concepts_0) == 2, f"Expected 2 concepts for neuron 0 from {source}"
        assert concepts_0[0].name == "animal"
        assert concepts_0[0].score == 0.95
        
        concepts_1 = concepts.dictionary.get(1)
        assert len(concepts_1) == 2, f"Expected 2 concepts for neuron 1 from {source}"
        assert concepts_1[0].name == "action"
        assert concepts_1[0].score == 0.92
        
        print(f"✅ AutoencoderConcepts verification passed for {source}")


if __name__ == "__main__":
    # Run the realistic test
    test_instance = TestRealisticConceptWorkflow()
    
    try:
        test_instance.test_complete_concept_naming_pipeline()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
