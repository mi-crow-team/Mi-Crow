#!/usr/bin/env python3
"""Test script to run locally for debugging."""

import sys
sys.path.insert(0, '/mnt/evafs/groups/mi2lab/akaniasty/Mi-Crow')

from experiments.slurm_sae_pipeline.config import PipelineConfig
from pathlib import Path

config_file = Path(__file__).parent / "configs" / "config_bielik12_polemo2.json"
cfg = PipelineConfig.from_json_file(config_file)

print("Config loaded successfully")
print(f"Model: {cfg.model.model_id}")
print(f"Dataset: {cfg.dataset.hf_dataset}")

# Try importing the problematic module
print("\nTrying to import mi_crow.store.store...")
try:
    from mi_crow.store.store import Store, TensorMetadata
    print("✅ Successfully imported Store and TensorMetadata")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
