"""
Debug script to print all available layers for different models.
"""

from pathlib import Path

from mi_crow.language_model.language_model import LanguageModel
from mi_crow.store.local_store import LocalStore

# Setup store
STORE_DIR = Path("store")
store = LocalStore(base_path=STORE_DIR)

# Models to inspect
MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "speakleash/Bielik-4.5B-v3.0-Instruct",
    "speakleash/Bielik-1.5B-v3.0-Instruct",
]

print("=" * 80)
print("MODEL LAYER INSPECTION")
print("=" * 80)
print()

for model_id in MODELS:
    print(f"\n{'=' * 80}")
    print(f"Model: {model_id}")
    print(f"{'=' * 80}\n")

    # Load model
    lm = LanguageModel.from_huggingface(model_id, store=store)

    # Print all layer names
    lm.layers.print_layer_names()

    print()
