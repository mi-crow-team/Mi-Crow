#!/usr/bin/env python3
"""
Helper script to discover the number of layers in Bielik 1.5B Instruct model
and suggest appropriate layer numbers for SAE training.

Usage:
    python discover_layers.py
    # or with custom model:
    MODEL_ID="speakleash/Bielik-1.5B-v3.0-Instruct" python discover_layers.py
"""

import os
from transformers import AutoConfig
from amber.language_model.language_model import LanguageModel
from amber.store.local_store import LocalStore
from pathlib import Path

MODEL_ID = os.getenv("MODEL_ID", "speakleash/Bielik-1.5B-v3.0-Instruct")
STORE_DIR = Path(os.getenv("STORE_DIR", "./store"))


def main():
    print(f"🔍 Discovering layer information for: {MODEL_ID}\n")

    # Method 1: Check model config
    print("📋 Method 1: Checking model configuration...")
    try:
        config = AutoConfig.from_pretrained(MODEL_ID)
        num_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "num_layers", None)
        if num_layers:
            print(f"   ✅ Found {num_layers} layers in model config")
            middle_layer = num_layers // 2
            print(f"   💡 Suggested middle layer: {middle_layer}")
        else:
            print("   ⚠️  Could not determine layer count from config")
            num_layers = None
    except Exception as e:
        print(f"   ❌ Error loading config: {e}")
        num_layers = None

    # Method 2: Load model and inspect layers
    print("\n📋 Method 2: Inspecting loaded model...")
    try:
        store = LocalStore(base_path=STORE_DIR)
        lm = LanguageModel.from_huggingface(MODEL_ID, store=store)
        
        # Get all layer names
        layer_names = lm.layers.get_layer_names()
        
        # Filter for post_attention_layernorm layers
        post_attn_layers = [
            name for name in layer_names 
            if "post_attention_layernorm" in name.lower() or "resid_mid" in name.lower()
        ]
        
        if post_attn_layers:
            print(f"   ✅ Found {len(post_attn_layers)} post_attention_layernorm layers")
            
            # Extract layer numbers
            layer_numbers = []
            for layer_name in post_attn_layers:
                # Try to extract layer number from name
                # Format: ..._layers_{N}_post_attention_layernorm
                parts = layer_name.split("_")
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            layer_num = int(parts[i + 1])
                            layer_numbers.append((layer_num, layer_name))
                            break
                        except ValueError:
                            continue
            
            if layer_numbers:
                layer_numbers.sort()
                print(f"\n   📊 Available post_attention_layernorm layers:")
                for layer_num, layer_name in layer_numbers[:10]:  # Show first 10
                    marker = " ← suggested" if layer_num == len(layer_numbers) // 2 else ""
                    print(f"      Layer {layer_num:2d}: {layer_name}{marker}")
                if len(layer_numbers) > 10:
                    print(f"      ... and {len(layer_numbers) - 10} more")
                
                total_layers = len(layer_numbers)
                middle_layer = layer_numbers[total_layers // 2][0]
                print(f"\n   💡 Total layers: {total_layers}")
                print(f"   💡 Suggested middle layer: {middle_layer}")
                print(f"   💡 Layer signature: llamaforcausallm_model_layers_{middle_layer}_post_attention_layernorm")
            else:
                print("   ⚠️  Could not extract layer numbers from layer names")
                print("   📋 First 5 post_attention_layernorm layers found:")
                for layer_name in post_attn_layers[:5]:
                    print(f"      - {layer_name}")
        else:
            print("   ⚠️  No post_attention_layernorm layers found")
            print("   📋 First 10 available layers:")
            for layer_name in layer_names[:10]:
                print(f"      - {layer_name}")
    
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("📝 Summary and Recommendations:")
    print("="*60)
    print(f"   Model: {MODEL_ID}")
    if num_layers:
        print(f"   Config layers: {num_layers}")
        print(f"   Recommended LAYER_NUM: {num_layers // 2}")
    print("\n   💡 For SAE training, use:")
    print(f"      export LAYER_NUM={num_layers // 2 if num_layers else 24}")
    print("\n   📚 See README.md 'Layer Selection for SAE Training' section")
    print("      for detailed explanation of why this layer is recommended.")


if __name__ == "__main__":
    main()

