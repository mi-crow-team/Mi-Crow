#!/usr/bin/env python3
"""
SLURM script to run sample inferences with SAEs and extract feature activations.

This script:
- Runs inference on test sentences
- Extracts top-k activating features per token
- Maps feature indices to concept names from dictionaries
- Generates structured output

Usage:
    python 08_run_sample_inferences.py --sae_paths path1.pt path2.pt --test_sentences_file sentences.txt
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

from config import PipelineConfig
from mi_crow.language_model.language_model import LanguageModel
from mi_crow.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary
from mi_crow.mechanistic.sae.sae import Sae
from mi_crow.mechanistic.sae.modules.topk_sae import TopKSae
from mi_crow.mechanistic.sae.modules.l1_sae import L1Sae
from mi_crow.store.local_store import LocalStore
from mi_crow.utils import get_logger

try:
    from server.utils import SAERegistry
except ImportError:
    SAERegistry = None

logger = get_logger(__name__)

script_dir = Path(__file__).parent
project_root = script_dir
while project_root != project_root.parent:
    if (project_root / "pyproject.toml").exists() or (project_root / ".git").exists():
        break
    project_root = project_root.parent

env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)


def load_sae_auto(sae_path: Path, device: str = "cpu") -> Sae:
    """
    Load SAE and auto-detect class from metadata.json or model file.
    """
    sae_path = Path(sae_path)
    if not sae_path.exists():
        raise ValueError(f"SAE file not found: {sae_path}")
    
    metadata_path = sae_path.parent / "metadata.json"
    target_device = torch.device(device)
    
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text())
            sae_class = meta.get("sae_class") or meta.get("sae_type")
            if sae_class:
                if SAERegistry is not None:
                    try:
                        registry = SAERegistry()
                        cls = registry.get_class(sae_class)
                        sae = cls.load(sae_path)
                        sae.sae_engine.to(target_device)
                        sae.context.device = device
                        return sae
                    except Exception as e:
                        logger.warning(f"Could not use SAERegistry for '{sae_class}': {e}, trying direct load")
                
                if sae_class == "TopKSae":
                    sae = TopKSae.load(sae_path)
                    sae.sae_engine.to(target_device)
                    sae.context.device = device
                    return sae
                elif sae_class == "L1Sae":
                    sae = L1Sae.load(sae_path)
                    sae.sae_engine.to(target_device)
                    sae.context.device = device
                    return sae
        except Exception as e:
            logger.warning(f"Could not read metadata.json: {e}, trying auto-detection")
    
    try:
        sae = TopKSae.load(sae_path)
        sae.sae_engine.to(target_device)
        sae.context.device = device
        return sae
    except Exception:
        sae = L1Sae.load(sae_path)
        sae.sae_engine.to(target_device)
        sae.context.device = device
        return sae


def run_inference_with_sae(
    lm: LanguageModel,
    sae: Sae,
    layer_sig: str,
    sentence: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Run inference on sentence and extract top-k features per token.
    
    Args:
        lm: Language model
        sae: SAE model
        layer_sig: Layer signature
        sentence: Input sentence
        top_k: Number of top features to extract per token
        
    Returns:
        List of token-level feature activations
    """
    device = lm.context.device
    inputs = lm.tokenizer.encode(sentence, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    input_ids = inputs["input_ids"][0]
    tokens = lm.tokenizer.convert_ids_to_tokens(input_ids)
    
    sae.eval()
    lm.model.eval()
    
    captured_activations = None
    
    def capture_hook(module, input, output):
        nonlocal captured_activations
        if isinstance(output, torch.Tensor):
            tensor = output
        elif hasattr(output, "last_hidden_state"):
            tensor = output.last_hidden_state
        elif isinstance(output, (tuple, list)):
            tensor = next((item for item in output if isinstance(item, torch.Tensor)), None)
        else:
            return output
        
        if tensor is not None:
            captured_activations = tensor.detach()
        return output
    
    hook_handle = None
    try:
        if layer_sig in lm.layers.name_to_layer:
            layer = lm.layers.name_to_layer[layer_sig]
            hook_handle = layer.register_forward_hook(capture_hook)
        
        with torch.no_grad():
            _ = lm.model(**inputs)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    
    if captured_activations is None:
        return []
    
    seq_len = captured_activations.shape[1]
    results = []
    
    with torch.no_grad():
        for token_idx in range(seq_len):
            activation = captured_activations[0, token_idx, :]
            codes = sae.encode(activation.unsqueeze(0))
            
            top_k_values, top_k_indices = torch.topk(torch.abs(codes[0]), k=min(top_k, codes.shape[1]))
            
            token_features = []
            for i in range(len(top_k_indices)):
                feature_idx = int(top_k_indices[i].item())
                feature_value = float(codes[0, feature_idx].item())
                token_features.append({
                    "feature_idx": feature_idx,
                    "activation": feature_value,
                    "abs_activation": float(top_k_values[i].item()),
                })
            
            results.append({
                "token_idx": token_idx,
                "token": tokens[token_idx] if token_idx < len(tokens) else "",
                "features": token_features,
            })
    
    return results


def annotate_sentence_with_features(
    sentence: str,
    feature_activations: List[Dict[str, Any]],
    concept_dict: Optional[ConceptDictionary] = None,
) -> Dict[str, Any]:
    """
    Map feature indices to concept names and create structured output.
    
    Args:
        sentence: Input sentence
        feature_activations: List of token-level feature activations
        concept_dict: Concept dictionary for name lookup
        
    Returns:
        Annotated sentence with concept names
    """
    annotated = {
        "sentence": sentence,
        "tokens": [],
    }
    
    for token_data in feature_activations:
        token_info = {
            "token": token_data["token"],
            "token_idx": token_data["token_idx"],
            "features": [],
        }
        
        for feature_data in token_data["features"]:
            feature_info = {
                "feature_idx": feature_data["feature_idx"],
                "activation": feature_data["activation"],
                "abs_activation": feature_data["abs_activation"],
            }
            
            if concept_dict:
                concept = concept_dict.get(feature_data["feature_idx"])
                if concept:
                    feature_info["concept_name"] = concept.name
                    feature_info["concept_score"] = concept.score
            
            token_info["features"].append(feature_info)
        
        annotated["tokens"].append(token_info)
    
    return annotated


def main():
    parser = argparse.ArgumentParser(description="Run sample inferences with SAEs")
    parser.add_argument("--sae_paths", type=str, nargs="+", required=True, help="Paths to SAE model files")
    parser.add_argument("--layer_signatures", type=str, nargs="+", required=True, help="Layer signatures for each SAE")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file")
    parser.add_argument("--test_sentences_file", type=str, default=None, help="File with test sentences (one per line)")
    parser.add_argument("--concept_dicts_dir", type=str, default=None, help="Directory containing concept dictionaries")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top features per token")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    if args.config:
        config_file = Path(args.config)
    else:
        config_file = script_dir / "config.json"
    
    cfg = PipelineConfig.from_json_file(config_file)
    
    MODEL_ID = cfg.model.model_id
    STORE_DIR = Path(cfg.storage.store_dir or str(script_dir / "store"))
    DEVICE = cfg.storage.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = script_dir / "inference_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(args.sae_paths) != len(args.layer_signatures):
        logger.error("❌ Error: Number of SAE paths must match number of layer signatures")
        return
    
    if args.test_sentences_file:
        with open(args.test_sentences_file, "r", encoding="utf-8") as f:
            test_sentences = [line.strip() for line in f if line.strip()]
    else:
        test_sentences = [
            "Pies szczekał na psa sąsiada.",
            "W 1410 roku odbyła się bitwa pod Grunwaldem.",
            "Napisz mi maila do szefa, że jestem chory.",
        ]
    
    logger.info("🚀 Starting Sample Inference")
    logger.info(f"📱 Using device: {DEVICE}")
    logger.info(f"🔧 Model: {MODEL_ID}")
    logger.info(f"📝 Test sentences: {len(test_sentences)}")
    
    logger.info("📥 Loading language model...")
    store = LocalStore(base_path=STORE_DIR)
    lm = LanguageModel.from_huggingface(MODEL_ID, store=store, device=DEVICE)
    lm.model.eval()
    
    all_results = {}
    
    for sae_path_str, layer_sig in zip(args.sae_paths, args.layer_signatures):
        sae_path = Path(sae_path_str)
        logger.info(f"📥 Loading SAE from {sae_path} for layer {layer_sig}...")
        
        sae = load_sae_auto(sae_path, device=DEVICE)
        sae_name = f"{sae_path.stem}_{layer_sig}"
        
        concept_dict = None
        if args.concept_dicts_dir:
            concept_dict_path = Path(args.concept_dicts_dir) / sae_path.stem / "concepts.json"
            if concept_dict_path.exists():
                concept_dict = ConceptDictionary(n_size=sae.context.n_latents)
                concept_dict.set_directory(concept_dict_path.parent)
                try:
                    concept_dict.load()
                    logger.info(f"   Loaded concept dictionary with {len(concept_dict.concepts_map)} concepts")
                except Exception as e:
                    logger.warning(f"   Could not load concept dictionary: {e}")
        
        sentence_results = []
        
        for sentence in test_sentences:
            logger.info(f"   Processing: {sentence[:50]}...")
            
            feature_activations = run_inference_with_sae(lm, sae, layer_sig, sentence, top_k=args.top_k)
            annotated = annotate_sentence_with_features(sentence, feature_activations, concept_dict)
            sentence_results.append(annotated)
        
        all_results[sae_name] = sentence_results
    
    output_file = output_dir / "sample_inferences.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Inference complete! Results saved to {output_file}")


if __name__ == "__main__":
    main()
