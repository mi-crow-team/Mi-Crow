#!/usr/bin/env python3
"""
SLURM script to run sample inferences with SAEs and extract feature activations.

This script:
- Runs inference on test sentences
- Extracts top-k activating features per token
- Maps feature indices to concept names from dictionaries
- Generates structured output
- Optional steering test: baseline vs steered generation (amplify concept) for a few examples

Usage:
    python 08_run_sample_inferences.py --sae_paths path1.pt --layer_signatures layer_sig ...
    python 08_run_sample_inferences.py ... --steering_test  # add baseline vs steered generation
"""

import argparse
import json
import re
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

STEERING_TEMPERATURE = 0.7
STEERING_AMPLIFICATION = 2.0


def map_layer_to_concept_dict(layer_sig: str, base_dir: Path) -> Optional[Path]:
    """Resolve concept dict path from layer signature (e.g. bielik-1.5b/layer_15)."""
    m = re.search(r"layers_(\d+)", layer_sig)
    if not m:
        return None
    layer_num = int(m.group(1))
    if layer_num in (15, 20):
        model_size = "1.5b"
    elif layer_num in (28, 38):
        model_size = "4.5b"
    else:
        return None
    p = base_dir / f"bielik-{model_size}" / f"layer_{layer_num}" / "concepts.json"
    return p if p.exists() else None


def reset_manipulation(sae: Sae) -> None:
    """Reset SAE concept manipulation to baseline."""
    sae.concepts.multiplication.data = torch.ones_like(sae.concepts.multiplication.data)
    sae.concepts.bias.data = torch.zeros_like(sae.concepts.bias.data)


def apply_concept_manipulation(sae: Sae, concept_indices: List[int], factor: float) -> None:
    """Amplify specific concept neurons by factor."""
    reset_manipulation(sae)
    for idx in concept_indices:
        if 0 <= idx < sae.context.n_latents:
            sae.concepts.multiplication.data[idx] = factor


def generate_text(
    lm: LanguageModel,
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = STEERING_TEMPERATURE,
) -> str:
    """Generate continuation from prompt."""
    dev = next(lm.model.parameters()).device
    encoded = lm.tokenizer([prompt], return_tensors="pt")
    encoded = {k: v.to(dev) for k, v in encoded.items()}
    with torch.no_grad():
        out = lm.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=lm.tokenizer.eos_token_id,
        )
    text = lm.tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip()


def concept_indices_for_name(concept_dict: ConceptDictionary, name: str) -> List[int]:
    """Return feature indices whose concept name matches (exact or contains)."""
    out = []
    name_lower = name.lower().replace("-", "_")
    for idx, c in concept_dict.concepts_map.items():
        n = c.name.lower().replace("-", "_")
        if name_lower == n or name_lower in n or n in name_lower:
            out.append(idx)
    return out


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
                    try:
                        sae = L1Sae.load(sae_path)
                    except (AssertionError, RuntimeError) as e:
                        if "cuda" in str(e).lower():
                            logger.warning("SAE saved with CUDA but CUDA unavailable; loading to CPU.")
                            payload = torch.load(sae_path, map_location="cpu", weights_only=False)
                            meta = payload.get("mi_crow_metadata", {})
                            n_latents = int(meta["n_latents"])
                            n_inputs = int(meta["n_inputs"])
                            sae = L1Sae(n_latents=n_latents, n_inputs=n_inputs, device="cpu")
                            if "sae_state_dict" in payload:
                                sae.sae_engine.load_state_dict(payload["sae_state_dict"])
                            elif "model" in payload:
                                sae.sae_engine.load_state_dict(payload["model"])
                            cs = meta.get("concepts_state", {})
                            if "multiplication" in cs:
                                sae.concepts.multiplication.data = cs["multiplication"].clone()
                            if "bias" in cs:
                                sae.concepts.bias.data = cs["bias"].clone()
                        else:
                            raise
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
        try:
            sae = L1Sae.load(sae_path)
        except (AssertionError, RuntimeError) as e:
            if "cuda" in str(e).lower():
                logger.warning("SAE saved with CUDA but CUDA unavailable; loading to CPU.")
                payload = torch.load(sae_path, map_location="cpu", weights_only=False)
                meta = payload.get("mi_crow_metadata", {})
                n_latents = int(meta["n_latents"])
                n_inputs = int(meta["n_inputs"])
                sae = L1Sae(n_latents=n_latents, n_inputs=n_inputs, device="cpu")
                if "sae_state_dict" in payload:
                    sae.sae_engine.load_state_dict(payload["sae_state_dict"])
                elif "model" in payload:
                    sae.sae_engine.load_state_dict(payload["model"])
                cs = meta.get("concepts_state", {})
                if "multiplication" in cs:
                    sae.concepts.multiplication.data = cs["multiplication"].clone()
                if "bias" in cs:
                    sae.concepts.bias.data = cs["bias"].clone()
            else:
                raise
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
    device = next(lm.model.parameters()).device
    enc = lm.tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in enc.items()}
    input_ids = inputs["input_ids"][0]
    tokens = lm.tokenizer.convert_ids_to_tokens(input_ids.tolist())
    
    if hasattr(sae, "sae_engine"):
        sae.sae_engine.eval()
    elif hasattr(sae, "eval"):
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
            if not concept_dict:
                continue
            concept = concept_dict.get(feature_data["feature_idx"])
            if not concept:
                continue
            feature_info = {
                "feature_idx": feature_data["feature_idx"],
                "activation": feature_data["activation"],
                "abs_activation": feature_data["abs_activation"],
                "concept_name": concept.name,
                "concept_score": concept.score,
            }
            token_info["features"].append(feature_info)
        
        if token_info["features"]:
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
    parser.add_argument("--steering_test", action="store_true", help="Run steering test: baseline vs steered generation for a few examples")
    parser.add_argument("--steering_max_tokens", type=int, default=30, help="Max new tokens for steering generations")
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
    first_sae, first_layer_sig, first_concept_dict = None, None, None

    for sae_path_str, layer_sig in zip(args.sae_paths, args.layer_signatures):
        sae_path = Path(sae_path_str)
        logger.info(f"📥 Loading SAE from {sae_path} for layer {layer_sig}...")
        
        sae = load_sae_auto(sae_path, device=DEVICE)
        sae_name = f"{sae_path.stem}_{layer_sig}"
        
        concept_dict = None
        if args.concept_dicts_dir:
            base = Path(args.concept_dicts_dir)
            concept_dict_path = map_layer_to_concept_dict(layer_sig, base)
            if not concept_dict_path or not concept_dict_path.exists():
                concept_dict_path = base / sae_path.stem / "concepts.json"
            if concept_dict_path.exists():
                concept_dict = ConceptDictionary(n_size=sae.context.n_latents)
                concept_dict.set_directory(concept_dict_path.parent)
                try:
                    concept_dict.load()
                    logger.info(f"   Loaded concept dictionary with {len(concept_dict.concepts_map)} concepts")
                except Exception as e:
                    logger.warning(f"   Could not load concept dictionary: {e}")
        
        if first_sae is None and concept_dict is not None:
            first_sae, first_layer_sig, first_concept_dict = sae, layer_sig, concept_dict
        
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

    if args.steering_test and first_sae is not None and first_concept_dict is not None:
        logger.info("🎛️ Running steering test...")
        _run_steering_test(
            lm=lm,
            sae=first_sae,
            layer_sig=first_layer_sig,
            concept_dict=first_concept_dict,
            output_dir=output_dir,
            max_new_tokens=args.steering_max_tokens,
        )


def _run_steering_test(
    lm: LanguageModel,
    sae: Sae,
    layer_sig: str,
    concept_dict: Optional[ConceptDictionary],
    output_dir: Path,
    max_new_tokens: int = 30,
) -> None:
    """
    Run steering examples: same prompt, baseline vs steered (amplify one concept).
    Expectation: baseline output is generic; steered output is more about the concept.
    """
    if not concept_dict or not concept_dict.concepts_map:
        logger.warning("⏭️ Steering test skipped: no concept dictionary")
        return

    steering_examples = [
        ("Kontynuuj: ", "pokoje_hotelowe"),
        ("Opowiedz krótko: ", "lekarz"),
        ("Napisz zdanie: ", "wyrażenie_sprawdzające"),
    ]
    results = []
    hook_id = None

    try:
        hook_id = lm.layers.register_hook(layer_sig, sae)
        sae.context.lm = lm
        sae.context.lm_layer_signature = layer_sig
    except Exception as e:
        logger.warning(f"⏭️ Steering test skipped: could not register SAE hook: {e}")
        return

    try:
        for prompt, concept_name in steering_examples:
            indices = concept_indices_for_name(concept_dict, concept_name)
            if not indices:
                logger.info(f"   Steering skip: no concept '{concept_name}' in dictionary")
                continue

            reset_manipulation(sae)
            baseline = generate_text(lm, prompt, max_new_tokens=max_new_tokens)
            apply_concept_manipulation(sae, indices, STEERING_AMPLIFICATION)
            steered = generate_text(lm, prompt, max_new_tokens=max_new_tokens)
            reset_manipulation(sae)

            results.append({
                "prompt": prompt,
                "concept_name": concept_name,
                "concept_indices": indices,
                "baseline_output": baseline,
                "steered_output": steered,
            })
            logger.info(f"   Steering: prompt='{prompt[:30]}...' concept={concept_name}")
            logger.info(f"      Baseline:  {baseline[:80]}...")
            logger.info(f"      Steered:   {steered[:80]}...")
    finally:
        if hook_id is not None:
            lm.layers.unregister_hook(hook_id)

    out_path = output_dir / "steering_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Steering test complete! Results saved to {out_path}")


if __name__ == "__main__":
    main()
