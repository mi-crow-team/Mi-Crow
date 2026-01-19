#!/usr/bin/env python3
"""
Concept Manipulation Experiments Script

This script demonstrates:
1. SAE comparisons across different Bielik models and layers
2. Concept detection and activation tracking
3. Concept manipulation effects on text generation

Usage:
    python 04_concept_manipulation_experiments.py
    python 04_concept_manipulation_experiments.py --test
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config import PipelineConfig
from mi_crow.language_model.language_model import LanguageModel
from mi_crow.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary
from mi_crow.mechanistic.sae.sae import Sae
from mi_crow.store.local_store import LocalStore
from mi_crow.utils import get_logger

try:
    from server.utils import SAERegistry
except ImportError:
    SAERegistry = None

from mi_crow.mechanistic.sae.modules.topk_sae import TopKSae
from mi_crow.mechanistic.sae.modules.l1_sae import L1Sae

logger = get_logger(__name__)

project_root = Path("/Users/adam/Projects/Inzynierka/codebase/experiments/slurm_sae_pipeline/")
CONFIG_FILE = project_root / "configs" / "config_bielik12_polemo2.json"
RESULTS_DIR = project_root / "results" / "concept_experiments"
CONCEPT_DICTS_DIR = project_root  / "dictionaries"

MAX_NEW_TOKENS = 50
AMPLIFICATION_FACTORS = [1.0, 1.5, 2.0, 2.5]
TEMPERATURE = 0.7

PROMPTS_FILE = project_root / "prompts.json"


def load_prompts(prompts_file: Path) -> Tuple[Dict[str, List[str]], List[str]]:
    """Load prompts from JSON file.
    
    Returns:
        Tuple of (categorized_prompts_dict, flat_list_of_all_prompts)
    """
    if not prompts_file.exists():
        logger.warning(f"Prompts file not found: {prompts_file}, using defaults")
        default = {"generic": ["Kontynuuj:", "Opowiedz o", "Wyjaśnij"]}
        all_default = default["generic"]
        return default, all_default
    
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
    
    all_prompts = []
    for category, prompts in prompts_data.items():
        all_prompts.extend(prompts)
    
    logger.info(f"Loaded {len(all_prompts)} prompts from {len(prompts_data)} categories")
    return prompts_data, all_prompts


def load_sae_auto(sae_path: Path, device: str = "cpu") -> Sae:
    """Load SAE and auto-detect class from metadata.json or model file.
    
    Args:
        sae_path: Path to SAE model file
        device: Device to load model to (default: "cpu")
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
                        if hasattr(sae.concepts, 'multiplication'):
                            sae.concepts.multiplication.data = sae.concepts.multiplication.data.to(target_device)
                        if hasattr(sae.concepts, 'bias'):
                            sae.concepts.bias.data = sae.concepts.bias.data.to(target_device)
                        return sae
                    except Exception as e:
                        logger.warning(f"Could not use SAERegistry for '{sae_class}': {e}, trying direct load")
                
                if sae_class == "TopKSae":
                    sae = TopKSae.load(sae_path)
                    sae.sae_engine.to(target_device)
                    sae.context.device = device
                    if hasattr(sae.concepts, 'multiplication'):
                        sae.concepts.multiplication.data = sae.concepts.multiplication.data.to(target_device)
                    if hasattr(sae.concepts, 'bias'):
                        sae.concepts.bias.data = sae.concepts.bias.data.to(target_device)
                    return sae
                elif sae_class == "L1Sae":
                    try:
                        sae = L1Sae.load(sae_path)
                    except (AssertionError, RuntimeError) as e:
                        if "CUDA" in str(e) or "cuda" in str(e).lower():
                            logger.warning(f"SAE was saved with CUDA but CUDA not available. Loading to CPU...")
                            p = Path(sae_path)
                            payload = torch.load(p, map_location='cpu')
                            mi_crow_meta = payload["mi_crow_metadata"]
                            n_latents = int(mi_crow_meta["n_latents"])
                            n_inputs = int(mi_crow_meta["n_inputs"])
                            sae = L1Sae(n_latents=n_latents, n_inputs=n_inputs, device="cpu")
                            if "sae_state_dict" in payload:
                                sae.sae_engine.load_state_dict(payload["sae_state_dict"])
                            elif "model" in payload:
                                sae.sae_engine.load_state_dict(payload["model"])
                            concepts_state = mi_crow_meta.get("concepts_state", {})
                            if concepts_state:
                                if "multiplication" in concepts_state:
                                    sae.concepts.multiplication.data = concepts_state["multiplication"]
                                if "bias" in concepts_state:
                                    sae.concepts.bias.data = concepts_state["bias"]
                        else:
                            raise
                    sae.sae_engine.to(target_device)
                    sae.context.device = device
                    if hasattr(sae.concepts, 'multiplication'):
                        sae.concepts.multiplication.data = sae.concepts.multiplication.data.to(target_device)
                    if hasattr(sae.concepts, 'bias'):
                        sae.concepts.bias.data = sae.concepts.bias.data.to(target_device)
                    return sae
        except Exception as e:
            logger.warning(f"Could not read metadata.json: {e}, trying auto-detection")
    
    try:
        sae = TopKSae.load(sae_path)
        sae.sae_engine.to(target_device)
        sae.context.device = device
        if hasattr(sae.concepts, 'multiplication'):
            sae.concepts.multiplication.data = sae.concepts.multiplication.data.to(target_device)
        if hasattr(sae.concepts, 'bias'):
            sae.concepts.bias.data = sae.concepts.bias.data.to(target_device)
        return sae
    except (ValueError, KeyError) as e:
        try:
            try:
                sae = L1Sae.load(sae_path)
            except (AssertionError, RuntimeError) as cuda_err:
                if "CUDA" in str(cuda_err) or "cuda" in str(cuda_err).lower():
                    logger.warning(f"SAE was saved with CUDA but CUDA not available. Loading to CPU...")
                    p = Path(sae_path)
                    payload = torch.load(p, map_location='cpu')
                    mi_crow_meta = payload["mi_crow_metadata"]
                    n_latents = int(mi_crow_meta["n_latents"])
                    n_inputs = int(mi_crow_meta["n_inputs"])
                    sae = L1Sae(n_latents=n_latents, n_inputs=n_inputs, device="cpu")
                    if "sae_state_dict" in payload:
                        sae.sae_engine.load_state_dict(payload["sae_state_dict"])
                    elif "model" in payload:
                        sae.sae_engine.load_state_dict(payload["model"])
                    concepts_state = mi_crow_meta.get("concepts_state", {})
                    if concepts_state:
                        if "multiplication" in concepts_state:
                            sae.concepts.multiplication.data = concepts_state["multiplication"]
                        if "bias" in concepts_state:
                            sae.concepts.bias.data = concepts_state["bias"]
                else:
                    raise
            sae.sae_engine.to(target_device)
            sae.context.device = device
            if hasattr(sae.concepts, 'multiplication'):
                sae.concepts.multiplication.data = sae.concepts.multiplication.data.to(target_device)
            if hasattr(sae.concepts, 'bias'):
                sae.concepts.bias.data = sae.concepts.bias.data.to(target_device)
            return sae
        except Exception as e2:
            raise ValueError(f"Could not load SAE from {sae_path}. Tried TopKSae and L1Sae. TopKSae error: {e}, L1Sae error: {e2}") from e2


def find_sae_paths(store_dir: Path, layer_signatures: List[str], test_mode: bool = False) -> Dict[str, Path]:
    """
    Find SAE model paths from store directory.
    
    Looks for SAE models in store/runs/ subdirectories matching layer signatures.
    In test mode, also checks local results directory.
    """
    sae_paths = {}
    
    if test_mode:
        local_results_dir = script_dir / "results" / "runs"
        if local_results_dir.exists():
            logger.info(f"🧪 TEST MODE: Checking local results directory: {local_results_dir}")
            for run_dir in sorted(local_results_dir.iterdir(), reverse=True):
                if not run_dir.is_dir():
                    continue
                
                sae_file = next(run_dir.glob("*.pt"), None)
                if not sae_file:
                    sae_file = next(run_dir.glob("model.pt"), None)
                
                if sae_file and sae_file.exists():
                    metadata_path = run_dir / "metadata.json"
                    meta_path = run_dir / "meta.json"
                    
                    layer = None
                    if metadata_path.exists():
                        try:
                            meta = json.loads(metadata_path.read_text())
                            layer = meta.get("layer") or meta.get("layer_signature")
                        except Exception as e:
                            logger.debug(f"Error reading metadata from {metadata_path}: {e}")
                    elif meta_path.exists():
                        try:
                            meta = json.loads(meta_path.read_text())
                            layer = meta.get("layer") or meta.get("layer_signature")
                        except Exception as e:
                            logger.debug(f"Error reading meta from {meta_path}: {e}")
                    
                    if layer:
                        for layer_sig in layer_signatures:
                            if layer_sig in str(layer) or str(layer) in layer_sig:
                                if layer_sig not in sae_paths:
                                    sae_paths[layer_sig] = sae_file
                                    logger.info(f"🧪 TEST MODE: Found local SAE for {layer_sig}: {sae_file}")
                    else:
                        run_name = run_dir.name
                        for layer_sig in layer_signatures:
                            if layer_sig.replace("_", "").replace("-", "") in run_name.replace("_", "").replace("-", ""):
                                if layer_sig not in sae_paths:
                                    sae_paths[layer_sig] = sae_file
                                    logger.info(f"🧪 TEST MODE: Found local SAE by name match for {layer_sig}: {sae_file}")
                                    break
    
    runs_dir = store_dir / "runs"
    
    if not runs_dir.exists():
        if not test_mode:
            logger.warning(f"Runs directory not found: {runs_dir}")
        return sae_paths
    
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        
        try:
            meta = json.loads(metadata_path.read_text())
            layer = meta.get("layer") or meta.get("layer_signature")
            sae_path = meta.get("sae_path") or next(run_dir.glob("*.pt"), None)
            
            if layer and sae_path:
                sae_path = Path(sae_path) if isinstance(sae_path, str) else sae_path
                if sae_path.exists():
                    for layer_sig in layer_signatures:
                        if layer_sig in str(layer) or str(layer) in layer_sig:
                            if layer_sig not in sae_paths:
                                sae_paths[layer_sig] = sae_path
                                logger.info(f"Found SAE for {layer_sig}: {sae_path}")
        except Exception as e:
            logger.debug(f"Error reading metadata from {metadata_path}: {e}")
            continue
    
    return sae_paths


def map_layer_to_concept_dict(layer_signature: str, test_mode: bool = False) -> Optional[Path]:
    """Map layer signature to concept dictionary path."""
    layer_num = None
    if "layers_15" in layer_signature:
        layer_num = 15
        model_size = "1.5b"
    elif "layers_20" in layer_signature:
        layer_num = 20
        model_size = "1.5b"
    elif "layers_28" in layer_signature:
        layer_num = 28
        model_size = "4.5b"
    elif "layers_38" in layer_signature:
        layer_num = 38
        model_size = "4.5b"
    else:
        return None
    
    concept_path = CONCEPT_DICTS_DIR / f"bielik-{model_size}" / f"layer_{layer_num}" / "concepts.json"
    
    if concept_path.exists():
        logger.info(f"Found concept dictionary: {concept_path}")
    else:
        logger.warning(f"Concept dictionary not found: {concept_path}")
    
    return concept_path if concept_path.exists() else None


def reset_manipulation(sae: Sae):
    """Reset SAE concept manipulation to baseline."""
    sae.concepts.multiplication.data = torch.ones_like(sae.concepts.multiplication.data)
    sae.concepts.bias.data = torch.zeros_like(sae.concepts.bias.data)


def apply_concept_manipulation(sae: Sae, concept_indices: List[int], factor: float):
    """Apply multiplication factor to specific concept neurons."""
    reset_manipulation(sae)
    for idx in concept_indices:
        if 0 <= idx < sae.context.n_latents:
            sae.concepts.multiplication.data[idx] = factor


def generate_text(lm: LanguageModel, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Generate text from prompt using language model."""
    encoded = lm.tokenizer([prompt], return_tensors="pt")
    encoded = {k: v.to(lm.model.device) for k, v in encoded.items()}
    
    with torch.no_grad():
        outputs = lm.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=lm.tokenizer.eos_token_id
        )
    
    generated = lm.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated[len(prompt):].strip()


def get_top_activating_concepts(
    sae: Sae,
    concept_dict: ConceptDictionary,
    top_k: int = 10
) -> List[Dict]:
    """Get top activating concepts from SAE metadata."""
    top_concepts = []
    
    if 'batch_items' not in sae.metadata or not sae.metadata['batch_items']:
        return top_concepts
    
    batch_items = sae.metadata['batch_items']
    if not batch_items:
        return top_concepts
    
    item_metadata = batch_items[0]
    activations = item_metadata.get('activations', {})
    
    if not activations:
        return top_concepts
    
    activation_items = [(idx, val) for idx, val in activations.items()]
    activation_items.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for neuron_idx, activation_value in activation_items[:top_k]:
        concept = concept_dict.get(neuron_idx)
        concept_info = {
            "neuron_idx": neuron_idx,
            "activation": activation_value,
            "concept_name": concept.name if concept else None,
            "concept_score": concept.score if concept else None
        }
        top_concepts.append(concept_info)
    
    return top_concepts


def get_top_activating_concepts(
    sae: Sae,
    concept_dict: ConceptDictionary,
    top_k: int = 10
) -> List[Dict]:
    """Get top activating concepts from SAE metadata."""
    top_concepts = []
    
    if 'batch_items' not in sae.metadata or not sae.metadata['batch_items']:
        return top_concepts
    
    batch_items = sae.metadata['batch_items']
    if not batch_items:
        return top_concepts
    
    item_metadata = batch_items[0]
    activations = item_metadata.get('activations', {})
    
    if not activations:
        return top_concepts
    
    activation_items = [(idx, val) for idx, val in activations.items()]
    activation_items.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for neuron_idx, activation_value in activation_items[:top_k]:
        concept = concept_dict.get(neuron_idx)
        concept_info = {
            "neuron_idx": neuron_idx,
            "activation": activation_value,
            "concept_name": concept.name if concept else None,
            "concept_score": concept.score if concept else None
        }
        top_concepts.append(concept_info)
    
    return top_concepts


def run_prompt_experiment(
    lm: LanguageModel,
    sae: Sae,
    layer_signature: str,
    concept_dict: ConceptDictionary,
    prompt: str,
    concept_indices: List[int],
    category: str
) -> Dict:
    """Run experiment for a single prompt with baseline and manipulated outputs."""
    hook_id = lm.layers.register_hook(layer_signature, sae)
    sae.context.lm = lm
    sae.context.lm_layer_signature = layer_signature
    
    result = {
        "prompt": prompt,
        "layer": layer_signature,
        "category": category,
        "concepts_manipulated": {str(idx): concept_dict.get(idx).name if concept_dict.get(idx) else "unknown" 
                                for idx in concept_indices},
        "baseline": {},
        "manipulated": []
    }
    
    reset_manipulation(sae)
    if 'batch_items' in sae.metadata:
        sae.metadata['batch_items'] = []
    if 'activations' in sae.tensor_metadata:
        sae.tensor_metadata['activations'] = None
    
    baseline_output = generate_text(lm, prompt, max_new_tokens=MAX_NEW_TOKENS)
    top_concepts_baseline = get_top_activating_concepts(sae, concept_dict, top_k=10)
    
    result["baseline"] = {
        "output": baseline_output,
        "top_activating_concepts": top_concepts_baseline
    }
    
    for factor in AMPLIFICATION_FACTORS[1:]:
        apply_concept_manipulation(sae, concept_indices, factor)
        if 'batch_items' in sae.metadata:
            sae.metadata['batch_items'] = []
        if 'activations' in sae.tensor_metadata:
            sae.tensor_metadata['activations'] = None
        
        manipulated_output = generate_text(lm, prompt, max_new_tokens=MAX_NEW_TOKENS)
        top_concepts_manipulated = get_top_activating_concepts(sae, concept_dict, top_k=10)
        
        result["manipulated"].append({
            "amplification_factor": factor,
            "output": manipulated_output,
            "top_activating_concepts": top_concepts_manipulated
        })
    
    reset_manipulation(sae)
    lm.layers.unregister_hook(hook_id)
    return result




def categorize_concepts(concept_dict: ConceptDictionary) -> Dict[str, List[int]]:
    """Categorize concepts by their names."""
    categories = {
        "emotion": [],
        "action": [],
        "grammar": [],
        "other": []
    }
    
    for neuron_idx, concept in concept_dict.concepts_map.items():
        name_lower = concept.name.lower()
        if any(word in name_lower for word in ["emocj", "radość", "smutek", "gniew", "strach"]):
            categories["emotion"].append(neuron_idx)
        elif any(word in name_lower for word in ["akcj", "czasownik", "robi", "wykonuj"]):
            categories["action"].append(neuron_idx)
        elif any(word in name_lower for word in ["rzeczownik", "przymiotnik", "gramatycz", "zdani"]):
            categories["grammar"].append(neuron_idx)
        else:
            categories["other"].append(neuron_idx)
    
    return categories


def compare_saes(saes: Dict[str, Tuple[Sae, ConceptDictionary]]) -> Dict:
    """Compare SAEs and their concept dictionaries."""
    comparison = {
        "sae_stats": {},
        "concept_overlap": {},
        "total_concepts": {}
    }
    
    for layer_sig, (sae, concept_dict) in saes.items():
        comparison["sae_stats"][layer_sig] = {
            "type": type(sae).__name__,
            "n_inputs": sae.context.n_inputs,
            "n_latents": sae.context.n_latents,
            "n_concepts": len(concept_dict.concepts_map)
        }
        comparison["total_concepts"][layer_sig] = len(concept_dict.concepts_map)
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Concept Manipulation Experiments")
    parser.add_argument("--test", action="store_true", help="Run in test mode: only Bielik 1.5B, 1 example")
    parser.add_argument("--sae-paths", type=str, nargs="+", default=None, help="Explicit SAE model paths (one per layer, in config order)")
    args = parser.parse_args()
    
    TEST_MODE = args.test
    EXPLICIT_SAE_PATHS = args.sae_paths
    
    print("🚀 Starting Concept Manipulation Experiments" + (" (TEST MODE)" if TEST_MODE else ""))
    logger.info("🚀 Starting Concept Manipulation Experiments" + (" (TEST MODE)" if TEST_MODE else ""))
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("📝 Loading prompts...")
    prompts_by_category, all_prompts = load_prompts(PROMPTS_FILE)
    
    if TEST_MODE:
        TEST_PROMPTS = all_prompts[:1]
        logger.info(f"🧪 TEST MODE: Using only 1 prompt: {TEST_PROMPTS[0]}")
    else:
        TEST_PROMPTS = all_prompts
    
    cfg = PipelineConfig.from_json_file(CONFIG_FILE)
    MODEL_ID = cfg.model.model_id
    LAYER_SIGNATURES = cfg.layer.layer_signature
    if isinstance(LAYER_SIGNATURES, str):
        LAYER_SIGNATURES = [LAYER_SIGNATURES]
    
    if TEST_MODE:
        original_layer_sigs = LAYER_SIGNATURES.copy() if isinstance(LAYER_SIGNATURES, list) else [LAYER_SIGNATURES]
        LAYER_SIGNATURES = [sig for sig in original_layer_sigs if "layers_15" in str(sig) or "layers_20" in str(sig)]
        if not LAYER_SIGNATURES:
            LAYER_SIGNATURES = [original_layer_sigs[0]] if original_layer_sigs else []
        else:
            LAYER_SIGNATURES = LAYER_SIGNATURES[:1]
        logger.info(f"🧪 TEST MODE: Using only Bielik 1.5B layer: {LAYER_SIGNATURES}")
        
        STORE_DIR = project_root / "store"
        DEVICE = "cpu"
        logger.info(f"🧪 TEST MODE: Using local store directory: {STORE_DIR}")
        logger.info(f"🧪 TEST MODE: Using CPU device")
    else:
        STORE_DIR = Path(cfg.storage.store_dir or str(project_root / "store"))
        DEVICE = cfg.storage.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"📱 Device: {DEVICE}")
    logger.info(f"🔧 Model: {MODEL_ID}")
    logger.info(f"🎯 Layers: {LAYER_SIGNATURES}")
    logger.info(f"📁 Store: {STORE_DIR}")
    
    logger.info("📥 Loading SAE models...")
    
    if EXPLICIT_SAE_PATHS:
        if len(EXPLICIT_SAE_PATHS) != len(LAYER_SIGNATURES):
            logger.error(f"❌ Number of SAE paths ({len(EXPLICIT_SAE_PATHS)}) must match number of layers ({len(LAYER_SIGNATURES)})")
            return
        
        sae_paths = {}
        for layer_sig, sae_path_str in zip(LAYER_SIGNATURES, EXPLICIT_SAE_PATHS):
            sae_path = Path(sae_path_str)
            if not sae_path.exists():
                logger.error(f"❌ SAE path does not exist: {sae_path}")
                return
            sae_paths[layer_sig] = sae_path
            logger.info(f"✅ Using explicit SAE path for {layer_sig}: {sae_path}")
    else:
        logger.info("📥 Finding SAE models...")
        sae_paths = find_sae_paths(STORE_DIR, LAYER_SIGNATURES, test_mode=TEST_MODE)
        
        if not sae_paths:
            logger.error("❌ No SAE models found. Please train SAEs first or use --sae-paths.")
            return
        
        logger.info(f"✅ Found {len(sae_paths)} SAE models")
    
    logger.info("📥 Loading language model...")
    store = LocalStore(base_path=STORE_DIR)
    lm = LanguageModel.from_huggingface(MODEL_ID, store=store, device=DEVICE)
    lm.model.eval()
    logger.info(f"✅ Model loaded: {lm.model_id}")
    
    logger.info("📥 Loading SAEs and concept dictionaries...")
    saes = {}
    for layer_sig in LAYER_SIGNATURES:
        if layer_sig not in sae_paths:
            logger.warning(f"⚠️  No SAE found for {layer_sig}, skipping")
            continue
        
        sae_path = sae_paths[layer_sig]
        sae = load_sae_auto(sae_path, device=DEVICE)
        
        concept_dict_path = map_layer_to_concept_dict(layer_sig, test_mode=TEST_MODE)
        if not concept_dict_path:
            logger.warning(f"⚠️  No concept dictionary found for {layer_sig}, skipping")
            continue
        
        concept_dict = ConceptDictionary.from_json(concept_dict_path, n_size=sae.context.n_latents)
        sae.concepts.dictionary = concept_dict
        
        saes[layer_sig] = (sae, concept_dict)
        logger.info(f"✅ Loaded {layer_sig}: {type(sae).__name__} with {len(concept_dict.concepts_map)} concepts")
    
    if not saes:
        logger.error("❌ No SAEs with concept dictionaries loaded. Exiting.")
        return
    
    logger.info("📊 Comparing SAEs...")
    comparison = compare_saes(saes)
    comparison_path = RESULTS_DIR / "comparison_report.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Comparison report saved: {comparison_path}")
    
    logger.info("🎛️  Running experiments per prompt...")
    all_prompt_results = []
    
    for layer_sig, (sae, concept_dict) in saes.items():
        categories = categorize_concepts(concept_dict)
        
        if TEST_MODE:
            categories = {k: v for k, v in list(categories.items())[:1] if v}
            logger.info(f"🧪 TEST MODE: Testing only {list(categories.keys())[0]} category")
        
        for category, indices in categories.items():
            if not indices:
                continue
            
            selected_indices = indices[:1] if TEST_MODE else indices[:3]
            if TEST_MODE:
                category_prompts = TEST_PROMPTS[:1]
            else:
                category_prompts = prompts_by_category.get(category, [])[:5] if category in prompts_by_category else TEST_PROMPTS[:5]
            
            for prompt in category_prompts:
                logger.info(f"Running experiment for prompt: {prompt[:50]}...")
                result = run_prompt_experiment(
                    lm, sae, layer_sig, concept_dict, prompt,
                    selected_indices, category
                )
                all_prompt_results.append(result)
            
            if TEST_MODE:
                break
    
    results_path = RESULTS_DIR / "prompt_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_prompt_results, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Results saved: {results_path} ({len(all_prompt_results)} prompts)")
    
    logger.info("📝 Generating summary report...")
    summary = f"""# Concept Manipulation Experiments Summary

Generated: {datetime.now().isoformat()}

## SAE Comparison

"""
    for layer_sig, stats in comparison["sae_stats"].items():
        summary += f"### {layer_sig}\n"
        summary += f"- Type: {stats['type']}\n"
        summary += f"- Dimensions: {stats['n_inputs']} → {stats['n_latents']}\n"
        summary += f"- Concepts: {stats['n_concepts']}\n\n"
    
    summary += "## Experiments\n\n"
    summary += f"Tested {len(all_prompt_results)} prompts across {len(saes)} SAE(s).\n"
    summary += f"Prompt categories: {', '.join(prompts_by_category.keys())}\n"
    summary += f"Amplification factors tested: {AMPLIFICATION_FACTORS}\n\n"
    
    summary += "## Results Structure\n\n"
    summary += "Results are organized per prompt in `prompt_results.json` with:\n"
    summary += "- Baseline output and top 10 activating concepts\n"
    summary += "- Manipulated outputs for each amplification factor (1.5x, 2.0x, 2.5x) with top 10 activating concepts\n"
    summary += "- All results in a single file for easy comparison\n\n"
    
    summary_path = RESULTS_DIR / "summary_report.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info(f"✅ Summary report saved: {summary_path}")
    
    logger.info("✅ All experiments completed!")


if __name__ == "__main__":
    main()
