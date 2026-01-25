#!/usr/bin/env python3
"""
Interactive naming + inference pipeline (build concept results from scratch).

No concept dict. Flow:
1. Prompts (sentences) from polemo2 test split.
2. Run inference on a sentence → gather top-N activating features (aggregate across tokens, skip most active).
3. Name each feature via top texts (Ollama); use cache on repeat. Skip sentence if <5 named.
4. Ask Ollama: "Do these concepts make sense for the sentence?" (TAK/NIE).
5. If TAK: perform concept manipulation (amplify x2) → generate baseline and steered outputs.
6. Ask Ollama: "Is steered output more biased towards concepts?" (TAK/NIE).
7. If bias check TAK: save to results. If NIE: omit. Iterate until target_results saved.

Usage:
    python 09_interactive_naming_inference.py \\
        --sae_paths path.pt --layer_signatures layer_sig \\
        --config config.json \\
        (--top_texts_dir path/to/top_texts | --top_texts_file path/to/top_texts_layer_0_..._batch_N.json) \\
        [--n_features_per_prompt 12] [--skip_most_active 2] [--target_results 10] [--cache_path ...] \\
        [--ollama_model SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M] [--validation_retries 2]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import requests
import torch

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# Add script dir for config import
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from config import PipelineConfig
from mi_crow.language_model.language_model import LanguageModel
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

_POLISH_NAMING_TEMPLATE = """Jesteś ekspertem w analizie języków naturalnych, specjalizującym się w interpretowalności sieci neuronowych dla języka polskiego.

Zadanie: Przeanalizuj podane przykłady tekstów i zidentyfikuj pojedynczy, precyzyjny koncept semantyczny lub językowy, który aktywuje dany neuron w modelu językowym Bielik.

Instrukcje:
1. W każdym fragmencie tekstu najważniejszy token jest oznaczony jako <token>.
2. Przeanalizuj wszystkie przykłady i znajdź wspólny wzorzec (semantyka, gramatyka, morfologia).
3. Zwróć nazwę w formacie snake_case.

Format odpowiedzi (TYLKO JSON, bez dodatkowego tekstu):
{{"nazwa_konceptu": "nazwa_snake_case", "opis": "Krótki opis.", "kategoria": "Semantyka|Gramatyka i Morfologia|Składnia|Kontekst", "pewność": 0.0-1.0}}

Przykłady tekstów:
{examples}

Zwróć TYLKO poprawny JSON."""

_VALIDATION_TEMPLATE_5 = """Zdanie: "{sentence}"

Koncepty: {concepts_str}

Czy te koncepty oddają treść zdania? Odpowiedz tylko TAK lub NIE."""

_BIAS_CHECK_TEMPLATE = """Prompt: "{prompt}"

Koncepty: {concepts_str}

Odpowiedź bazowa (bez manipulacji): "{baseline}"

Odpowiedź z manipulacją (wzmocnienie konceptów x2): "{steered}"

Czy odpowiedź z manipulacją zawiera więcej treści związanej z tymi konceptami niż odpowiedź bazowa? 
Odpowiedz tylko TAK lub NIE."""

_STEERING_PRIORITY_TEMPLATE = """Zdanie: "{sentence}"

Koncepty:
{concepts_list}

Który z tych konceptów jest najbardziej podatny na sterowanie (manipulację) w kontekście tego zdania? 
Odpowiedz tylko nazwą konceptu (bez dodatkowych wyjaśnień)."""


def _load_sae_auto(sae_path: Path, device: str = "cpu") -> Sae:
    """Load SAE (TopK or L1), with CPU fallback when CUDA unavailable."""
    sae_path = Path(sae_path)
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE not found: {sae_path}")
    meta_path = sae_path.parent / "metadata.json"
    target = torch.device(device)

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            cls_name = meta.get("sae_class") or meta.get("sae_type")
            if cls_name == "TopKSae":
                sae = TopKSae.load(sae_path)
                sae.sae_engine.to(target)
                sae.context.device = device
                return sae
            if cls_name == "L1Sae":
                try:
                    sae = L1Sae.load(sae_path)
                except (AssertionError, RuntimeError) as e:
                    if "cuda" in str(e).lower():
                        logger.warning("SAE saved with CUDA; loading to CPU.")
                        payload = torch.load(sae_path, map_location="cpu", weights_only=False)
                        m = payload.get("mi_crow_metadata", {})
                        sae = L1Sae(n_latents=int(m["n_latents"]), n_inputs=int(m["n_inputs"]), device="cpu")
                        if "sae_state_dict" in payload:
                            sae.sae_engine.load_state_dict(payload["sae_state_dict"])
                        elif "model" in payload:
                            sae.sae_engine.load_state_dict(payload["model"])
                        cs = m.get("concepts_state", {})
                        if "multiplication" in cs:
                            sae.concepts.multiplication.data = cs["multiplication"].clone()
                        if "bias" in cs:
                            sae.concepts.bias.data = cs["bias"].clone()
                    else:
                        raise
                sae.sae_engine.to(target)
                sae.context.device = device
                return sae
        except Exception as e:
            logger.warning(f"metadata load failed: {e}")

    try:
        sae = TopKSae.load(sae_path)
    except Exception:
        try:
            sae = L1Sae.load(sae_path)
        except (AssertionError, RuntimeError) as e:
            if "cuda" in str(e).lower():
                logger.warning("SAE saved with CUDA; loading to CPU.")
                payload = torch.load(sae_path, map_location="cpu", weights_only=False)
                m = payload.get("mi_crow_metadata", {})
                sae = L1Sae(n_latents=int(m["n_latents"]), n_inputs=int(m["n_inputs"]), device="cpu")
                if "sae_state_dict" in payload:
                    sae.sae_engine.load_state_dict(payload["sae_state_dict"])
                elif "model" in payload:
                    sae.sae_engine.load_state_dict(payload["model"])
                cs = m.get("concepts_state", {})
                if "multiplication" in cs:
                    sae.concepts.multiplication.data = cs["multiplication"].clone()
                if "bias" in cs:
                    sae.concepts.bias.data = cs["bias"].clone()
            else:
                raise
    sae.sae_engine.to(target)
    sae.context.device = device
    return sae


def _run_inference(
    lm: LanguageModel,
    sae: Sae,
    layer_sig: str,
    sentence: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Run inference on sentence, return token-level top-k features."""
    dev = next(lm.model.parameters()).device
    enc = lm.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(dev) for k, v in enc.items()}
    input_ids = inputs["input_ids"][0]
    tokens = lm.tokenizer.convert_ids_to_tokens(input_ids.tolist())

    if hasattr(sae, "sae_engine"):
        sae.sae_engine.eval()
    lm.model.eval()

    captured = None

    def _hook(module, inp, out):
        nonlocal captured
        t = out if isinstance(out, torch.Tensor) else getattr(out, "last_hidden_state", None) or (out[0] if isinstance(out, (tuple, list)) else None)
        if t is not None:
            captured = t.detach()
        return out

    handle = None
    if layer_sig in lm.layers.name_to_layer:
        handle = lm.layers.name_to_layer[layer_sig].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            _ = lm.model(**inputs)
    finally:
        if handle is not None:
            handle.remove()

    if captured is None:
        return []

    out = []
    seq_len = captured.shape[1]
    with torch.no_grad():
        for ti in range(seq_len):
            act = captured[0, ti, :]
            codes = sae.encode(act.unsqueeze(0))
            vals, idxs = torch.topk(torch.abs(codes[0]), k=min(top_k, codes.shape[1]))
            feats = []
            for i in range(len(idxs)):
                fi = int(idxs[i].item())
                feats.append({
                    "feature_idx": fi,
                    "activation": float(codes[0, fi].item()),
                    "abs_activation": float(vals[i].item()),
                })
            out.append({
                "token_idx": ti,
                "token": tokens[ti] if ti < len(tokens) else "",
                "features": feats,
            })
    return out


def _format_texts_for_prompt(texts: List[Dict[str, Any]], max_texts: int = 15) -> str:
    """Format top-text examples for naming prompt."""
    sorted_ = sorted(texts, key=lambda x: x.get("score", 0.0), reverse=True)[:max_texts]
    lines = []
    for i, d in enumerate(sorted_, 1):
        text = d.get("text", "")
        ti = d.get("token_idx", 0)
        words = text.split()
        if 0 <= ti < len(words):
            words[ti] = f"<{words[ti]}>"
        lines.append(f"{i}. {' '.join(words)}")
    return "\n".join(lines)


def _call_ollama(prompt: str, model: str, ollama_url: str) -> str:
    """Call Ollama /api/generate, return raw response text."""
    r = requests.post(
        f"{ollama_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _name_feature_ollama(
    feature_idx: int,
    texts: List[Dict[str, Any]],
    model: str,
    ollama_url: str,
) -> Optional[Dict[str, Any]]:
    """Generate concept name via Ollama. Returns {name, score} or None."""
    formatted = _format_texts_for_prompt(texts, max_texts=15)
    if not formatted.strip():
        return None
    full = _POLISH_NAMING_TEMPLATE.format(examples=formatted)
    raw = _call_ollama(full, model, ollama_url)
    if not raw:
        return None
    m = re.search(r"\{[^{}]*nazwa_konceptu[^{}]*\}", raw, re.DOTALL)
    js = m.group(0) if m else raw
    try:
        data = json.loads(js)
    except json.JSONDecodeError:
        return None
    name = data.get("nazwa_konceptu") or "Nieznany"
    score = float(data.get("pewność", 0.5))
    return {"name": name, "score": score}


def _load_top_texts_latest_batch(top_texts_dir: Path) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """Load top texts, latest batch per layer. Returns layer_sig -> feature_idx -> list of text dicts."""
    aggregated: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    if not top_texts_dir.exists():
        return aggregated
    files = list(top_texts_dir.glob("top_texts_layer_*.json"))
    if not files:
        return aggregated

    by_layer: Dict[str, List[Tuple[int, Path]]] = {}
    for f in files:
        mo = re.search(r"top_texts_layer_\d+_(llamaforcausallm_model_layers_\d+)", f.name)
        if not mo:
            continue
        layer = mo.group(1)
        batch_mo = re.search(r"_batch_(\d+)\.json", f.name)
        batch = int(batch_mo.group(1)) if batch_mo else 0
        by_layer.setdefault(layer, []).append((batch, f))

    for layer, pairs in by_layer.items():
        _, path = max(pairs, key=lambda x: x[0])
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        aggregated[layer] = {}
        for k, v in data.items():
            try:
                idx = int(k)
                if isinstance(v, list):
                    aggregated[layer][idx] = v
            except (ValueError, TypeError):
                pass
    return aggregated


def _load_top_texts_from_file(path: Path) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """
    Load a single top_texts JSON file. Layer is parsed from filename
    (e.g. top_texts_layer_0_llamaforcausallm_model_layers_15_batch_480.json -> layers_15).
    Returns layer_sig -> feature_idx -> list of text dicts.
    """
    out: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    if not path.exists():
        return out
    mo = re.search(r"top_texts_layer_\d+_(llamaforcausallm_model_layers_\d+)", path.name)
    if not mo:
        logger.warning("Could not parse layer from top_texts filename: %s", path.name)
        return out
    layer_sig = mo.group(1)
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    out[layer_sig] = {}
    for k, v in data.items():
        try:
            idx = int(k)
            if isinstance(v, list):
                out[layer_sig][idx] = v
        except (ValueError, TypeError):
            pass
    logger.info("Loaded top texts from %s (%s, %d features)", path, layer_sig, len(out[layer_sig]))
    return out


def _load_polemo2_test_texts(limit: Optional[int] = None, max_retries: int = 5) -> List[str]:
    """
    Load polemo2-official test split. Downloads full dataset locally if not cached.
    Checks cache first (cache/polemo2_test.jsonl), then downloads via datasets library if needed.
    
    Returns list of non-empty text strings. Uses config all_sentence, split test.
    """
    cache_dir = _script_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    jsonl_file = cache_dir / "polemo2_test.jsonl"

    # Try to load from cache first
    if jsonl_file.exists() and jsonl_file.stat().st_size > 0:
        logger.info("Loading polemo2 test from cache: %s", jsonl_file)
        all_texts: List[str] = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    text = row.get("text", "")
                    if text and isinstance(text, str):
                        text = text.strip()
                        if text:
                            all_texts.append(text)
                except json.JSONDecodeError:
                    continue
        logger.info("Loaded %d texts from cache", len(all_texts))
        if limit is not None:
            return all_texts[:limit]
        return all_texts

    # Cache not found, download full dataset using datasets library
    if not HAS_DATASETS:
        logger.error("datasets library not available. Install with: pip install datasets")
        logger.warning("Falling back to API-based fetching (may be slow and rate-limited)")
        # Fallback to old API method would go here, but let's just return empty
        return []

    logger.info("Cache not found, downloading full polemo2 test split from HuggingFace...")
    try:
        dataset = load_dataset("clarin-pl/polemo2-official", "all_sentence", split="test")
        logger.info("Downloaded dataset, processing texts...")
        
        all_texts: List[str] = []
        all_rows: List[Dict[str, str]] = []
        
        for item in dataset:
            text = item.get("text", "")
            if text and isinstance(text, str):
                text = text.strip()
                if text:
                    all_texts.append(text)
                    all_rows.append({"text": text})
        
        # Save to cache
        logger.info("Saving %d texts to cache: %s", len(all_rows), jsonl_file)
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        logger.info("Loaded %d polemo2 test texts.", len(all_texts))
        if limit is not None:
            return all_texts[:limit]
        return all_texts
        
    except Exception as e:
        logger.error("Failed to download polemo2 dataset: %s", e)
        logger.warning("You may need to install datasets: pip install datasets")
        return []


def _load_cache(cache_path: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Load name cache: layer_sig -> feature_idx -> {name, score}."""
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        out = {}
        for layer, feats in raw.items():
            out[layer] = {int(k): v for k, v in feats.items()}
        return out
    except Exception as e:
        logger.warning(f"Could not load cache {cache_path}: {e}")
        return {}


def _save_cache(cache_path: Path, cache: Dict[str, Dict[int, Dict[str, Any]]]) -> None:
    """Persist name cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    serial = {layer: {str(k): v for k, v in feats.items()} for layer, feats in cache.items()}
    cache_path.write_text(json.dumps(serial, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_offset(offset_path: Path) -> int:
    """Load last processed sentence index from offset file."""
    if not offset_path.exists():
        return 0
    try:
        content = offset_path.read_text(encoding="utf-8").strip()
        return int(content) if content else 0
    except (ValueError, IOError):
        return 0


def _save_offset(offset_path: Path, index: int) -> None:
    """Save last processed sentence index to offset file."""
    offset_path.parent.mkdir(parents=True, exist_ok=True)
    offset_path.write_text(str(index), encoding="utf-8")


def _validate_concepts_ollama(sentence: str, concepts_str: str, model: str, ollama_url: str) -> bool:
    """Ask Ollama if concepts reflect the sentence. Returns True for TAK, False otherwise."""
    prompt = _VALIDATION_TEMPLATE_5.format(sentence=sentence, concepts_str=concepts_str)
    raw = _call_ollama(prompt, model, ollama_url)
    u = raw.upper().strip()
    t = u.find("TAK")
    n = u.find("NIE")
    if t >= 0 and (n < 0 or t < n):
        return True
    return False


def _reset_manipulation(sae: Sae) -> None:
    """Reset SAE concept manipulation to baseline."""
    sae.concepts.multiplication.data = torch.ones_like(sae.concepts.multiplication.data)
    sae.concepts.bias.data = torch.zeros_like(sae.concepts.bias.data)


def _apply_concept_manipulation(sae: Sae, concept_indices: List[int], factor: float) -> None:
    """Amplify specific concept neurons by factor."""
    _reset_manipulation(sae)
    for idx in concept_indices:
        if 0 <= idx < sae.context.n_latents:
            sae.concepts.multiplication.data[idx] = factor


def _generate_text(
    lm: LanguageModel,
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 0.7,
    num_samples: int = 1,
) -> Union[str, List[str]]:
    """
    Generate continuation from prompt.
    
    Args:
        num_samples: If > 1, returns list of samples; otherwise single string.
    """
    dev = next(lm.model.parameters()).device
    encoded = lm.tokenizer([prompt], return_tensors="pt")
    encoded = {k: v.to(dev) for k, v in encoded.items()}
    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            out = lm.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=lm.tokenizer.eos_token_id,
                eos_token_id=lm.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
            text = lm.tokenizer.decode(out[0], skip_special_tokens=True)
            continuation = text[len(prompt):].strip()
            samples.append(continuation)
    return samples[0] if num_samples == 1 else samples


def _identify_most_steerable_concept(
    sentence: str,
    concepts: List[Dict[str, Any]],
    model: str,
    ollama_url: str,
) -> Optional[str]:
    """
    Ask LLM to identify which concept is most prone to steering.
    
    Args:
        sentence: Input sentence
        concepts: List of concept dicts with 'concept_name' key
        model: Ollama model name
        ollama_url: Ollama API URL
        
    Returns:
        Concept name that is most steerable, or None if failed
    """
    concepts_list = "\n".join(f"- {c['concept_name']}" for c in concepts)
    prompt = _STEERING_PRIORITY_TEMPLATE.format(sentence=sentence, concepts_list=concepts_list)
    
    try:
        response = _call_ollama(prompt, model, ollama_url)
        # Extract concept name from response (might have extra text)
        response_clean = response.strip().split("\n")[0].strip()
        
        # Try to match against our concept names
        concept_names = [c["concept_name"].lower() for c in concepts]
        response_lower = response_clean.lower()
        
        for i, name in enumerate(concept_names):
            if name in response_lower or response_lower in name:
                return concepts[i]["concept_name"]
        
        # If no match, return first word/phrase that looks like a concept name
        return response_clean
    except Exception as e:
        logger.warning("  Failed to identify steerable concept: %s", e)
        return None


def _check_bias_increase_ollama(
    prompt: str,
    concepts_str: str,
    baseline: Union[str, List[str]],
    steered: Union[str, List[str]],
    model: str,
    ollama_url: str,
) -> bool:
    """
    Ask Ollama if steered output is more biased towards concepts.
    If multiple samples provided, checks majority vote.
    Returns True for TAK.
    """
    if isinstance(baseline, list) and isinstance(steered, list):
        votes = []
        for b, s in zip(baseline, steered):
            full_prompt = _BIAS_CHECK_TEMPLATE.format(
                prompt=prompt,
                concepts_str=concepts_str,
                baseline=b,
                steered=s,
            )
            raw = _call_ollama(full_prompt, model, ollama_url)
            u = raw.upper().strip()
            t = u.find("TAK")
            n = u.find("NIE")
            votes.append(t >= 0 and (n < 0 or t < n))
        return sum(votes) > len(votes) / 2
    else:
        baseline_str = baseline if isinstance(baseline, str) else baseline[0]
        steered_str = steered if isinstance(steered, str) else steered[0]
        full_prompt = _BIAS_CHECK_TEMPLATE.format(
            prompt=prompt,
            concepts_str=concepts_str,
            baseline=baseline_str,
            steered=steered_str,
        )
        raw = _call_ollama(full_prompt, model, ollama_url)
        u = raw.upper().strip()
        t = u.find("TAK")
        n = u.find("NIE")
        if t >= 0 and (n < 0 or t < n):
            return True
        return False


def _pick_top_k_features(
    activations: List[Dict[str, Any]],
    k: int,
    skip_top: int = 0,
) -> List[Tuple[int, float]]:
    """
    Aggregate (feature_idx, abs_activation) across tokens, sort by abs_activation desc.
    Skip the first skip_top (often always-active); return the next k.
    """
    best: Dict[int, float] = {}
    for tok in activations:
        for f in tok["features"]:
            idx = f["feature_idx"]
            v = float(f.get("abs_activation", 0.0))
            best[idx] = max(best.get(idx, 0.0), v)
    by_val = sorted(best.items(), key=lambda x: -x[1])
    start = min(skip_top, len(by_val))
    return by_val[start : start + k]


def run_interactive_pipeline(
    lm: LanguageModel,
    sae: Sae,
    layer_sig: str,
    sentences: Iterator[str],
    top_texts_by_layer: Dict[str, Dict[int, List[Dict[str, Any]]]],
    cache: Dict[str, Dict[int, Dict[str, Any]]],
    cache_path: Path,
    target_results: int,
    n_features_per_prompt: int,
    skip_most_active: int,
    ollama_model: str,
    ollama_url: str,
    validation_retries: int,
    manipulation_samples: int = 3,
    manipulation_temperature: float = 0.7,
    top_concepts_for_manipulation: Optional[int] = None,
    offset_path: Optional[Path] = None,
    start_index: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[int, Dict[str, Any]]], int]:
    """
    No concept dict. Per sentence: infer → top-k features (skip first skip_most_active as
    often always-active) → name each via top texts (or cache). Validate with all named;
    require at least 5. Iterate until target_results saved.
    Returns (results, cache, last_processed_index).
    """
    results: List[Dict[str, Any]] = []
    top_texts = top_texts_by_layer.get(layer_sig, {})
    min_named = 5
    infer_top_k = max(5, n_features_per_prompt + skip_most_active)
    current_index = start_index

    hook_id = None
    try:
        hook_id = lm.layers.register_hook(layer_sig, sae)
        sae.context.lm = lm
        sae.context.lm_layer_signature = layer_sig
        logger.info("Registered SAE hook for layer %s (will remain active for all sentences)", layer_sig)

        for sent in sentences:
            if len(results) >= target_results:
                break
            current_index += 1
            logger.info("Processing sentence %d: %s...", current_index, sent[:60])
            try:
                activations = _run_inference(lm, sae, layer_sig, sent, top_k=infer_top_k)
                if not activations:
                    logger.warning("  No activations, skipping.")
                    continue

                selected = _pick_top_k_features(activations, n_features_per_prompt, skip_most_active)
                if len(selected) < min_named:
                    logger.warning("  Fewer than %d features after skip, skipping.", min_named)
                    continue

                names: Dict[int, str] = {}
                scores_map: Dict[int, float] = {}
                if layer_sig not in cache:
                    cache[layer_sig] = {}

                missing_top_texts: List[int] = []
                for idx, _ in selected:
                    if idx in cache[layer_sig]:
                        names[idx] = cache[layer_sig][idx]["name"]
                        scores_map[idx] = cache[layer_sig][idx]["score"]
                        continue
                    texts = top_texts.get(idx, [])
                    if not texts:
                        missing_top_texts.append(idx)
                        continue
                    res = _name_feature_ollama(idx, texts, ollama_model, ollama_url)
                    if res:
                        names[idx] = res["name"]
                        scores_map[idx] = res["score"]
                        cache[layer_sig][idx] = {"name": res["name"], "score": res["score"]}
                        logger.info("  Named feature %s -> %s (cached)", idx, res["name"])

                if len(names) < min_named:
                    logger.warning(
                        "  Could not name %d+ features, skipping. Selected %s; missing top texts for: %s. "
                        "Use top_texts from 03 run on the same layer so those features have examples.",
                        min_named,
                        [idx for idx, _ in selected],
                        missing_top_texts or "none (naming failed for some)",
                    )
                    continue

                named_selected = [(idx, v) for idx, v in selected if idx in names]
                concepts_list = [names[idx] for idx, _ in named_selected]
                concepts_str = ", ".join(concepts_list)
                ok = False
                for _ in range(validation_retries):
                    if _validate_concepts_ollama(sent, concepts_str, ollama_model, ollama_url):
                        ok = True
                        break
                if not ok:
                    logger.info("  Validation NIE: skipping sentence.")
                    continue

                logger.info("  Validation TAK: testing manipulation...")
                try:
                    prompt = sent.rstrip(".!?") + " "
                    
                    # Prepare concepts list for LLM (includes all concepts for validation)
                    concepts_list = [
                        {"feature_idx": idx, "concept_name": names[idx], "concept_score": scores_map[idx]}
                        for idx, _ in named_selected
                    ]
                    
                    # Filter out "czasownik" concepts from steering (but keep in concepts_list for validation)
                    concepts_for_steering = [c for c in concepts_list if c["concept_name"].lower() != "czasownik"]
                    if not concepts_for_steering:
                        logger.warning("  All concepts are 'czasownik', skipping steering test.")
                        continue
                    logger.info("  Filtered out %d 'czasownik' concepts from steering, %d concepts available for manipulation", 
                              len(concepts_list) - len(concepts_for_steering), len(concepts_for_steering))
                    
                    # Ask LLM which concept is most prone to steering (from non-czasownik concepts)
                    logger.info("  Identifying most steerable concept...")
                    most_steerable = _identify_most_steerable_concept(sent, concepts_for_steering, ollama_model, ollama_url)
                    
                    # Determine order: most steerable first, then others
                    concept_order = []
                    if most_steerable:
                        # Find the most steerable concept and put it first
                        for c in concepts_for_steering:
                            if c["concept_name"] == most_steerable:
                                concept_order.append(c)
                                break
                        # Add remaining concepts
                        for c in concepts_for_steering:
                            if c not in concept_order:
                                concept_order.append(c)
                        logger.info("  Most steerable concept identified: %s", most_steerable)
                    else:
                        # Fallback: use activation order
                        concept_order = sorted(concepts_for_steering, key=lambda x: scores_map.get(x["feature_idx"], 0.0), reverse=True)
                        logger.info("  Could not identify most steerable, using activation order")
                    
                    # Limit to top_concepts_for_manipulation if specified
                    if top_concepts_for_manipulation:
                        concept_order = concept_order[:top_concepts_for_manipulation]
                    
                    # Generate baseline once
                    lm.layers.disable_hook(hook_id)
                    baseline = _generate_text(
                        lm, prompt, max_new_tokens=30, temperature=0.7, num_samples=manipulation_samples
                    )
                    lm.layers.enable_hook(hook_id)
                    
                    baseline_str = baseline if isinstance(baseline, str) else "\n".join(f"  Sample {i+1}: {s}" for i, s in enumerate(baseline))
                    logger.info("  Baseline output:\n%s", baseline_str)
                    
                    # Test each concept individually
                    validated_concepts = []
                    all_manipulations = []
                    
                    for concept in concept_order:
                        concept_idx = concept["feature_idx"]
                        concept_name = concept["concept_name"]
                        logger.info("  Testing manipulation for concept: %s (feature %d)", concept_name, concept_idx)
                        
                        try:
                            _apply_concept_manipulation(sae, [concept_idx], factor=2.0)
                            steered = _generate_text(
                                lm, prompt, max_new_tokens=30, temperature=0.7, num_samples=manipulation_samples
                            )
                            _reset_manipulation(sae)
                            
                            steered_str = steered if isinstance(steered, str) else "\n".join(f"  Sample {i+1}: {s}" for i, s in enumerate(steered))
                            logger.info("  Steered output for %s:\n%s", concept_name, steered_str)
                            
                            # Check bias
                            concept_str = concept_name
                            bias_ok = False
                            for _ in range(validation_retries):
                                if _check_bias_increase_ollama(prompt, concept_str, baseline, steered, ollama_model, ollama_url):
                                    bias_ok = True
                                    break
                            
                            manipulation_result = {
                                "concept": concept,
                                "baseline_output": baseline,
                                "steered_output": steered,
                                "bias_check_passed": bias_ok,
                            }
                            all_manipulations.append(manipulation_result)
                            
                            if bias_ok:
                                logger.info("  Bias check TAK for %s", concept_name)
                                validated_concepts.append(concept_idx)
                            else:
                                logger.info("  Bias check NIE for %s", concept_name)
                        except Exception as e:
                            logger.warning("  Manipulation failed for %s: %s", concept_name, e)
                            manipulation_result = {
                                "concept": concept,
                                "baseline_output": baseline,
                                "steered_output": None,
                                "bias_check_passed": False,
                                "error": str(e),
                            }
                            all_manipulations.append(manipulation_result)
                    
                    # Save results: include all manipulations, but only if at least one passed bias check
                    if all_manipulations and validated_concepts:
                        concepts_out = [
                            {"feature_idx": idx, "concept_name": names[idx], "concept_score": scores_map[idx]}
                            for idx, _ in named_selected
                        ]
                        result_entry = {
                            "sentence": sent,
                            "prompt_used": prompt,
                            "concepts": concepts_out,
                            "most_steerable_concept": most_steerable,
                            "manipulations": [
                                {
                                    "concept_name": m["concept"]["concept_name"],
                                    "concept_idx": m["concept"]["feature_idx"],
                                    "baseline_output": m["baseline_output"],
                                    "steered_output": m["steered_output"],
                                    "bias_check_passed": m["bias_check_passed"],
                                    "error": m.get("error"),
                                }
                                for m in all_manipulations
                            ],
                            "validated_concepts": validated_concepts,
                        }
                        
                        logger.info("  Saving result with %d validated concept(s) out of %d tested", len(validated_concepts), len(all_manipulations))
                        results.append(result_entry)
                        _save_cache(cache_path, cache)
                    elif all_manipulations:
                        logger.info("  No concepts passed bias check, skipping result (tested %d concepts)", len(all_manipulations))
                    else:
                        logger.warning("  No manipulations performed, skipping.")
                except Exception as e:
                    logger.warning("  Manipulation test failed: %s, skipping.", e)
            finally:
                if offset_path is not None:
                    _save_offset(offset_path, current_index)
    finally:
        if hook_id is not None:
            lm.layers.unregister_hook(hook_id)
            logger.info("Unregistered SAE hook for layer %s", layer_sig)

    return results, cache, current_index


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive naming + inference (build from scratch)")
    ap.add_argument("--sae_paths", type=str, nargs="+", required=True)
    ap.add_argument("--layer_signatures", type=str, nargs="+", required=True)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--top_texts_dir", type=str, default=None, help="Top texts dir from 03 (latest batch per layer)")
    ap.add_argument("--top_texts_file", type=str, default=None, help="Single top_texts JSON (overrides dir for that layer)")
    ap.add_argument("--polemo2_test_limit", type=int, default=5000, help="Max polemo2 test texts to fetch")
    ap.add_argument("--target_results", type=int, default=10, help="Stop after this many validated results")
    ap.add_argument("--n_features_per_prompt", type=int, default=12, help="Take this many neurons per prompt (after skip)")
    ap.add_argument("--skip_most_active", type=int, default=2, help="Skip top N most active (often always-active)")
    ap.add_argument("--cache_path", type=str, default=None)
    ap.add_argument("--ollama_model", type=str, default="SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M")
    ap.add_argument("--ollama_url", type=str, default="http://localhost:11434")
    ap.add_argument("--validation_retries", type=int, default=2)
    ap.add_argument("--manipulation_samples", type=int, default=3, help="Number of samples per condition for manipulation test")
    ap.add_argument("--manipulation_temperature", type=float, default=0.7, help="Temperature for manipulation generation")
    ap.add_argument("--top_concepts_for_manipulation", type=int, default=None, help="Use only top N concepts for manipulation (default: all)")
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else _script_dir / "config.json"
    cfg = PipelineConfig.from_json_file(cfg_path)
    model_id = cfg.model.model_id
    store_dir = Path(cfg.storage.store_dir or str(_script_dir / "store"))
    device = cfg.storage.device or ("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir) if args.output_dir else _script_dir / "interactive_naming_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_path) if args.cache_path else out_dir / "name_cache.json"

    if not args.top_texts_dir and not args.top_texts_file:
        raise SystemExit("Provide either --top_texts_dir or --top_texts_file.")

    logger.info("Loading polemo2 test split...")
    # Load ALL texts from cache (ignore limit for offset tracking to work)
    # The limit was only for API fetching, but with local cache we want all texts
    all_sentences = _load_polemo2_test_texts(limit=None)
    logger.info("Loaded %d polemo2 test texts from cache.", len(all_sentences))
    
    if not all_sentences:
        logger.error("No sentences loaded! Check cache file or dataset download.")
        raise SystemExit("No sentences available to process.")

    offset_path = out_dir / "offset.txt"
    start_index = _load_offset(offset_path)
    if start_index > 0:
        logger.info("Resuming from sentence index %d (skipping first %d sentences)", start_index, start_index)
        sentences = all_sentences[start_index:]
        logger.info("Remaining sentences after offset: %d", len(sentences))
    else:
        logger.info("Starting from beginning (index 0)")
        sentences = all_sentences
    
    if not sentences:
        logger.warning("No sentences remaining after offset! Reset offset.txt to start from beginning.")
        raise SystemExit("No sentences to process after applying offset.")

    logger.info("Loading LM...")
    store = LocalStore(base_path=store_dir)
    lm = LanguageModel.from_huggingface(model_id, store=store, device=device)
    lm.model.eval()

    logger.info("Loading top texts...")
    if args.top_texts_file:
        top_texts = _load_top_texts_from_file(Path(args.top_texts_file))
    else:
        top_texts = _load_top_texts_latest_batch(Path(args.top_texts_dir))

    cache = _load_cache(cache_path)

    out_file = out_dir / "interactive_naming_results.json"
    all_results: Dict[str, Any] = {}
    if out_file.exists():
        try:
            existing = json.loads(out_file.read_text(encoding="utf-8"))
            all_results.update(existing)
            logger.info("Loaded existing results from %s", out_file)
        except Exception as e:
            logger.warning("Could not load existing results: %s", e)

    for sae_path_str, layer_sig in zip(args.sae_paths, args.layer_signatures):
        sae_path = Path(sae_path_str)
        logger.info("Loading SAE: %s (%s)", sae_path, layer_sig)
        sae = _load_sae_auto(sae_path, device=device)

        res, cache, last_index = run_interactive_pipeline(
            lm=lm,
            sae=sae,
            layer_sig=layer_sig,
            sentences=iter(sentences),
            top_texts_by_layer=top_texts,
            cache=cache,
            cache_path=cache_path,
            target_results=args.target_results,
            n_features_per_prompt=args.n_features_per_prompt,
            skip_most_active=args.skip_most_active,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            validation_retries=args.validation_retries,
            manipulation_samples=args.manipulation_samples,
            manipulation_temperature=args.manipulation_temperature,
            top_concepts_for_manipulation=args.top_concepts_for_manipulation,
            offset_path=offset_path,
            start_index=start_index,
        )
        key = f"{sae_path.stem}_{layer_sig}"
        if key in all_results:
            all_results[key].extend(res)
        else:
            all_results[key] = res
        start_index = last_index

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info("Saved results to %s (%d entries for %s)", out_file, len(all_results[key]), key)

    _save_cache(cache_path, cache)
    logger.info("Done. Results: %s, cache: %s, offset: %s", out_file, cache_path, offset_path)


if __name__ == "__main__":
    main()
