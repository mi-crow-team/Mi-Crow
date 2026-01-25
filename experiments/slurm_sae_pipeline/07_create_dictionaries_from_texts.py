#!/usr/bin/env python3
"""
Script to create concept dictionaries from top texts collections using Ollama.

This script:
- Loads and aggregates top texts from collection directories
- Selects features with good examples using quality scoring
- Uses Ollama (local LLM) to generate Polish concept names
- Creates and saves concept dictionaries

Usage:
    python 07_create_dictionaries_from_texts.py \
        --top_texts_dir store/runs/top_texts_collection_20260119_225330 \
        --n_features 50 \
        --ollama_model llama3.2
"""

import argparse
import json
import re
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np

from mi_crow.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary
from mi_crow.utils import get_logger

logger = get_logger(__name__)

POLISH_PROMPT_TEMPLATE = """Jesteś ekspertem w analizie języków naturalnych, specjalizującym się w interpretowalności sieci neuronowych dla języka polskiego.

Zadanie: Przeanalizuj podane przykłady tekstów i zidentyfikuj pojedynczy, precyzyjny koncept semantyczny lub językowy, który aktywuje dany neuron w modelu językowym Bielik.

Instrukcje:
1. W każdym fragmencie tekstu najważniejszy token (miejsce najsilniejszej aktywacji) jest oznaczony jako <token>.
2. Przeanalizuj wszystkie przykłady i znajdź wspólny wzorzec semantyczny, gramatyczny lub morfologiczny.
3. Zwróć uwagę na:
   - Powtarzające się słowa, frazy lub tematy semantyczne
   - Kategorie gramatyczne (rzeczowniki, czasowniki, przymiotniki, itp.)
   - Cechy morfologiczne (końcówki fleksyjne, formy gramatyczne)
   - Konteksty użycia (np. język formalny, potoczny, specjalistyczny)

Kategorie konceptów:
- Semantyka: tematy, domeny, znaczenia (np. "pokoje_hotelowe", "treści_medyczne")
- Gramatyka i Morfologia: kategorie gramatyczne, formy (np. "rzeczowniki_zbiorowe", "czasowniki_dokonane")
- Składnia: struktury zdaniowe, relacje (np. "spójniki_współrzędne")
- Kontekst: style, rejestry językowe

Format odpowiedzi (TYLKO JSON, bez dodatkowego tekstu):
{{
  "nazwa_konceptu": "nazwa_w_formacie_snake_case",
  "opis": "Krótki opis konceptu (1-2 zdania)",
  "kategoria": "Semantyka|Gramatyka i Morfologia|Składnia|Kontekst",
  "pewność": 0.0-1.0
}}

Przykłady tekstów:
{examples}

Zwróć TYLKO poprawny JSON, bez żadnych dodatkowych komentarzy, wyjaśnień czy formatowania markdown."""


def load_and_aggregate_top_texts(collection_dir: Path) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """
    Load top texts from collection directory, using only the latest batch per layer.
    
    Args:
        collection_dir: Directory containing top_texts_layer_*.json files
        
    Returns:
        Dictionary mapping layer signature to feature index -> texts mapping
    """
    aggregated = defaultdict(lambda: defaultdict(list))
    
    if not collection_dir.exists():
        logger.warning(f"Collection directory not found: {collection_dir}")
        return {}
    
    json_files = list(collection_dir.glob("top_texts_layer_*.json"))
    if not json_files:
        logger.warning(f"No top_texts_layer_*.json files found in {collection_dir}")
        return {}
    
    logger.info(f"📥 Found {len(json_files)} top texts files")
    
    # Group files by layer signature and find latest batch for each
    layer_files = defaultdict(list)
    for json_file in json_files:
        layer_sig = _extract_layer_signature(json_file.name)
        if layer_sig:
            batch_num = _extract_batch_number(json_file.name)
            layer_files[layer_sig].append((batch_num, json_file))
    
    # Select only the latest batch for each layer
    selected_files = {}
    for layer_sig, files in layer_files.items():
        if files:
            latest_batch, latest_file = max(files, key=lambda x: x[0])
            selected_files[layer_sig] = latest_file
            logger.info(f"   Layer {layer_sig}: using batch {latest_batch} from {latest_file.name}")
    
    # Load only the selected files
    for layer_sig, json_file in selected_files.items():
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for feature_idx_str, texts in data.items():
                try:
                    feature_idx = int(feature_idx_str)
                    if isinstance(texts, list):
                        aggregated[layer_sig][feature_idx].extend(texts)
                except (ValueError, TypeError):
                    continue
        except Exception as e:
            logger.warning(f"Error loading {json_file}: {e}")
            continue
    
    for layer_sig, features in aggregated.items():
        total_features = len(features)
        total_texts = sum(len(texts) for texts in features.values())
        logger.info(f"   Layer {layer_sig}: {total_features} features, {total_texts} total texts")
    
    return dict(aggregated)


def _extract_layer_signature(filename: str) -> Optional[str]:
    """
    Extract layer signature from filename like 'top_texts_layer_0_llamaforcausallm_model_layers_15_batch_10.json'.
    
    Args:
        filename: Filename to parse
        
    Returns:
        Layer signature (e.g., 'llamaforcausallm_model_layers_15') or None
    """
    match = re.search(r"top_texts_layer_\d+_(llamaforcausallm_model_layers_\d+)", filename)
    if match:
        return match.group(1)
    return None


def _extract_batch_number(filename: str) -> int:
    """
    Extract batch number from filename like 'top_texts_layer_0_llamaforcausallm_model_layers_15_batch_10.json'.
    
    Args:
        filename: Filename to parse
        
    Returns:
        Batch number (e.g., 10) or 0 if not found
    """
    match = re.search(r"_batch_(\d+)\.json", filename)
    if match:
        return int(match.group(1))
    return 0


def format_texts_for_prompt(texts: List[Dict[str, Any]], max_texts: int = 15) -> str:
    """
    Format texts for Polish prompt with highlighted tokens.
    
    Args:
        texts: List of text dictionaries with 'text', 'token_idx', 'token_str', 'score'
        max_texts: Maximum number of texts to include
        
    Returns:
        Formatted string for prompt
    """
    texts_sorted = sorted(texts, key=lambda x: x.get("score", 0.0), reverse=True)[:max_texts]
    
    examples = []
    for i, text_data in enumerate(texts_sorted, 1):
        text = text_data.get("text", "")
        token_idx = text_data.get("token_idx", 0)
        
        words = text.split()
        if 0 <= token_idx < len(words):
            words[token_idx] = f"<{words[token_idx]}>"
        
        example = f"{i}. {' '.join(words)}"
        examples.append(example)
    
    return "\n".join(examples)


def call_ollama(
    prompt: str,
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
) -> Dict[str, Any]:
    """
    Call Ollama API to generate concept name and description.
    
    Args:
        prompt: Formatted prompt with text examples
        model: Ollama model name (default: "llama3.2")
        ollama_url: Ollama API URL (default: "http://localhost:11434")
        
    Returns:
        Dictionary with nazwa_konceptu, opis, kategoria, pewność
    """
    api_url = f"{ollama_url}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
        }
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get("response", "")
        
        if not response_text.strip():
            raise ValueError("Empty response from Ollama")
        
        json_match = re.search(r"\{[^{}]*nazwa_konceptu[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
        else:
            parsed = json.loads(response_text)
        
        return {
            "nazwa_konceptu": parsed.get("nazwa_konceptu", "Nieznany"),
            "opis": parsed.get("opis", ""),
            "kategoria": parsed.get("kategoria", "Semantyka"),
            "pewność": float(parsed.get("pewność", 0.5)),
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from Ollama response: {e}")
        logger.error(f"Response text: {response_text[:500]}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calling Ollama: {e}")
        raise


def evaluate_example_quality(texts: List[Dict[str, Any]]) -> float:
    """
    Evaluate quality of examples for a feature.
    
    Args:
        texts: List of text examples with 'text', 'score', 'token_str', etc.
        
    Returns:
        Quality score (0-1, higher is better)
    """
    if not texts:
        return 0.0
    
    n_examples = len(texts)
    if n_examples < 3:
        return 0.0
    
    avg_score = np.mean([t.get("score", 0.0) for t in texts])
    max_score = max([t.get("score", 0.0) for t in texts], default=0.0)
    
    text_lengths = [len(t.get("text", "").split()) for t in texts]
    avg_length = np.mean(text_lengths) if text_lengths else 0.0
    
    unique_tokens = set()
    for t in texts:
        token_str = t.get("token_str", "")
        if token_str:
            unique_tokens.add(token_str.lower())
    token_diversity = len(unique_tokens) / max(n_examples, 1)
    
    unique_contexts = set()
    for t in texts:
        text = t.get("text", "")
        words = text.split()
        if len(words) >= 3:
            context = " ".join(words[:3]).lower()
            unique_contexts.add(context)
    context_diversity = len(unique_contexts) / max(n_examples, 1)
    
    quality_score = (
        0.3 * min(n_examples / 15.0, 1.0) +
        0.3 * min(avg_score / 10.0, 1.0) +
        0.2 * min(avg_length / 20.0, 1.0) +
        0.1 * token_diversity +
        0.1 * context_diversity
    )
    
    return min(1.0, quality_score)


def select_features_for_dictionary(
    aggregated_texts: Dict[int, List[Dict[str, Any]]],
    n_features: int = 50,
) -> List[int]:
    """
    Select top N features based on example quality.
    
    Args:
        aggregated_texts: Dictionary mapping feature index to text examples
        n_features: Number of features to select
        
    Returns:
        List of selected feature indices sorted by quality
    """
    feature_scores = {}
    
    for feature_idx, texts in aggregated_texts.items():
        
        quality_score = evaluate_example_quality(texts)
        
        if quality_score > 0.0:
            total_activation = sum(t.get("score", 0.0) for t in texts)
            avg_activation = total_activation / len(texts)
            
            combined_score = quality_score * 0.7 + min(avg_activation / 20.0, 1.0) * 0.3
            feature_scores[feature_idx] = {
                "combined_score": combined_score,
                "quality_score": quality_score,
                "avg_activation": avg_activation,
                "n_examples": len(texts),
            }
    
    sorted_features = sorted(
        feature_scores.items(),
        key=lambda x: x[1]["combined_score"],
        reverse=True
    )
    
    selected = [idx for idx, _ in sorted_features[:n_features]]
    
    logger.info(f"📊 Selected {len(selected)} features with good examples:")
    logger.info(f"   Top 5: {selected[:5]}")
    for idx in selected[:5]:
        if idx in feature_scores:
            info = feature_scores[idx]
            logger.info(f"      Feature {idx}: quality={info['quality_score']:.2f}, "
                      f"activation={info['avg_activation']:.2f}, examples={info['n_examples']}")
    
    return selected


def determine_model_and_layer(layer_signature: str) -> Tuple[str, int]:
    """
    Determine model and layer from layer signature.
    
    Args:
        layer_signature: Layer signature (e.g., 'llamaforcausallm_model_layers_15')
        
    Returns:
        Tuple of (model_name, layer_num)
    """
    match = re.search(r"layers_(\d+)", layer_signature)
    if not match:
        raise ValueError(f"Could not extract layer number from signature: {layer_signature}")
    
    layer_num = int(match.group(1))
    
    if layer_num in [15, 20]:
        model_name = "bielik-1.5b"
    elif layer_num in [28, 38]:
        model_name = "bielik-4.5b"
    else:
        raise ValueError(f"Unknown layer number {layer_num}, cannot determine model")
    
    return model_name, layer_num


def generate_concept_name_ollama(
    feature_idx: int,
    texts: List[Dict[str, Any]],
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
) -> Optional[Dict[str, Any]]:
    """
    Generate concept name using Ollama.
    
    Args:
        feature_idx: Feature index
        texts: List of text examples
        model: Ollama model name
        ollama_url: Ollama API URL
        
    Returns:
        Dictionary with name, category, confidence or None if failed
    """
    try:
        texts_formatted = format_texts_for_prompt(texts, max_texts=15)
        
        if not texts_formatted.strip():
            logger.warning(f"   No valid examples after formatting for feature {feature_idx}")
            return None
        
        prompt = POLISH_PROMPT_TEMPLATE.format(examples=texts_formatted)
        explanation = call_ollama(prompt, model=model, ollama_url=ollama_url)
        
        return {
            "name": explanation["nazwa_konceptu"],
            "category": explanation["kategoria"],
            "confidence": explanation["pewność"],
        }
    except Exception as e:
        logger.error(f"   Error generating name for feature {feature_idx}: {e}")
        return None


def generate_concepts_report(
    concept_dict: ConceptDictionary,
    aggregated_texts: Dict[int, List[Dict[str, Any]]],
    output_path: Path,
) -> None:
    """
    Generate a text report showing concepts with sample texts.
    
    Args:
        concept_dict: Concept dictionary with named concepts
        aggregated_texts: Dictionary mapping feature index to text examples
        output_path: Path to save the report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("CONCEPT DICTIONARY REPORT")
    lines.append("=" * 80)
    lines.append(f"\nTotal concepts: {len(concept_dict.concepts_map)}")
    lines.append(f"Generated: {Path(__file__).name}\n")
    lines.append("=" * 80)
    lines.append("")
    
    sorted_concepts = sorted(
        concept_dict.concepts_map.items(),
        key=lambda x: x[1].score,
        reverse=True
    )
    
    for feature_idx, concept in sorted_concepts:
        lines.append(f"\n{'='*80}")
        lines.append(f"Feature {feature_idx}: {concept.name}")
        lines.append(f"Score: {concept.score:.3f}")
        lines.append(f"{'='*80}")
        
        if feature_idx in aggregated_texts:
            texts = aggregated_texts[feature_idx]
            texts_sorted = sorted(texts, key=lambda x: x.get("score", 0.0), reverse=True)
            sample_texts = texts_sorted[:5]
            
            lines.append(f"\nSample texts (top 5 of {len(texts)} total):")
            lines.append("-" * 80)
            
            for i, text_data in enumerate(sample_texts, 1):
                text = text_data.get("text", "")
                score = text_data.get("score", 0.0)
                token_idx = text_data.get("token_idx", 0)
                token_str = text_data.get("token_str", "")
                
                words = text.split()
                if 0 <= token_idx < len(words):
                    words[token_idx] = f"[{words[token_idx]}]"
                    highlighted_text = " ".join(words)
                else:
                    highlighted_text = text
                    if token_str and token_str in text:
                        highlighted_text = text.replace(token_str, f"[{token_str}]", 1)
                
                lines.append(f"\n{i}. Activation: {score:.3f}")
                lines.append(f"   Token: '{token_str}' (index {token_idx})")
                lines.append(f"   Text: {highlighted_text}")
        else:
            lines.append("\nNo text examples available for this feature.")
        
        lines.append("")
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"📄 Generated concepts report: {output_path}")


def create_concept_dictionary(
    aggregated_texts: Dict[int, List[Dict[str, Any]]],
    n_size: int,
    selected_features: List[int],
    ollama_model: str,
    ollama_url: str,
) -> ConceptDictionary:
    """
    Create concept dictionary by generating names for selected features.
    
    Args:
        aggregated_texts: Dictionary mapping feature index to text examples
        n_size: Number of latents in SAE
        selected_features: List of feature indices to name
        ollama_model: Ollama model name
        ollama_url: Ollama API URL
        
    Returns:
        ConceptDictionary with generated names
    """
    concept_dict = ConceptDictionary(n_size=n_size)
    
    for i, feature_idx in enumerate(selected_features, 1):
        if feature_idx not in aggregated_texts:
            logger.warning(f"   Feature {feature_idx} has no texts, skipping")
            continue
        
        texts = aggregated_texts[feature_idx]
        if not texts:
            continue
        
        logger.info(f"📝 Naming feature {feature_idx} ({i}/{len(selected_features)})...")
        logger.info(f"   Examples: {len(texts)}, Avg activation: {np.mean([t.get('score', 0.0) for t in texts]):.3f}")
        
        result = generate_concept_name_ollama(
            feature_idx,
            texts,
            model=ollama_model,
            ollama_url=ollama_url,
        )
        
        if result:
            concept_name = result["name"]
            confidence = result["confidence"]
            
            concept_dict.add(feature_idx, concept_name, confidence)
            
            logger.info(f"   ✅ Named: {concept_name} (confidence: {confidence:.2f})")
        else:
            logger.warning(f"   ❌ Failed to generate name for feature {feature_idx}")
    
    return concept_dict


def main():
    parser = argparse.ArgumentParser(description="Create concept dictionaries from top texts using Ollama")
    parser.add_argument("--top_texts_dir", type=str, required=True, help="Directory containing top_texts_layer_*.json files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for dictionaries (default: dictionaries/ in script dir)")
    parser.add_argument("--n_features", type=int, default=50, help="Number of features to name (default: 50)")
    parser.add_argument("--ollama_model", type=str, default="SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M", help="Ollama model name (default: SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M)")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434", help="Ollama API URL (default: http://localhost:11434)")
    parser.add_argument("--layer_signature", type=str, default=None, help="Specific layer signature to process (default: process all layers)")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    if args.output_dir:
        output_base_dir = Path(args.output_dir)
    else:
        output_base_dir = script_dir / "dictionaries"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    top_texts_dir = Path(args.top_texts_dir)
    
    logger.info("🚀 Starting Concept Dictionary Creation")
    logger.info(f"📁 Top texts directory: {top_texts_dir}")
    logger.info(f"🤖 Ollama model: {args.ollama_model}, URL: {args.ollama_url}")
    
    logger.info("📥 Loading and aggregating top texts...")
    aggregated_by_layer = load_and_aggregate_top_texts(top_texts_dir)
    
    if not aggregated_by_layer:
        logger.error("❌ No top texts found. Please check the directory path.")
        return
    
    layers_to_process = [args.layer_signature] if args.layer_signature else list(aggregated_by_layer.keys())
    
    for layer_sig in layers_to_process:
        if layer_sig not in aggregated_by_layer:
            logger.warning(f"Layer signature {layer_sig} not found in aggregated texts, skipping")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing layer: {layer_sig}")
        logger.info(f"{'='*60}")
        
        aggregated_texts = aggregated_by_layer[layer_sig]
        
        try:
            model_name, layer_num = determine_model_and_layer(layer_sig)
        except ValueError as e:
            logger.error(f"❌ {e}")
            continue
        
        logger.info(f"📊 Model: {model_name}, Layer: {layer_num}")
        
        logger.info(f"📊 Analyzing top texts to find features with good examples...")
        selected_features = select_features_for_dictionary(
            aggregated_texts,
            n_features=args.n_features,
        )
        
        if not selected_features:
            logger.warning(f"❌ No features selected for layer {layer_sig}")
            continue
        
        logger.info(f"📊 Selected {len(selected_features)} features to name")
        
        output_dir = output_base_dir / model_name / f"layer_{layer_num}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        concept_dict = ConceptDictionary(n_size=6144)
        concept_dict.set_directory(output_dir)
        
        if (output_dir / "concepts.json").exists():
            try:
                concept_dict.load()
                logger.info(f"   Loaded existing concept dictionary with {len(concept_dict.concepts_map)} concepts")
            except Exception as e:
                logger.warning(f"   Could not load existing dictionary: {e}")
        
        logger.info("📝 Generating concept names...")
        new_dict = create_concept_dictionary(
            aggregated_texts,
            n_size=6144,
            selected_features=selected_features,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
        )
        
        for feature_idx, concept in new_dict.concepts_map.items():
            concept_dict.add(feature_idx, concept.name, concept.score)
        
        concept_dict.save()
        logger.info(f"💾 Saved concept dictionary to {output_dir / 'concepts.json'}")
        logger.info(f"✅ Total concepts: {len(concept_dict.concepts_map)}")
        
        logger.info("📄 Generating concepts report...")
        report_path = output_dir / "concepts_report.txt"
        generate_concepts_report(concept_dict, aggregated_texts, report_path)
    
    logger.info("\n✅ Concept dictionary creation complete!")


if __name__ == "__main__":
    main()
