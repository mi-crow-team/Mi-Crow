#!/usr/bin/env python3
"""
SLURM script to validate SAE models with comprehensive metrics.

This script calculates:
- Fraction of Variance Explained (FVE)
- L0 Sparsity (active feature count per token)
- Cross-Entropy Loss Degradation
- Dead Latent Ratio
- Feature Frequency Distribution

Usage:
    python 06_validate_sae.py --sae_paths path1.pt path2.pt --config config.json
"""

import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from config import PipelineConfig
from mi_crow.datasets import TextDataset
from mi_crow.language_model.language_model import LanguageModel
from mi_crow.mechanistic.sae.sae import Sae
from mi_crow.mechanistic.sae.modules.topk_sae import TopKSae
from mi_crow.mechanistic.sae.modules.l1_sae import L1Sae
from mi_crow.store.local_store import LocalStore
from mi_crow.store.store_dataloader import StoreDataloader
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
    
    Args:
        sae_path: Path to SAE model file
        device: Device to load model to
        
    Returns:
        Loaded SAE instance
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


def calculate_fve(sae: Sae, activations_loader: StoreDataloader) -> float:
    """
    Calculate Fraction of Variance Explained (FVE).
    
    Formula: FVE = 1 - Σ||x_i - x̂_i||² / Σ||x_i - μ||²
    
    Args:
        sae: SAE model
        activations_loader: DataLoader for validation activations
        
    Returns:
        FVE score (0-1, higher is better)
    """
    total_reconstruction_error = 0.0
    total_variance = 0.0
    n_samples = 0
    mean_activation = None
    
    sae.eval()
    with torch.no_grad():
        for batch in activations_loader:
            if isinstance(batch, dict):
                batch = batch.get("activations", None)
                if batch is None:
                    continue
            
            if batch is None or not isinstance(batch, torch.Tensor):
                continue
            
            if len(batch.shape) > 2:
                batch = batch.view(-1, batch.shape[-1])
            
            batch = batch.to(sae.context.device)
            
            if mean_activation is None:
                mean_activation = batch.mean(dim=0, keepdim=True)
            else:
                mean_activation = (mean_activation * n_samples + batch.mean(dim=0, keepdim=True) * batch.shape[0]) / (n_samples + batch.shape[0])
            
            x_hat = sae.forward(batch)
            
            reconstruction_error = torch.sum((batch - x_hat) ** 2)
            variance = torch.sum((batch - mean_activation) ** 2)
            
            total_reconstruction_error += reconstruction_error.item()
            total_variance += variance.item()
            n_samples += batch.shape[0]
    
    if total_variance == 0:
        return 0.0
    
    fve = 1.0 - (total_reconstruction_error / total_variance)
    return max(0.0, min(1.0, fve))


def calculate_l0_sparsity(sae: Sae, activations_loader: StoreDataloader) -> Dict[str, float]:
    """
    Calculate L0 sparsity (average number of active features per token).
    
    Args:
        sae: SAE model
        activations_loader: DataLoader for validation activations
        
    Returns:
        Dictionary with mean, median, std of active feature counts
    """
    active_counts = []
    
    sae.eval()
    with torch.no_grad():
        for batch in activations_loader:
            if isinstance(batch, dict):
                batch = batch.get("activations", None)
                if batch is None:
                    continue
            
            if batch is None or not isinstance(batch, torch.Tensor):
                continue
            
            if len(batch.shape) > 2:
                batch = batch.view(-1, batch.shape[-1])
            
            batch = batch.to(sae.context.device)
            
            codes = sae.encode(batch)
            
            threshold = 1e-6
            active = (torch.abs(codes) > threshold).sum(dim=1)
            active_counts.extend(active.cpu().tolist())
    
    if not active_counts:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}
    
    active_counts = np.array(active_counts)
    return {
        "mean": float(np.mean(active_counts)),
        "median": float(np.median(active_counts)),
        "std": float(np.std(active_counts)),
    }


def calculate_ce_degradation(
    lm: LanguageModel,
    sae: Sae,
    layer_sig: str,
    validation_texts: List[str],
    max_samples: int = 1000,
) -> float:
    """
    Calculate Cross-Entropy loss degradation by replacing activations with SAE reconstruction.
    
    Args:
        lm: Language model
        sae: SAE model
        layer_sig: Layer signature where SAE is attached
        validation_texts: List of validation text samples
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Relative perplexity increase: (PPL_SAE - PPL_baseline) / PPL_baseline
    """
    lm.model.eval()
    sae.eval()
    
    total_loss_baseline = 0.0
    total_loss_sae = 0.0
    total_tokens = 0
    
    device = lm.context.device
    
    class ReconstructionHook:
        def __init__(self, sae: Sae):
            self.sae = sae
            self.original_output = None
        
        def __call__(self, module, input, output):
            if isinstance(output, torch.Tensor):
                tensor = output
            elif hasattr(output, "last_hidden_state"):
                tensor = output.last_hidden_state
            elif isinstance(output, (tuple, list)):
                tensor = next((item for item in output if isinstance(item, torch.Tensor)), None)
            else:
                return output
            
            if tensor is None:
                return output
            
            original_shape = tensor.shape
            if len(original_shape) > 2:
                tensor_flat = tensor.reshape(-1, original_shape[-1])
            else:
                tensor_flat = tensor
            
            x_hat = self.sae.forward(tensor_flat)
            
            if len(original_shape) > 2:
                x_hat = x_hat.reshape(original_shape)
            
            if isinstance(output, torch.Tensor):
                return x_hat
            elif hasattr(output, "last_hidden_state"):
                output.last_hidden_state = x_hat
                return output
            elif isinstance(output, (tuple, list)):
                return (x_hat,) + output[1:]
            return output
    
    hook = ReconstructionHook(sae)
    
    with torch.no_grad():
        for i, text in enumerate(validation_texts[:max_samples]):
            try:
                inputs = lm.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                if inputs["input_ids"].shape[1] < 2:
                    continue
                
                input_ids = inputs["input_ids"]
                labels = input_ids.clone()
                
                baseline_outputs = lm.model(**inputs)
                baseline_logits = baseline_outputs.logits if hasattr(baseline_outputs, "logits") else baseline_outputs[0]
                
                shift_logits = baseline_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                baseline_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum")
                
                total_loss_baseline += baseline_loss.item()
                
                hook_handle = None
                try:
                    if layer_sig in lm.layers.name_to_layer:
                        layer = lm.layers.name_to_layer[layer_sig]
                        hook_handle = layer.register_forward_hook(hook)
                    
                    sae_outputs = lm.model(**inputs)
                    sae_logits = sae_outputs.logits if hasattr(sae_outputs, "logits") else sae_outputs[0]
                    
                    shift_logits = sae_logits[..., :-1, :].contiguous()
                    sae_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="sum")
                    
                    total_loss_sae += sae_loss.item()
                finally:
                    if hook_handle is not None:
                        hook_handle.remove()
                
                total_tokens += (input_ids.shape[1] - 1)
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
    
    if total_tokens == 0:
        return 0.0
    
    ppl_baseline = torch.exp(torch.tensor(total_loss_baseline / total_tokens))
    ppl_sae = torch.exp(torch.tensor(total_loss_sae / total_tokens))
    
    relative_increase = (ppl_sae - ppl_baseline) / ppl_baseline
    return float(relative_increase)


def calculate_dead_features(
    sae: Sae,
    activations_loader: StoreDataloader,
    threshold: float = 1e-6,
) -> float:
    """
    Calculate percentage of features that never activate.
    
    Args:
        sae: SAE model
        activations_loader: DataLoader for activations
        threshold: Activation threshold
        
    Returns:
        Dead feature ratio (0-1)
    """
    feature_activations = defaultdict(int)
    n_samples = 0
    
    sae.eval()
    with torch.no_grad():
        for batch in activations_loader:
            if isinstance(batch, dict):
                batch = batch.get("activations", None)
                if batch is None:
                    continue
            
            if batch is None or not isinstance(batch, torch.Tensor):
                continue
            
            if len(batch.shape) > 2:
                batch = batch.view(-1, batch.shape[-1])
            
            batch = batch.to(sae.context.device)
            
            codes = sae.encode(batch)
            active = torch.abs(codes) > threshold
            
            for i in range(codes.shape[1]):
                if active[:, i].any():
                    feature_activations[i] += 1
            
            n_samples += codes.shape[0]
    
    n_features = sae.context.n_latents
    n_dead = n_features - len(feature_activations)
    
    return float(n_dead / n_features) if n_features > 0 else 0.0


def calculate_feature_frequency_distribution(
    sae: Sae,
    activations_loader: StoreDataloader,
    threshold: float = 1e-6,
) -> Dict[int, int]:
    """
    Calculate feature activation frequency distribution.
    
    Args:
        sae: SAE model
        activations_loader: DataLoader for activations
        threshold: Activation threshold
        
    Returns:
        Dictionary mapping feature index to activation count
    """
    feature_counts = defaultdict(int)
    
    sae.eval()
    with torch.no_grad():
        for batch in activations_loader:
            if isinstance(batch, dict):
                batch = batch.get("activations", None)
                if batch is None:
                    continue
            
            if batch is None or not isinstance(batch, torch.Tensor):
                continue
            
            if len(batch.shape) > 2:
                batch = batch.view(-1, batch.shape[-1])
            
            batch = batch.to(sae.context.device)
            
            codes = sae.encode(batch)
            active = torch.abs(codes) > threshold
            
            for i in range(codes.shape[1]):
                count = active[:, i].sum().item()
                feature_counts[i] += count
    
    return dict(feature_counts)


def plot_feature_frequency_distribution(
    feature_counts: Dict[int, int],
    output_path: Path,
    sae_name: str,
) -> None:
    """
    Plot feature frequency distribution (log-log plot).
    
    Args:
        feature_counts: Dictionary mapping feature index to activation count
        output_path: Path to save plot
        sae_name: Name of SAE for title
    """
    if not feature_counts:
        logger.warning("No feature counts to plot")
        return
    
    counts = sorted(feature_counts.values(), reverse=True)
    ranks = np.arange(1, len(counts) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, counts, "b.", alpha=0.6)
    plt.xlabel("Feature Rank (by frequency)")
    plt.ylabel("Activation Count")
    plt.title(f"Feature Frequency Distribution - {sae_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Validate SAE models")
    parser.add_argument("--sae_paths", type=str, nargs="+", help="Paths to SAE model files")
    parser.add_argument("--layer_signatures", type=str, nargs="+", help="Layer signatures for each SAE")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file")
    parser.add_argument("--run_id", type=str, default=None, help="Run ID for validation activations")
    parser.add_argument("--validation_tokens", type=int, default=1000000, help="Number of validation tokens")
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
        output_dir = script_dir / "validation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.sae_paths:
        logger.error("❌ Error: --sae_paths is required")
        return
    
    if not args.layer_signatures:
        logger.error("❌ Error: --layer_signatures is required")
        return
    
    if len(args.sae_paths) != len(args.layer_signatures):
        logger.error("❌ Error: Number of SAE paths must match number of layer signatures")
        return
    
    logger.info("🚀 Starting SAE Validation")
    logger.info(f"📱 Using device: {DEVICE}")
    logger.info(f"🔧 Model: {MODEL_ID}")
    logger.info(f"📁 Store directory: {STORE_DIR}")
    
    logger.info("📥 Loading language model...")
    store = LocalStore(base_path=STORE_DIR)
    lm = LanguageModel.from_huggingface(MODEL_ID, store=store, device=DEVICE)
    lm.model.eval()
    
    if args.run_id:
        run_id = args.run_id
    else:
        run_id_file = STORE_DIR / "run_id.txt"
        if not run_id_file.exists():
            logger.error("❌ Error: run_id.txt not found. Please provide --run_id")
            return
        with open(run_id_file, "r") as f:
            run_id = f.read().strip()
    
    logger.info(f"📁 Using run ID: {run_id}")
    
    validation_dataset = TextDataset(
        hf_dataset=cfg.dataset.hf_dataset,
        data_split="validation" if cfg.dataset.data_split == "train" else cfg.dataset.data_split,
        text_field=cfg.dataset.text_field,
        data_limit=None,
    )
    
    validation_texts = [item["text"] for item in validation_dataset[:1000]]
    
    results = {}
    
    for sae_path_str, layer_sig in zip(args.sae_paths, args.layer_signatures):
        sae_path = Path(sae_path_str)
        logger.info(f"📥 Loading SAE from {sae_path} for layer {layer_sig}...")
        
        sae = load_sae_auto(sae_path, device=DEVICE)
        sae_name = f"{sae_path.stem}_{layer_sig}"
        
        logger.info(f"✅ Loaded {type(sae).__name__}: {sae.context.n_inputs} -> {sae.context.n_latents}")
        
        activations_loader = StoreDataloader(
            store=store,
            run_id=run_id,
            layer=layer_sig,
            key="activations",
            batch_size=32,
            device=DEVICE,
            max_batches=None,
        )
        
        logger.info("📊 Calculating FVE...")
        fve = calculate_fve(sae, activations_loader)
        logger.info(f"   FVE: {fve:.4f}")
        
        logger.info("📊 Calculating L0 sparsity...")
        l0_stats = calculate_l0_sparsity(sae, activations_loader)
        logger.info(f"   Mean active features: {l0_stats['mean']:.2f} ± {l0_stats['std']:.2f}")
        
        logger.info("📊 Calculating dead features...")
        dead_ratio = calculate_dead_features(sae, activations_loader)
        logger.info(f"   Dead feature ratio: {dead_ratio:.4f} ({dead_ratio*100:.2f}%)")
        
        logger.info("📊 Calculating feature frequency distribution...")
        feature_counts = calculate_feature_frequency_distribution(sae, activations_loader)
        plot_feature_frequency_distribution(feature_counts, plots_dir / f"feature_frequency_{sae_name}.png", sae_name)
        
        logger.info("📊 Calculating CE degradation...")
        ce_degradation = calculate_ce_degradation(lm, sae, layer_sig, validation_texts)
        logger.info(f"   Relative PPL increase: {ce_degradation:.4f} ({ce_degradation*100:.2f}%)")
        
        results[sae_name] = {
            "fve": fve,
            "l0_sparsity": l0_stats,
            "dead_feature_ratio": dead_ratio,
            "ce_degradation": ce_degradation,
            "n_features": sae.context.n_latents,
            "n_inputs": sae.context.n_inputs,
        }
    
    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Validation complete! Results saved to {report_path}")


if __name__ == "__main__":
    main()
