"""Model persistence (save/load) operations."""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
from typing import TYPE_CHECKING

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from mi_crow.language_model.contracts import ModelMetadata
from mi_crow.language_model.hook_metadata import collect_hooks_metadata
from mi_crow.language_model.utils import extract_model_id

if TYPE_CHECKING:
    from mi_crow.language_model.language_model import LanguageModel
    from mi_crow.store.store import Store


def save_model(
        language_model: "LanguageModel",
        path: Path | str | None = None
) -> Path:
    """
    Save the model and its metadata to the store.
    
    Args:
        language_model: LanguageModel instance to save
        path: Optional path to save the model. If None, defaults to {model_id}/model.pt
              relative to the store base path.
              
    Returns:
        Path where the model was saved
        
    Raises:
        ValueError: If store is not set
        OSError: If file operations fail
    """
    if language_model.store is None:
        raise ValueError("Store must be provided or set on the language model")
    
    # Determine save path
    if path is None:
        save_path = Path(language_model.store.base_path) / language_model.model_id / "model.pt"
    else:
        save_path = Path(path)
        # If path is relative, make it relative to store base path
        if not save_path.is_absolute():
            save_path = Path(language_model.store.base_path) / save_path
    
    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect hooks information
    hooks_info = collect_hooks_metadata(language_model.context)
    
    # Save model state dict
    model_state_dict = language_model.model.state_dict()
    
    # Create metadata
    metadata = ModelMetadata(
        model_id=language_model.model_id,
        hooks=hooks_info,
        model_path=str(save_path)
    )
    
    # Save everything in a single file
    payload = {
        "model_state_dict": model_state_dict,
        "metadata": asdict(metadata),
    }
    
    try:
        torch.save(payload, save_path)
    except OSError as e:
        raise OSError(
            f"Failed to save model to {save_path}. Error: {e}"
        ) from e
    
    from mi_crow.utils import get_logger
    logger = get_logger(__name__)
    logger.info(f"Saved model to {save_path}")
    
    return save_path


def load_model_from_saved_file(
        cls: type["LanguageModel"],
        saved_path: Path | str,
        store: "Store",
        model_id: str | None = None,
        device: str | torch.device | None = None,
) -> "LanguageModel":
    """
    Load a language model from a saved file (created by save_model).
    
    Args:
        cls: LanguageModel class
        saved_path: Path to the saved model file (.pt file)
        store: Store instance for persistence
        model_id: Optional model identifier. If not provided, will use the model_id from saved metadata.
                 If provided, will be used to load the model architecture from HuggingFace.
        device: Optional device string or torch.device (defaults to 'cpu' if None)
                 
    Returns:
        LanguageModel instance
        
    Raises:
        FileNotFoundError: If the saved file doesn't exist
        ValueError: If the saved file format is invalid or model_id is required but not provided
        RuntimeError: If model loading fails
    """
    if store is None:
        raise ValueError("store cannot be None")
    
    saved_path = Path(saved_path)
    if not saved_path.exists():
        raise FileNotFoundError(f"Saved model file not found: {saved_path}")
    
    # Load the saved payload
    try:
        payload = torch.load(saved_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model file {saved_path}. Error: {e}"
        ) from e
    
    # Validate payload structure
    if "model_state_dict" not in payload:
        raise ValueError(f"Invalid saved model format: missing 'model_state_dict' key in {saved_path}")
    if "metadata" not in payload:
        raise ValueError(f"Invalid saved model format: missing 'metadata' key in {saved_path}")
    
    model_state_dict = payload["model_state_dict"]
    metadata_dict = payload["metadata"]
    
    # Get model_id from metadata or use provided one
    saved_model_id = metadata_dict.get("model_id")
    if model_id is None:
        if saved_model_id is None:
            raise ValueError(
                f"model_id not found in saved metadata and not provided. "
                f"Please provide model_id parameter."
            )
        model_id = saved_model_id
    
    # Load model and tokenizer from HuggingFace using model_id
    # This assumes model_id is a valid HuggingFace model name
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as e:
        raise ValueError(
            f"Failed to load model '{model_id}' from HuggingFace. "
            f"Error: {e}. "
            f"Please ensure model_id is a valid HuggingFace model name."
        ) from e
    
    # Load the saved state dict into the model
    try:
        model.load_state_dict(model_state_dict)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load state dict into model '{model_id}'. Error: {e}"
        ) from e
    
    # Create LanguageModel instance
    lm = cls(model, tokenizer, store, model_id=model_id, device=device)
    
    # Note: Hooks are not automatically restored as they require hook instances
    # The hook metadata is available in metadata_dict["hooks"] if needed
    
    from mi_crow.utils import get_logger
    logger = get_logger(__name__)
    logger.info(f"Loaded model from {saved_path} (model_id: {model_id})")
    
    return lm

