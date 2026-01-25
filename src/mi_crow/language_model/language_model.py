from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path
from typing import Sequence, Any, Dict, List, TYPE_CHECKING, Set, Tuple

import torch
from torch import nn, Tensor
from transformers import PreTrainedTokenizerBase

from mi_crow.language_model.layers import LanguageModelLayers
from mi_crow.language_model.tokenizer import LanguageModelTokenizer
from mi_crow.language_model.activations import LanguageModelActivations
from mi_crow.language_model.context import LanguageModelContext
from mi_crow.language_model.inference import InferenceEngine
from mi_crow.language_model.persistence import save_model, load_model_from_saved_file
from mi_crow.language_model.initialization import initialize_model_id, create_from_huggingface, create_from_local_torch
from mi_crow.language_model.device_manager import normalize_device, sync_model_to_context_device
from mi_crow.store.store import Store
from mi_crow.utils import get_logger

if TYPE_CHECKING:
    from mi_crow.mechanistic.sae.concepts.input_tracker import InputTracker

logger = get_logger(__name__)


def _extract_special_token_ids(tokenizer: PreTrainedTokenizerBase) -> Set[int]:
    """
    Extract special token IDs from a tokenizer.
    
    Prioritizes the common case (all_special_ids) and falls back to
    individual token ID attributes for edge cases.
    
    Handles cases where token_id attributes may be lists (e.g., eos_token_id: [4, 2]).
    
    Args:
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Set of special token IDs
    """
    special_ids = set()
    
    # Common case: most tokenizers have all_special_ids
    if hasattr(tokenizer, 'all_special_ids'):
        all_special_ids = tokenizer.all_special_ids
        if all_special_ids and isinstance(all_special_ids, (list, tuple, set)):
            special_ids.update(all_special_ids)
            return special_ids  # Early return for common case
    
    # Fallback: extract from individual token ID attributes
    def add_token_id(token_id):
        if token_id is None:
            return
        if isinstance(token_id, (list, tuple)):
            special_ids.update(token_id)
        else:
            special_ids.add(token_id)
    
    token_id_attrs = ['pad_token_id', 'eos_token_id', 'bos_token_id', 'unk_token_id', 
                     'cls_token_id', 'sep_token_id', 'mask_token_id']
    for attr in token_id_attrs:
        token_id = getattr(tokenizer, attr, None)
        add_token_id(token_id)
    
    return special_ids


class LanguageModel:
    """
    Fence-style language model wrapper.
    
    Provides a unified interface for working with language models, including:
    - Model initialization and configuration
    - Inference operations through the inference property
    - Hook management (detectors and controllers)
    - Model persistence
    - Activation tracking
    """

    def __init__(
            self,
            model: nn.Module,
            tokenizer: PreTrainedTokenizerBase,
            store: Store,
            model_id: str | None = None,
            device: str | torch.device | None = None,
    ):
        """
        Initialize LanguageModel.
        
        Args:
            model: PyTorch model module
            tokenizer: HuggingFace tokenizer
            store: Store instance for persistence
            model_id: Optional model identifier (auto-extracted if not provided)
            device: Optional device string or torch.device (defaults to 'cpu' if None)
        """
        self.context = LanguageModelContext(self)
        self.context.model = model
        self.context.tokenizer = tokenizer
        self.context.model_id = initialize_model_id(model, model_id)
        self.context.store = store
        self.context.special_token_ids = _extract_special_token_ids(tokenizer)
        self.context.device = normalize_device(device)
        sync_model_to_context_device(self)

        self.layers = LanguageModelLayers(self.context)
        self.lm_tokenizer = LanguageModelTokenizer(self.context)
        self.activations = LanguageModelActivations(self.context)
        self.inference = InferenceEngine(self)

        self._input_tracker: "InputTracker | None" = None


    @property
    def model(self) -> nn.Module:
        """Get the underlying PyTorch model."""
        return self.context.model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Get the tokenizer."""
        return self.context.tokenizer

    @property
    def model_id(self) -> str:
        """Get the model identifier."""
        return self.context.model_id

    @property
    def store(self) -> Store:
        """Get the store instance."""
        return self.context.store

    @store.setter
    def store(self, value: Store) -> None:
        """Set the store instance."""
        self.context.store = value

    def tokenize(self, texts: Sequence[str], **kwargs: Any) -> Any:
        """
        Tokenize texts using the language model tokenizer.
        
        Args:
            texts: Sequence of text strings to tokenize
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Tokenized encodings
        """
        return self.lm_tokenizer.tokenize(texts, **kwargs)

    def get_input_tracker(self) -> "InputTracker | None":
        """
        Get the input tracker instance if it exists.
        
        Returns:
            InputTracker instance or None
        """
        return self._input_tracker

    def get_all_detector_metadata(self) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Tensor]]]:
        """
        Get metadata from all registered detectors.
        
        Returns:
            Tuple of (detectors_metadata, detectors_tensor_metadata)
        """
        detectors = self.layers.get_detectors()
        detectors_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)
        detectors_tensor_metadata: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)

        for detector in detectors:
            detectors_metadata[detector.layer_signature] = dict(detector.metadata)
            detectors_tensor_metadata[detector.layer_signature] = dict(detector.tensor_metadata)

        return detectors_metadata, detectors_tensor_metadata

    def clear_detectors(self) -> None:
        """
        Clear all accumulated metadata for registered detectors.

        This is useful when running multiple independent inference runs
        (e.g. separate `infer_texts` / `infer_dataset` calls) and you want
        to ensure that detector state does not leak between runs.
        """
        detectors = self.layers.get_detectors()
        for detector in detectors:
            detector.metadata.clear()
            detector.tensor_metadata.clear()

            clear_captured = getattr(detector, "clear_captured", None)
            if callable(clear_captured):
                clear_captured()

    def save_detector_metadata(self, run_name: str, batch_idx: int | None, unified: bool = False, clear_after_save: bool = True) -> str:
        """
        Save detector metadata to store.
        
        Args:
            run_name: Name of the run
            batch_idx: Batch index. Ignored when ``unified`` is True.
            unified: If True, save metadata in a single detectors directory
                for the whole run instead of per‑batch directories.
            clear_after_save: If True, clear detector metadata after saving to free memory.
                Defaults to True to prevent OOM errors when processing large batches.
            
        Returns:
            Path where metadata was saved
            
        Raises:
            ValueError: If store is not set
        """
        if self.store is None:
            raise ValueError("Store must be provided or set on the language model")
        
        detectors_metadata, detectors_tensor_metadata = self.get_all_detector_metadata()
        
        if unified:
            result = self.store.put_run_detector_metadata(run_name, detectors_metadata, detectors_tensor_metadata)
        else:
            if batch_idx is None:
                raise ValueError("batch_idx must be provided when unified is False")
            result = self.store.put_detector_metadata(run_name, batch_idx, detectors_metadata, detectors_tensor_metadata)
        
        if clear_after_save:
            for layer_signature in list(detectors_tensor_metadata.keys()):
                detector_tensors = detectors_tensor_metadata[layer_signature]
                for tensor_key in list(detector_tensors.keys()):
                    del detector_tensors[tensor_key]
                del detectors_tensor_metadata[layer_signature]
            detectors_metadata.clear()
            
            detectors = self.layers.get_detectors()
            for detector in detectors:
                clear_captured = getattr(detector, "clear_captured", None)
                if callable(clear_captured):
                    clear_captured()
                for key in list(detector.tensor_metadata.keys()):
                    del detector.tensor_metadata[key]
                detector.metadata.clear()
            
            gc.collect()
        
        return result

    def _ensure_input_tracker(self) -> "InputTracker":
        """
        Ensure InputTracker singleton exists.
        
        Returns:
            The InputTracker instance
        """
        if self._input_tracker is not None:
            return self._input_tracker

        from mi_crow.mechanistic.sae.concepts.input_tracker import InputTracker

        self._input_tracker = InputTracker(language_model=self)

        logger.debug(f"Created InputTracker singleton for {self.context.model_id}")

        return self._input_tracker

    def save_model(self, path: Path | str | None = None) -> Path:
        """
        Save the model and its metadata to the store.
        
        Args:
            path: Optional path to save the model. If None, defaults to {model_id}/model.pt
                  relative to the store base path.
                  
        Returns:
            Path where the model was saved
            
        Raises:
            ValueError: If store is not set
        """
        return save_model(self, path)

    @classmethod
    def from_huggingface(
            cls,
            model_name: str,
            store: Store,
            tokenizer_params: dict = None,
            model_params: dict = None,
            device: str | torch.device | None = None,
    ) -> "LanguageModel":
        """
        Load a language model from HuggingFace Hub.
        
        Automatically loads model to GPU if device is "cuda" and CUDA is available.
        This prevents OOM errors by keeping the model on GPU instead of CPU RAM.
        
        Args:
            model_name: HuggingFace model identifier
            store: Store instance for persistence
            tokenizer_params: Optional tokenizer parameters
            model_params: Optional model parameters
            device: Target device ("cuda", "cpu", "mps"). If "cuda" and CUDA is available,
                model will be loaded directly to GPU using device_map="auto"
                (via the HuggingFace factory helpers).
            
        Returns:
            LanguageModel instance
        """
        return create_from_huggingface(cls, model_name, store, tokenizer_params, model_params, device)

    @classmethod
    def from_local_torch(
            cls,
            model_path: str,
            tokenizer_path: str,
            store: Store,
            device: str | torch.device | None = None,
    ) -> "LanguageModel":
        """
        Load a language model from local HuggingFace paths.
        
        Args:
            model_path: Path to the model directory or file
            tokenizer_path: Path to the tokenizer directory or file
            store: Store instance for persistence
            device: Optional device string or torch.device (defaults to 'cpu' if None)
            
        Returns:
            LanguageModel instance
        """
        return create_from_local_torch(cls, model_path, tokenizer_path, store, device)

    @classmethod
    def from_local(
            cls,
            saved_path: Path | str,
            store: Store,
            model_id: str | None = None,
            device: str | torch.device | None = None,
    ) -> "LanguageModel":
        """
        Load a language model from a saved file (created by save_model).
        
        Args:
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
        """
        return load_model_from_saved_file(cls, saved_path, store, model_id, device)
