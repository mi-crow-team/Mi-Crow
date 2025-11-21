"""Inference engine for language models."""

from __future__ import annotations

from typing import Sequence, Any, Dict, List, TYPE_CHECKING

import torch
from torch import nn

from amber.language_model.utils import get_device_from_model, move_tensors_to_device, extract_logits_from_output

if TYPE_CHECKING:
    from amber.language_model.language_model import LanguageModel
    from amber.hooks.controller import Controller


class InferenceEngine:
    """Handles inference operations for LanguageModel."""
    
    def __init__(self, language_model: "LanguageModel"):
        """
        Initialize inference engine.
        
        Args:
            language_model: LanguageModel instance
        """
        self.lm = language_model
    
    def prepare_tokenizer_kwargs(self, tok_kwargs: Dict | None) -> Dict[str, Any]:
        """
        Prepare tokenizer keyword arguments with defaults.
        
        Args:
            tok_kwargs: Optional tokenizer keyword arguments
            
        Returns:
            Dictionary of tokenizer kwargs with defaults applied
        """
        if tok_kwargs is None:
            tok_kwargs = {}
        return {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
            **tok_kwargs,
        }
    
    def setup_trackers(self, texts: Sequence[str]) -> None:
        """
        Setup input trackers for current texts.
        
        Args:
            texts: Sequence of input texts
        """
        if self.lm._input_tracker is not None and self.lm._input_tracker.enabled:
            self.lm._input_tracker.set_current_texts(texts)
    
    def prepare_controllers(self, with_controllers: bool) -> List["Controller"]:
        """
        Prepare controllers for inference, disabling if needed.
        
        Args:
            with_controllers: Whether to keep controllers enabled
            
        Returns:
            List of controllers that were disabled (to restore later)
        """
        controllers_to_restore = []
        if not with_controllers:
            controllers = self.lm.layers.get_controllers()
            for controller in controllers:
                if controller.enabled:
                    controller.disable()
                    controllers_to_restore.append(controller)
        return controllers_to_restore
    
    def restore_controllers(self, controllers_to_restore: List["Controller"]) -> None:
        """
        Restore controllers that were disabled.
        
        Args:
            controllers_to_restore: List of controllers to restore
        """
        for controller in controllers_to_restore:
            controller.enable()
    
    def run_model_forward(
            self,
            enc: Dict[str, torch.Tensor],
            autocast: bool,
            device_type: str,
            autocast_dtype: torch.dtype | None
    ) -> Any:
        """
        Run model forward pass with optional autocast.
        
        Args:
            enc: Encoded inputs dictionary
            autocast: Whether to use automatic mixed precision
            device_type: Device type string ("cuda", "cpu", etc.)
            autocast_dtype: Optional dtype for autocast
            
        Returns:
            Model output
        """
        with torch.inference_mode():
            if autocast and device_type == "cuda":
                amp_dtype = autocast_dtype or torch.float16
                with torch.autocast(device_type, dtype=amp_dtype):
                    return self.lm.model(**enc)
            return self.lm.model(**enc)
    
    def execute_inference(
            self,
            texts: Sequence[str],
            tok_kwargs: Dict | None = None,
            autocast: bool = True,
            autocast_dtype: torch.dtype | None = None,
            with_controllers: bool = True,
    ) -> tuple[Any, Dict[str, torch.Tensor]]:
        """
        Execute inference on texts.
        
        Args:
            texts: Sequence of input texts
            tok_kwargs: Optional tokenizer keyword arguments
            autocast: Whether to use automatic mixed precision
            autocast_dtype: Optional dtype for autocast
            with_controllers: Whether to use controllers during inference
            
        Returns:
            Tuple of (model_output, encodings)
            
        Raises:
            ValueError: If texts is empty or tokenizer is not initialized
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        if self.lm.tokenizer is None:
            raise ValueError("Tokenizer must be initialized before running inference")

        tok_kwargs = self.prepare_tokenizer_kwargs(tok_kwargs)
        enc = self.lm.tokenize(texts, **tok_kwargs)

        device = get_device_from_model(self.lm.model)
        device_type = str(device.type)
        enc = move_tensors_to_device(enc, device)

        self.lm.model.eval()

        self.setup_trackers(texts)

        controllers_to_restore = self.prepare_controllers(with_controllers)

        try:
            output = self.run_model_forward(enc, autocast, device_type, autocast_dtype)
            return output, enc
        finally:
            self.restore_controllers(controllers_to_restore)
    
    def extract_logits(self, output: Any) -> torch.Tensor:
        """
        Extract logits tensor from model output.
        
        Args:
            output: Model output
            
        Returns:
            Logits tensor
        """
        return extract_logits_from_output(output)

