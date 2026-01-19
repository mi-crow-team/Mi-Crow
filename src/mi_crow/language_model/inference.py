"""Inference engine for language models."""

from __future__ import annotations

import datetime
from typing import Sequence, Any, Dict, List, TYPE_CHECKING

import torch
from torch import nn

from mi_crow.language_model.utils import move_tensors_to_device, extract_logits_from_output
from mi_crow.language_model.device_manager import sync_model_to_context_device
from mi_crow.utils import get_logger

if TYPE_CHECKING:
    from mi_crow.language_model.language_model import LanguageModel
    from mi_crow.hooks.controller import Controller
    from mi_crow.datasets import BaseDataset
    from mi_crow.store.store import Store

logger = get_logger(__name__)


class _EarlyStopInference(Exception):
    """Internal exception used to stop model forward pass after a specific layer."""

    def __init__(self, output: Any):
        super().__init__("Early stop after requested layer")
        self.output = output


class InferenceEngine:
    """Handles inference operations for LanguageModel."""
    
    def __init__(self, language_model: "LanguageModel"):
        """
        Initialize inference engine.
        
        Args:
            language_model: LanguageModel instance
        """
        self.lm = language_model
    
    def _prepare_tokenizer_kwargs(self, tok_kwargs: Dict | None) -> Dict[str, Any]:
        """
        Prepare tokenizer keyword arguments with defaults.

        Args:
            tok_kwargs: Optional tokenizer keyword arguments

        Returns:
            Dictionary of tokenizer kwargs with defaults applied
        """
        if tok_kwargs is None:
            tok_kwargs = {}
        
        padding_strategy = tok_kwargs.pop("padding", True)
        if padding_strategy is True and "max_length" in tok_kwargs:
            padding_strategy = "longest"
        
        result = {
            "padding": padding_strategy,
            "truncation": True,
            "return_tensors": "pt",
            **tok_kwargs,
        }
        
        return result
    
    def _setup_trackers(self, texts: Sequence[str]) -> None:
        """
        Setup input trackers for current texts.

        Args:
            texts: Sequence of input texts
        """
        if self.lm._input_tracker is not None and self.lm._input_tracker.enabled:
            self.lm._input_tracker.set_current_texts(texts)
    
    def _setup_model_input_detectors(self, enc: Dict[str, torch.Tensor]) -> None:
        """
        Automatically set inputs from encodings for all registered ModelInputDetector hooks.

        This is necessary because PyTorch's pre_forward hook doesn't receive kwargs,
        so ModelInputDetector hooks can't automatically capture attention masks when
        models are called with **kwargs (e.g., model(**encodings)).

        Args:
            enc: Encoded inputs dictionary
        """
        from mi_crow.hooks.implementations.model_input_detector import ModelInputDetector
        
        detectors = self.lm.layers.get_detectors()
        for detector in detectors:
            if isinstance(detector, ModelInputDetector):
                detector.set_inputs_from_encodings(enc, module=self.lm.model)
    
    def _prepare_controllers(self, with_controllers: bool) -> List["Controller"]:
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
    
    def _restore_controllers(self, controllers_to_restore: List["Controller"]) -> None:
        """
        Restore controllers that were disabled.

        Args:
            controllers_to_restore: List of controllers to restore
        """
        for controller in controllers_to_restore:
            controller.enable()
    
    def _run_model_forward(
            self,
            enc: Dict[str, torch.Tensor],
            autocast: bool,
            device_type: str,
            autocast_dtype: torch.dtype | None,
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
        try:
            with torch.inference_mode():
                if autocast and device_type == "cuda":
                    amp_dtype = autocast_dtype or torch.float16
                    with torch.autocast(device_type, dtype=amp_dtype):
                        return self.lm.model(**enc)
                return self.lm.model(**enc)
        except _EarlyStopInference as e:
            # Early stopping hook raised this to short‑circuit the remaining forward pass.
            # We return the output captured at the requested layer.
            return e.output
    
    def execute_inference(
            self,
            texts: Sequence[str],
            tok_kwargs: Dict | None = None,
            autocast: bool = True,
            autocast_dtype: torch.dtype | None = None,
            with_controllers: bool = True,
            stop_after_layer: str | int | None = None,
    ) -> tuple[Any, Dict[str, torch.Tensor]]:
        """
        Execute inference on texts.
        
        Args:
            texts: Sequence of input texts
            tok_kwargs: Optional tokenizer keyword arguments
            autocast: Whether to use automatic mixed precision
            autocast_dtype: Optional dtype for autocast
            with_controllers: Whether to use controllers during inference
            stop_after_layer: Optional layer signature (name or index) after which
                the forward pass should be stopped early
            
        Returns:
            Tuple of (model_output, encodings)
            
        Raises:
            ValueError: If texts is empty or tokenizer is not initialized
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        if self.lm.tokenizer is None:
            raise ValueError("Tokenizer must be initialized before running inference")

        tok_kwargs = self._prepare_tokenizer_kwargs(tok_kwargs)
        logger.debug(f"[DEBUG] About to tokenize {len(texts)} texts...")
        enc = self.lm.tokenize(texts, **tok_kwargs)
        logger.debug(f"[DEBUG] Tokenization completed, shape: {enc['input_ids'].shape if isinstance(enc, dict) else 'N/A'}")

        device = torch.device(self.lm.context.device)
        device_type = str(device.type)

        sync_model_to_context_device(self.lm)

        enc = move_tensors_to_device(enc, device)

        self.lm.model.eval()

        self._setup_trackers(texts)
        self._setup_model_input_detectors(enc)

        controllers_to_restore = self._prepare_controllers(with_controllers)

        hook_handle = None
        try:
            if stop_after_layer is not None:
                # Register a temporary forward hook that stops the forward pass
                def _early_stop_hook(module: nn.Module, inputs: tuple, output: Any):
                    raise _EarlyStopInference(output)

                hook_handle = self.lm.layers.register_forward_hook_for_layer(
                    stop_after_layer, _early_stop_hook
                )

            output = self._run_model_forward(enc, autocast, device_type, autocast_dtype)
            return output, enc
        finally:
            if hook_handle is not None:
                try:
                    hook_handle.remove()
                except Exception:
                    pass
            self._restore_controllers(controllers_to_restore)
    
    def extract_logits(self, output: Any) -> torch.Tensor:
        """
        Extract logits tensor from model output.
        
        Args:
            output: Model output
            
        Returns:
            Logits tensor
        """
        return extract_logits_from_output(output)
    
    def _extract_dataset_info(self, dataset: "BaseDataset | None") -> Dict[str, Any]:
        """
        Extract dataset information for metadata.
        
        Args:
            dataset: Optional dataset instance
            
        Returns:
            Dictionary with dataset information
        """
        if dataset is None:
            return {}
        
        try:
            ds_id = str(getattr(dataset, "dataset_dir", ""))
            ds_len = int(len(dataset))
            return {
                "dataset_dir": ds_id,
                "length": ds_len,
            }
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return {
                "dataset_dir": "",
                "length": -1,
            }
    
    def _prepare_run_metadata(
        self,
        dataset: "BaseDataset | None" = None,
        run_name: str | None = None,
        options: Dict[str, Any] | None = None,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Prepare run metadata dictionary.
        
        Args:
            dataset: Optional dataset (for dataset info)
            run_name: Optional run name (generates if None)
            options: Optional dict of options to include
            
        Returns:
            Tuple of (run_name, metadata_dict)
        """
        if run_name is None:
            run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if options is None:
            options = {}
        
        dataset_info = self._extract_dataset_info(dataset)
        
        meta: Dict[str, Any] = {
            "run_name": run_name,
            "model": getattr(self.lm.model, "model_name", self.lm.model.__class__.__name__),
            "options": options.copy(),
        }
        
        if dataset_info:
            meta["dataset"] = dataset_info
        
        return run_name, meta
    
    @staticmethod
    def _save_run_metadata(
        store: "Store",
        run_name: str,
        meta: Dict[str, Any],
        verbose: bool = False,
    ) -> None:
        """
        Save run metadata to store.
        
        Args:
            store: Store to save to
            run_name: Run name
            meta: Metadata dictionary
            verbose: Whether to log
        """
        try:
            store.put_run_metadata(run_name, meta)
        except (OSError, IOError, ValueError, RuntimeError) as e:
            if verbose:
                logger.warning(f"Failed to save run metadata for {run_name}: {e}")
    
    def infer_texts(
        self,
        texts: Sequence[str],
        run_name: str | None = None,
        batch_size: int | None = None,
        tok_kwargs: Dict | None = None,
        autocast: bool = True,
        autocast_dtype: torch.dtype | None = None,
        with_controllers: bool = True,
        clear_detectors_before: bool = False,
        verbose: bool = False,
        stop_after_layer: str | int | None = None,
        save_in_batches: bool = True,
    ) -> tuple[Any, Dict[str, torch.Tensor]] | tuple[List[Any], List[Dict[str, torch.Tensor]]]:
        """
        Run inference on list of strings with optional metadata saving.
        
        Args:
            texts: Sequence of input texts
            run_name: Optional run name for saving metadata (if None, no metadata saved)
            batch_size: Optional batch size for processing (if None, processes all at once)
            tok_kwargs: Optional tokenizer keyword arguments
            autocast: Whether to use automatic mixed precision
            autocast_dtype: Optional dtype for autocast
            with_controllers: Whether to use controllers during inference
            clear_detectors_before: If True, clears all detector state before running
            verbose: Whether to log progress
            stop_after_layer: Optional layer signature (name or index) after which
                the forward pass should be stopped early
            save_in_batches: If True, save detector metadata in per‑batch
                directories. If False, aggregate all detector metadata for
                the run under a single detectors directory.
            
        Returns:
            If batch_size is None or >= len(texts): Tuple of (model_output, encodings)
            If batch_size < len(texts): Tuple of (list of outputs, list of encodings)
            
        Raises:
            ValueError: If texts is empty or tokenizer is not initialized
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        if self.lm.tokenizer is None:
            raise ValueError("Tokenizer must be initialized before running inference")
        
        if clear_detectors_before:
            self.lm.clear_detectors()

        store = self.lm.store
        if run_name is not None and store is None:
            raise ValueError("Store must be provided to save metadata")
        
        if batch_size is None or batch_size >= len(texts):
            output, enc = self.execute_inference(
                texts,
                tok_kwargs=tok_kwargs,
                autocast=autocast,
                autocast_dtype=autocast_dtype,
                with_controllers=with_controllers,
                stop_after_layer=stop_after_layer,
            )
            
            if run_name is not None:
                options = {
                    "batch_size": len(texts),
                    "max_length": tok_kwargs.get("max_length") if tok_kwargs else None,
                }
                _, meta = self._prepare_run_metadata(dataset=None, run_name=run_name, options=options)
                self._save_run_metadata(store, run_name, meta, verbose)
                self.lm.save_detector_metadata(run_name, 0, unified=not save_in_batches)
            
            return output, enc
        
        all_outputs = []
        all_encodings = []
        batch_counter = 0
        
        if run_name is not None:
            options = {
                "batch_size": batch_size,
                "max_length": tok_kwargs.get("max_length") if tok_kwargs else None,
            }
            _, meta = self._prepare_run_metadata(dataset=None, run_name=run_name, options=options)
            self._save_run_metadata(store, run_name, meta, verbose)
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            output, enc = self.execute_inference(
                batch_texts,
                tok_kwargs=tok_kwargs,
                autocast=autocast,
                autocast_dtype=autocast_dtype,
                with_controllers=with_controllers,
                stop_after_layer=stop_after_layer,
            )
            
            all_outputs.append(output)
            all_encodings.append(enc)
            
            if run_name is not None:
                self.lm.save_detector_metadata(run_name, batch_counter, unified=not save_in_batches)
                if verbose:
                    logger.info(f"Saved batch {batch_counter} for run={run_name}")
            
            batch_counter += 1
        
        return all_outputs, all_encodings
    
    def infer_dataset(
        self,
        dataset: "BaseDataset",
        run_name: str | None = None,
        batch_size: int = 32,
        tok_kwargs: Dict | None = None,
        autocast: bool = True,
        autocast_dtype: torch.dtype | None = None,
        with_controllers: bool = True,
        free_cuda_cache_every: int | None = 0,
        clear_detectors_before: bool = False,
        verbose: bool = False,
        stop_after_layer: str | int | None = None,
        save_in_batches: bool = True,
    ) -> str:
        """
        Run inference on whole dataset with metadata saving.
        
        Args:
            dataset: Dataset to process
            run_name: Optional run name (generated if None)
            batch_size: Batch size for processing
            tok_kwargs: Optional tokenizer keyword arguments
            autocast: Whether to use automatic mixed precision
            autocast_dtype: Optional dtype for autocast
            with_controllers: Whether to use controllers during inference
            free_cuda_cache_every: Clear CUDA cache every N batches (0 or None to disable)
            clear_detectors_before: If True, clears all detector state before running
            verbose: Whether to log progress
            stop_after_layer: Optional layer signature (name or index) after which
                the forward pass should be stopped early
            
        Returns:
            Run name used for saving
            
        Raises:
            ValueError: If model or store is not initialized
        """
        if clear_detectors_before:
            self.lm.clear_detectors()

        model: nn.Module | None = self.lm.model
        if model is None:
            raise ValueError("Model must be initialized before running")
        
        store = self.lm.store
        if store is None:
            raise ValueError("Store must be provided or set on the language model")
        
        device = torch.device(self.lm.context.device)
        device_type = str(device.type)
        
        options = {
            "max_length": tok_kwargs.get("max_length") if tok_kwargs else None,
            "batch_size": int(batch_size),
        }
        
        run_name, meta = self._prepare_run_metadata(dataset=dataset, run_name=run_name, options=options)
        
        if verbose:
            logger.info(
                f"Starting infer_dataset: run={run_name}, "
                f"batch_size={batch_size}, device={device_type}"
            )
        
        self._save_run_metadata(store, run_name, meta, verbose)
        
        batch_counter = 0
        
        with torch.inference_mode():
            for batch_index, batch in enumerate(dataset.iter_batches(batch_size)):
                if not batch:
                    continue
                
                texts = dataset.extract_texts_from_batch(batch)
                
                self.execute_inference(
                    texts,
                    tok_kwargs=tok_kwargs,
                    autocast=autocast,
                    autocast_dtype=autocast_dtype,
                    with_controllers=with_controllers,
                    stop_after_layer=stop_after_layer,
                )
                
                self.lm.save_detector_metadata(run_name, batch_index, unified=not save_in_batches)
                
                batch_counter += 1
                
                if device_type == "cuda" and free_cuda_cache_every and free_cuda_cache_every > 0:
                    if (batch_counter % free_cuda_cache_every) == 0:
                        torch.cuda.empty_cache()
                        if verbose:
                            logger.info("Emptied CUDA cache")
                
                if verbose:
                    logger.info(f"Saved batch {batch_index} for run={run_name}")
        
        if verbose:
            logger.info(f"Completed infer_dataset: run={run_name}, batches_saved={batch_counter}")
        
        return run_name

