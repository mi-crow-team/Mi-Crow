from typing import TYPE_CHECKING, Dict, Any
import datetime

import torch
from torch import nn

from amber.adapters import BaseDataset
from amber.hooks import HookType
from amber.hooks.implementations.activation_saver import LayerActivationDetector
from amber.store.store import Store
from amber.utils import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class LanguageModelActivations:
    """Handles activation saving and processing for LanguageModel."""
    
    def __init__(self, context: "LanguageModelContext"):
        """
        Initialize LanguageModelActivations.
        
        Args:
            context: LanguageModelContext instance
        """
        self.context = context

    def _setup_detector(
            self,
            layer_signature: str | int,
            hook_id_suffix: str
    ) -> tuple[LayerActivationDetector, str]:
        """
        Create and register an activation detector.
        
        Args:
            layer_signature: Layer to attach detector to
            hook_id_suffix: Suffix for hook ID
            
        Returns:
            Tuple of (detector, hook_id)
        """
        detector = LayerActivationDetector(
            layer_signature=layer_signature,
            hook_id=f"detector_{hook_id_suffix}"
        )

        hook_id = self.context.language_model.layers.register_hook(
            layer_signature,
            detector,
            HookType.FORWARD
        )

        return detector, hook_id

    def _cleanup_detector(self, hook_id: str) -> None:
        """
        Unregister a detector hook.
        
        Args:
            hook_id: Hook ID to unregister
        """
        try:
            self.context.language_model.layers.unregister_hook(hook_id)
        except (KeyError, ValueError, RuntimeError):
            pass

    def _normalize_layer_signatures(
            self,
            layer_signatures: str | int | list[str | int] | None
    ) -> tuple[str | None, list[str]]:
        """
        Normalize layer signatures to string format.
        
        Args:
            layer_signatures: Single layer signature or list of layer signatures
            
        Returns:
            Tuple of (single layer string or None, list of layer strings)
        """
        if isinstance(layer_signatures, (str, int)):
            layer_sig_str = str(layer_signatures)
            layer_sig_list = [layer_sig_str]
        elif isinstance(layer_signatures, list):
            layer_sig_list = [str(sig) for sig in layer_signatures]
            layer_sig_str = layer_sig_list[0] if len(layer_sig_list) == 1 else None
        else:
            layer_sig_str = None
            layer_sig_list = []
        return layer_sig_str, layer_sig_list

    def _extract_dataset_info(self, dataset: BaseDataset | None) -> Dict[str, Any]:
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
            layer_signatures: str | int | list[str | int] | None,
            dataset: BaseDataset | None = None,
            run_name: str | None = None,
            options: Dict[str, Any] | None = None,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Prepare run metadata dictionary.
        
        Args:
            layer_signatures: Single layer signature or list of layer signatures
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

        # Convert layer_signatures to list of strings
        if isinstance(layer_signatures, (str, int)):
            layer_sig_list = [str(layer_signatures)]
        elif isinstance(layer_signatures, list):
            layer_sig_list = [str(sig) for sig in layer_signatures]
        else:
            layer_sig_list = []

        dataset_info = self._extract_dataset_info(dataset)

        meta: Dict[str, Any] = {
            "run_name": run_name,
            "model": getattr(self.context.model, "model_name", self.context.model.__class__.__name__),
            "options": options.copy(),
        }

        if layer_sig_list:
            meta["layer_signatures"] = layer_sig_list
            meta["num_layers"] = len(layer_sig_list)
        if dataset_info:
            meta["dataset"] = dataset_info

        return run_name, meta

    @staticmethod
    def _save_run_metadata(
            store: Store,
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
        logger = get_logger(__name__)
        try:
            store.put_run_metadata(run_name, meta)
        except (OSError, IOError, ValueError, RuntimeError) as e:
            if verbose:
                logger.warning(f"Failed to save run metadata for {run_name}: {e}")

    def _process_batch(
            self,
            texts: list[str],
            run_name: str,
            batch_index: int,
            max_length: int | None,
            autocast: bool,
            autocast_dtype: torch.dtype | None,
            dtype: torch.dtype | None,
            verbose: bool
    ) -> None:
        """
        Process a single batch of texts.
        
        Args:
            texts: List of text strings
            run_name: Run name
            batch_index: Batch index
            max_length: Optional max length for tokenization
            autocast: Whether to use autocast
            autocast_dtype: Optional dtype for autocast
            dtype: Optional dtype to convert activations to
            verbose: Whether to log progress
        """
        if not texts:
            return

        tok_kwargs = {}
        if max_length is not None:
            tok_kwargs["max_length"] = max_length

        self.context.language_model._inference_engine.execute_inference(
            texts,
            tok_kwargs=tok_kwargs,
            autocast=autocast,
            autocast_dtype=autocast_dtype,
        )

        if dtype is not None:
            self._convert_activations_to_dtype(dtype)

        self.context.language_model.save_detector_metadata(run_name, batch_index)

        if verbose:
            logger.info(f"Saved batch {batch_index} for run={run_name}")

    def _convert_activations_to_dtype(self, dtype: torch.dtype) -> None:
        """
        Convert captured activations to specified dtype.
        
        Args:
            dtype: Target dtype
        """
        detectors = self.context.language_model.layers.get_detectors()
        for detector in detectors:
            if "activations" in detector.tensor_metadata:
                detector.tensor_metadata["activations"] = detector.tensor_metadata["activations"].to(dtype)

    def _manage_cuda_cache(
            self,
            batch_counter: int,
            free_cuda_cache_every: int | None,
            device_type: str,
            verbose: bool
    ) -> None:
        """
        Manage CUDA cache clearing.
        
        Args:
            batch_counter: Current batch counter
            free_cuda_cache_every: Clear cache every N batches (0 or None to disable)
            device_type: Device type string
            verbose: Whether to log
        """
        if device_type == "cuda" and free_cuda_cache_every and free_cuda_cache_every > 0:
            if (batch_counter % free_cuda_cache_every) == 0:
                torch.cuda.empty_cache()
                if verbose:
                    logger.info("Emptied CUDA cache")

    def save_activations_dataset(
            self,
            dataset: BaseDataset,
            layer_signature: str | int,
            run_name: str | None = None,
            batch_size: int = 32,
            *,
            dtype: torch.dtype | None = None,
            max_length: int | None = None,
            autocast: bool = True,
            autocast_dtype: torch.dtype | None = None,
            free_cuda_cache_every: int | None = 0,
            verbose: bool = False,
    ) -> str:
        """
        Save activations from a dataset.
        
        Args:
            dataset: Dataset to process
            layer_signature: Layer signature to capture activations from
            run_name: Optional run name (generated if None)
            batch_size: Batch size for processing
            dtype: Optional dtype to convert activations to
            max_length: Optional max length for tokenization
            autocast: Whether to use autocast
            autocast_dtype: Optional dtype for autocast
            free_cuda_cache_every: Clear CUDA cache every N batches (0 or None to disable)
            verbose: Whether to log progress
            
        Returns:
            Run name used for saving
            
        Raises:
            ValueError: If model or store is not initialized
        """
        model: nn.Module | None = self.context.model
        if model is None:
            raise ValueError("Model must be initialized before running")

        store = self.context.store
        if store is None:
            raise ValueError("Store must be provided or set on the language model")
        
        from amber.language_model.utils import get_device_from_model
        device = get_device_from_model(model)
        device_type = str(device.type)

        options = {
            "dtype": str(dtype) if dtype is not None else None,
            "max_length": max_length,
            "batch_size": int(batch_size),
        }

        run_name, meta = self._prepare_run_metadata(
            layer_signature, dataset=dataset, run_name=run_name, options=options
        )

        if verbose:
            logger.info(
                f"Starting save_activations_dataset: run={run_name}, layer={layer_signature}, "
                f"batch_size={batch_size}, device={device_type}"
            )

        self._save_run_metadata(store, run_name, meta, verbose)

        detector, hook_id = self._setup_detector(layer_signature, f"save_{run_name}")
        batch_counter = 0

        try:
            with torch.inference_mode():
                for batch_index, texts in enumerate(dataset.iter_batches(batch_size)):
                    self._process_batch(
                        texts,
                        run_name,
                        batch_index,
                        max_length,
                        autocast,
                        autocast_dtype,
                        dtype,
                        verbose
                    )
                    batch_counter += 1
                    self._manage_cuda_cache(batch_counter, free_cuda_cache_every, device_type, verbose)
        finally:
            self._cleanup_detector(hook_id)
            if verbose:
                logger.info(f"Completed save_activations_dataset: run={run_name}, batches_saved={batch_counter}")

        return run_name
