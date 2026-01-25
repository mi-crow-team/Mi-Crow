import datetime
import gc
from typing import TYPE_CHECKING, Any, Dict, Sequence

import torch
from torch import nn

from mi_crow.datasets import BaseDataset
from mi_crow.hooks import HookType
from mi_crow.hooks.implementations.layer_activation_detector import LayerActivationDetector
from mi_crow.hooks.implementations.model_input_detector import ModelInputDetector
from mi_crow.store.store import Store
from mi_crow.utils import get_logger

if TYPE_CHECKING:
    from mi_crow.language_model.context import LanguageModelContext

logger = get_logger(__name__)


class LanguageModelActivations:
    """Handles activation saving and processing for LanguageModel."""

    def __init__(self, context: "LanguageModelContext"):  # noqa: F821
        """
        Initialize LanguageModelActivations.

        Args:
            context: LanguageModelContext instance
        """
        self.context = context

    def _setup_detector(
        self, layer_signature: str | int, hook_id_suffix: str, dtype: torch.dtype | None = None
    ) -> tuple[LayerActivationDetector, str]:
        """
        Create and register an activation detector.

        Args:
            layer_signature: Layer to attach detector to
            hook_id_suffix: Suffix for hook ID
            dtype: Optional dtype for activations

        Returns:
            Tuple of (detector, hook_id)
        """
        detector = LayerActivationDetector(
            layer_signature=layer_signature,
            hook_id=f"detector_{hook_id_suffix}",
            target_dtype=dtype,
        )

        hook_id = self.context.language_model.layers.register_hook(layer_signature, detector, HookType.FORWARD)

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

    def _setup_attention_mask_detector(self, run_name: str) -> tuple[ModelInputDetector, str]:
        """
        Create and register an attention mask detector.

        Args:
            run_name: Run name for hook ID

        Returns:
            Tuple of (detector, hook_id)
        """
        attention_mask_layer_sig = "attention_masks"
        root_model = self.context.model

        if attention_mask_layer_sig not in self.context.language_model.layers.name_to_layer:
            self.context.language_model.layers.name_to_layer[attention_mask_layer_sig] = root_model

        detector = ModelInputDetector(
            layer_signature=attention_mask_layer_sig,
            hook_id=f"attention_mask_detector_{run_name}",
            save_input_ids=False,
            save_attention_mask=True,
        )

        hook_id = self.context.language_model.layers.register_hook(
            attention_mask_layer_sig, detector, HookType.PRE_FORWARD
        )

        return detector, hook_id

    def _setup_activation_hooks(
        self,
        layer_sig_list: list[str],
        run_name: str,
        save_attention_mask: bool,
        dtype: torch.dtype | None = None,
    ) -> tuple[list[str], str | None]:
        """
        Setup activation hooks for saving.

        Args:
            layer_sig_list: List of layer signatures to hook
            run_name: Run name for hook IDs
            save_attention_mask: Whether to setup attention mask detector
            dtype: Optional dtype for activations

        Returns:
            Tuple of (hook_ids list, attention_mask_hook_id or None)
        """
        hook_ids: list[str] = []
        for sig in layer_sig_list:
            _, hook_id = self._setup_detector(sig, f"save_{run_name}_{sig}", dtype=dtype)
            hook_ids.append(hook_id)

        attention_mask_hook_id: str | None = None
        if save_attention_mask:
            _, attention_mask_hook_id = self._setup_attention_mask_detector(run_name)

        return hook_ids, attention_mask_hook_id

    def _teardown_activation_hooks(
        self,
        hook_ids: list[str],
        attention_mask_hook_id: str | None,
    ) -> None:
        """
        Teardown activation hooks.

        Args:
            hook_ids: List of hook IDs to cleanup
            attention_mask_hook_id: Optional attention mask hook ID to cleanup
        """
        for hook_id in hook_ids:
            self._cleanup_detector(hook_id)
        if attention_mask_hook_id is not None:
            self._cleanup_detector(attention_mask_hook_id)

    def _validate_save_prerequisites(self) -> tuple[nn.Module, Store]:
        """
        Validate prerequisites for saving activations.

        Returns:
            Tuple of (model, store)

        Raises:
            ValueError: If model or store is not initialized
        """
        model: nn.Module | None = self.context.model
        if model is None:
            raise ValueError("Model must be initialized before running")

        store = self.context.store
        if store is None:
            raise ValueError("Store must be provided or set on the language model")

        return model, store

    def _prepare_save_metadata(
        self,
        layer_signature: str | int | list[str | int],
        dataset: BaseDataset | None,
        run_name: str | None,
        options: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any], list[str]]:
        """
        Prepare metadata for activation saving.

        Args:
            layer_signature: Layer signature(s) to save
            dataset: Optional dataset
            run_name: Optional run name
            options: Options dictionary

        Returns:
            Tuple of (run_name, metadata, layer_sig_list)
        """
        _, layer_sig_list = self._normalize_layer_signatures(layer_signature)
        run_name, meta = self._prepare_run_metadata(
            layer_signature, dataset=dataset, run_name=run_name, options=options
        )
        return run_name, meta, layer_sig_list

    def _normalize_layer_signatures(
        self, layer_signatures: str | int | list[str | int] | None
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

        if isinstance(layer_signatures, (str, int)):
            layer_sig_list = [str(layer_signatures)]
        elif isinstance(layer_signatures, list):
            layer_sig_list = [str(sig) for sig in layer_signatures]
        else:
            layer_sig_list = []

        run_name_base, meta = self.context.language_model.inference._prepare_run_metadata(
            dataset=dataset, run_name=run_name, options=options
        )

        if layer_sig_list:
            meta["layer_signatures"] = layer_sig_list
            meta["num_layers"] = len(layer_sig_list)

        return run_name_base, meta

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
        from mi_crow.language_model.inference import InferenceEngine

        InferenceEngine._save_run_metadata(store, run_name, meta, verbose)

    def _process_batch(
        self,
        texts: Sequence[str],
        run_name: str,
        batch_index: int,
        max_length: int | None,
        autocast: bool,
        autocast_dtype: torch.dtype | None,
        dtype: torch.dtype | None,
        verbose: bool,
        save_in_batches: bool = True,
        stop_after_layer: str | int | None = None,
    ) -> None:
        """Process a single batch of texts.

        Args:
            texts: Sequence of text strings
            run_name: Run name
            batch_index: Batch index
            max_length: Optional max length for tokenization
            autocast: Whether to use autocast
            autocast_dtype: Optional dtype for autocast
            dtype: Optional dtype to convert activations to
            verbose: Whether to log progress
            stop_after_layer: Optional layer signature to stop after (name or index)
        """
        if not texts:
            return

        tok_kwargs = {}
        if max_length is not None:
            tok_kwargs["max_length"] = max_length

        self.context.language_model.inference.execute_inference(
            texts,
            tok_kwargs=tok_kwargs,
            autocast=autocast,
            autocast_dtype=autocast_dtype,
            stop_after_layer=stop_after_layer,
        )

        self.context.language_model.save_detector_metadata(
            run_name,
            batch_index,
            unified=not save_in_batches,
        )

        # Synchronize CUDA to ensure async CPU transfers from detector hooks complete
        # Only synchronize if CUDA is actually available and initialized
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except (AssertionError, RuntimeError):
            # CUDA not available or not initialized (e.g., in test environment)
            pass

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except (AssertionError, RuntimeError):
                # CUDA not available or not initialized
                pass

        if verbose:
            logger.info(f"Saved batch {batch_index} for run={run_name}")

    def _convert_activations_to_dtype(self, dtype: torch.dtype) -> None:
        """
        Convert all captured activations in detectors to the specified dtype.

        Args:
            dtype: Target dtype to convert activations to
        """
        detectors = self.context.language_model.layers.get_detectors()
        for detector in detectors:
            if hasattr(detector, "tensor_metadata") and "activations" in detector.tensor_metadata:
                tensor = detector.tensor_metadata["activations"]
                if tensor.dtype != dtype:
                    detector.tensor_metadata["activations"] = tensor.to(dtype)

    def _manage_cuda_cache(
        self, batch_counter: int, free_cuda_cache_every: int | None, device_type: str, verbose: bool
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
        layer_signature: str | int | list[str | int],
        run_name: str | None = None,
        batch_size: int = 32,
        *,
        dtype: torch.dtype | None = None,
        max_length: int | None = None,
        autocast: bool = True,
        autocast_dtype: torch.dtype | None = None,
        free_cuda_cache_every: int | None = None,
        verbose: bool = False,
        save_in_batches: bool = True,
        save_attention_mask: bool = False,
        stop_after_last_layer: bool = True,
    ) -> str:
        """
        Save activations from a dataset.

        Args:
            dataset: Dataset to process
            layer_signature: Layer signature (or list of signatures) to capture activations from
            run_name: Optional run name (generated if None)
            batch_size: Batch size for processing
            dtype: Optional dtype to convert activations to
            max_length: Optional max length for tokenization
            autocast: Whether to use autocast
            autocast_dtype: Optional dtype for autocast
            free_cuda_cache_every: Clear CUDA cache every N batches (None to auto-detect, 0 to disable)
            verbose: Whether to log progress
            save_attention_mask: Whether to also save attention masks (automatically attaches ModelInputDetector)
            stop_after_last_layer: Whether to stop model forward pass after the last requested layer
                to save memory and time. Defaults to True.

        Returns:
            Run name used for saving

        Raises:
            ValueError: If model or store is not initialized
        """
        model, store = self._validate_save_prerequisites()

        device = torch.device(self.context.device)
        device_type = str(device.type)

        if free_cuda_cache_every is None:
            free_cuda_cache_every = 5 if device_type == "cuda" else 0

        options = {
            "dtype": str(dtype) if dtype is not None else None,
            "max_length": max_length,
            "batch_size": int(batch_size),
            "stop_after_last_layer": stop_after_last_layer,
        }

        run_name, meta, layer_sig_list = self._prepare_save_metadata(layer_signature, dataset, run_name, options)

        if verbose:
            logger.info(
                f"Starting save_activations_dataset: run={run_name}, layers={layer_sig_list}, "
                f"batch_size={batch_size}, device={device_type}"
            )

        self._save_run_metadata(store, run_name, meta, verbose)

        hook_ids, attention_mask_hook_id = self._setup_activation_hooks(
            layer_sig_list, run_name, save_attention_mask, dtype=dtype
        )

        batch_counter = 0
        # Stop after last hooked layer if requested
        stop_after = layer_sig_list[-1] if (layer_sig_list and stop_after_last_layer) else None

        try:
            with torch.inference_mode():
                for batch_index, batch in enumerate(dataset.iter_batches(batch_size)):
                    texts = dataset.extract_texts_from_batch(batch)
                    self._process_batch(
                        texts,
                        run_name,
                        batch_index,
                        max_length,
                        autocast,
                        autocast_dtype,
                        dtype,
                        verbose,
                        save_in_batches=save_in_batches,
                        stop_after_layer=stop_after,
                    )
                    batch_counter += 1

                    self._manage_cuda_cache(batch_counter, free_cuda_cache_every, device_type, verbose)
        finally:
            self._teardown_activation_hooks(hook_ids, attention_mask_hook_id)
            if verbose:
                logger.info(f"Completed save_activations_dataset: run={run_name}, batches_saved={batch_counter}")

        return run_name

    def save_activations(
        self,
        texts: Sequence[str],
        layer_signature: str | int | list[str | int],
        run_name: str | None = None,
        batch_size: int | None = None,
        *,
        dtype: torch.dtype | None = None,
        max_length: int | None = None,
        autocast: bool = True,
        autocast_dtype: torch.dtype | None = None,
        free_cuda_cache_every: int | None = 0,
        verbose: bool = False,
        save_in_batches: bool = True,
        save_attention_mask: bool = False,
        stop_after_last_layer: bool = True,
    ) -> str:
        """
        Save activations from a list of texts.

        Args:
            texts: Sequence of text strings to process
            layer_signature: Layer signature (or list of signatures) to capture activations from
            run_name: Optional run name (generated if None)
            batch_size: Optional batch size for processing (if None, processes all at once)
            dtype: Optional dtype to convert activations to
            max_length: Optional max length for tokenization
            autocast: Whether to use autocast
            autocast_dtype: Optional dtype for autocast
            free_cuda_cache_every: Clear CUDA cache every N batches (0 or None to disable)
            verbose: Whether to log progress
            save_attention_mask: Whether to also save attention masks (automatically attaches ModelInputDetector)
            stop_after_last_layer: Whether to stop model forward pass after the last requested layer
                to save memory and time. Defaults to True.

        Returns:
            Run name used for saving

        Raises:
            ValueError: If model or store is not initialized
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        model, store = self._validate_save_prerequisites()

        device = torch.device(self.context.device)
        device_type = str(device.type)

        if batch_size is None:
            batch_size = len(texts)

        options = {
            "dtype": str(dtype) if dtype is not None else None,
            "max_length": max_length,
            "batch_size": int(batch_size),
            "stop_after_last_layer": stop_after_last_layer,
        }

        run_name, meta, layer_sig_list = self._prepare_save_metadata(layer_signature, None, run_name, options)

        if verbose:
            logger.info(
                f"Starting save_activations: run={run_name}, layers={layer_sig_list}, "
                f"batch_size={batch_size}, device={device_type}"
            )

        self._save_run_metadata(store, run_name, meta, verbose)

        hook_ids, attention_mask_hook_id = self._setup_activation_hooks(
            layer_sig_list, run_name, save_attention_mask, dtype=dtype
        )

        batch_counter = 0
        # Stop after last hooked layer if requested
        stop_after = layer_sig_list[-1] if (layer_sig_list and stop_after_last_layer) else None

        try:
            with torch.inference_mode():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    batch_index = i // batch_size

                    self._process_batch(
                        batch_texts,
                        run_name,
                        batch_index,
                        max_length,
                        autocast,
                        autocast_dtype,
                        dtype,
                        verbose,
                        save_in_batches=save_in_batches,
                        stop_after_layer=stop_after,
                    )
                    batch_counter += 1
                    self._manage_cuda_cache(batch_counter, free_cuda_cache_every, device_type, verbose)
        finally:
            self._teardown_activation_hooks(hook_ids, attention_mask_hook_id)
            if verbose:
                logger.info(f"Completed save_activations: run={run_name}, batches_saved={batch_counter}")

        return run_name
