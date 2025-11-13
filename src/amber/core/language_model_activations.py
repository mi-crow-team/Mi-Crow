from typing import TYPE_CHECKING, Sequence, Dict, Any, List
import datetime

from amber.adapters import BaseDataset
from amber.adapters.text_snippet_dataset import TextSnippetDataset
from amber.hooks import HookType
from amber.utils import get_logger
from amber.hooks.activation_saver import LayerActivationDetector
from amber.store.store import Store

import torch

if TYPE_CHECKING:
    from amber.core.language_model_context import LanguageModelContext
    from amber.core.language_model import LanguageModel


class LanguageModelActivations:
    def __init__(self, context: "LanguageModelContext"):
        self.context = context

    def _setup_detector(
            self,
            layer_signature: str | int,
            hook_id_suffix: str
    ) -> tuple[LayerActivationDetector, str]:
        """
        Create and register an activation detector.
        
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
        """Unregister a detector hook."""
        try:
            self.context.language_model.layers.unregister_hook(hook_id)
        except Exception:
            pass

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

        # Prepare layer signature info
        if isinstance(layer_signatures, (str, int)):
            layer_sig_str = str(layer_signatures)
            layer_sig_list = [layer_sig_str]
        elif isinstance(layer_signatures, list):
            layer_sig_list = [str(sig) for sig in layer_signatures]
            layer_sig_str = layer_sig_list[0] if len(layer_sig_list) == 1 else None
        else:
            layer_sig_str = None
            layer_sig_list = []

        # Prepare dataset info
        dataset_info = {}
        if dataset is not None:
            try:
                ds_id = str(getattr(dataset, "cache_dir", ""))
                ds_len = int(len(dataset))
                dataset_info = {
                    "cache_dir": ds_id,
                    "length": ds_len,
                }
            except Exception:
                dataset_info = {
                    "cache_dir": "",
                    "length": -1,
                }

        # Build metadata
        meta: Dict[str, Any] = {
            "run_name": run_name,
            "model": getattr(self.context.model, "model_name", self.context.model.__class__.__name__),
            "options": options.copy(),
        }

        if layer_sig_str is not None:
            meta["layer_signature"] = layer_sig_str
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
            store.put_run_meta(run_name, meta)
        except Exception:
            # Non-fatal if metadata cannot be stored
            if verbose:
                logger.warning(f"Failed to save run metadata for {run_name}")

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
        model: LanguageModel | None = self.context.model
        if model is None:
            raise ValueError("Model must be initialized before running")

        store = self.context.store
        device = self.context.device
        device_type = str(getattr(device, 'type', 'cpu'))

        logger = get_logger(__name__)

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
                    if not texts:
                        continue

                    tok_kwargs = {}
                    if max_length is not None:
                        tok_kwargs["max_length"] = max_length

                    self.context.language_model._inference(
                        texts,
                        tok_kwargs=tok_kwargs,
                        autocast=autocast,
                        autocast_dtype=autocast_dtype,
                    )

                    # Apply dtype conversion if specified
                    if dtype is not None:
                        detectors = self.context.language_model.layers.get_detectors()
                        for detector in detectors:
                            if "activations" in detector.tensor_metadata:
                                detector.tensor_metadata["activations"] = detector.tensor_metadata["activations"].to(dtype)

                    self.context.language_model.save_detector_metadata(
                        run_name,
                        batch_index
                    )

                    if verbose:
                        logger.info(f"Saved batch {batch_index} for run={run_name}")
                    batch_counter += 1

                    if device_type == "cuda" and free_cuda_cache_every and free_cuda_cache_every > 0:
                        if (batch_counter % free_cuda_cache_every) == 0:
                            torch.cuda.empty_cache()
                            if verbose:
                                logger.info("Emptied CUDA cache")
        finally:
            self._cleanup_detector(hook_id)
            if verbose:
                logger.info(f"Completed save_activations_dataset: run={run_name}, batches_saved={batch_counter}")
