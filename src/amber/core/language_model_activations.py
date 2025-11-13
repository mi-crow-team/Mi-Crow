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

    def _process_activation_tensor(
            self,
            act: torch.Tensor,
            inp_ids: torch.Tensor | None,
            dtype: torch.dtype | None,
            device_type: str,
    ) -> torch.Tensor:
        """
        Process activation tensor: reshape, convert dtype, move to CPU.
        
        Args:
            act: Activation tensor
            inp_ids: Optional input_ids tensor for reshaping
            dtype: Optional target dtype
            device_type: Device type string ('cuda' or 'cpu')
            
        Returns:
            Processed activation tensor
        """
        # Try to reshape from [N, D] to [B, T, D] if possible
        if inp_ids is not None:
            try:
                if act.dim() == 2 and inp_ids.dim() == 2:
                    B, T = int(inp_ids.shape[0]), int(inp_ids.shape[1])
                    N, D = int(act.shape[0]), int(act.shape[1])
                    if B * T == N:
                        act = act.view(B, T, D)
            except Exception:
                # Best-effort reshape; continue without raising
                pass

        # Convert dtype if specified
        if dtype is not None:
            try:
                act = act.to(dtype, copy=False)
            except Exception:
                act = act.to(dtype)

        # Move to CPU
        if device_type == "cuda":
            act = act.to("cpu", non_blocking=True)
        else:
            act = act.to("cpu")

        return act

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

    def _save_detector_metadata_and_cleanup(
            self,
            store: Store,
            run_name: str,
            hook_ids: str | List[str],
            verbose: bool = False,
    ) -> None:
        """
        Save detector metadata and cleanup hooks.
        
        Args:
            store: Store to save to
            run_name: Run name
            hook_ids: Single hook ID or list of hook IDs to cleanup
            verbose: Whether to log
        """
        logger = get_logger(__name__)

        # Save detector metadata
        try:
            detector_meta_path = f"{run_name}_detector_metadata"
            self.context.language_model.save_detector_metadata(detector_meta_path, store=store)
            if verbose:
                logger.info(f"Saved detector metadata to {detector_meta_path}")
        except Exception:
            # Non-fatal if detector metadata cannot be stored
            if verbose:
                logger.warning("Failed to save detector metadata")

        # Cleanup hooks
        if isinstance(hook_ids, str):
            hook_ids = [hook_ids]
        for hook_id in hook_ids:
            self._cleanup_detector(hook_id)

    def _process_batch_activations(
            self,
            detectors: Dict[str | int, LayerActivationDetector] | LayerActivationDetector,
            layer_signatures: str | int | list[str | int] | None,
            payload: Dict[str, torch.Tensor],
            inp_ids: torch.Tensor | None,
            dtype: torch.dtype | None,
            device_type: str,
            save_inputs: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        Process activations from detectors and add to payload.
        
        Args:
            detectors: Single detector or dict of layer_sig -> detector
            layer_signatures: Layer signature(s) - used for naming
            payload: Payload dictionary to add activations to
            inp_ids: Optional input_ids for reshaping
            dtype: Optional target dtype
            device_type: Device type string
            save_inputs: Whether inputs are being saved
            
        Returns:
            Updated payload dictionary
        """
        # Handle single detector case
        if isinstance(detectors, LayerActivationDetector):
            detector = detectors
            act = detector.get_captured()
            if act is not None:
                act = self._process_activation_tensor(act, inp_ids, dtype, device_type)
                payload["activations"] = act
                detector.clear_captured()
        else:
            # Handle multiple detectors
            for layer_sig, detector in detectors.items():
                act = detector.get_captured()
                if act is not None:
                    act = self._process_activation_tensor(act, inp_ids, dtype, device_type)
                    # Store with layer prefix
                    safe_layer_name = str(layer_sig).replace("/", "_")
                    payload[f"activations_{safe_layer_name}"] = act
                    detector.clear_captured()

        return payload

    def capture_activations(
            self,
            texts: Sequence[str],
            layer_signature: str | int,
            *,
            autocast: bool = True,
            autocast_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Capture activations from a single layer for a batch of texts.
        
        Args:
            texts: List of text strings to process
            layer_signature: Layer name or index to capture from
            autocast: Whether to use automatic mixed precision
            autocast_dtype: Data type for autocast (default: float16 on CUDA)
            
        Returns:
            Activation tensor from the specified layer
        """
        # Setup detector
        detector, hook_id = self._setup_detector(layer_signature, f"capture_{id(texts)}")

        try:
            # Run inference
            _ = self.context.language_model._inference(
                texts,
                autocast=autocast,
                autocast_dtype=autocast_dtype,
            )

            # Get captured activations
            activations = detector.get_captured()
            if activations is None:
                raise RuntimeError(f"Failed to capture activations from layer '{layer_signature}'")

            return activations

        finally:
            self._cleanup_detector(hook_id)

    def capture_activations_all_layers(
            self,
            texts: Sequence[str],
            layer_signatures: list[str | int] | None = None,
            *,
            autocast: bool = True,
            autocast_dtype: torch.dtype | None = None,
    ) -> Dict[str | int, torch.Tensor]:
        """
        Capture activations from multiple layers for a batch of texts.
        
        Args:
            texts: List of text strings to process
            layer_signatures: List of layer names/indices. If None, captures from all layers
            autocast: Whether to use automatic mixed precision
            autocast_dtype: Data type for autocast
            
        Returns:
            Dictionary mapping layer signatures to activation tensors
        """
        # If no layer signatures provided, get all layers
        if layer_signatures is None:
            layer_signatures = self.context.language_model.layers.get_layer_names()

        # Setup detectors for each layer
        detectors = {}
        hook_ids = []

        for layer_sig in layer_signatures:
            detector, hook_id = self._setup_detector(layer_sig, f"all_{layer_sig}_{id(texts)}")
            detectors[layer_sig] = detector
            hook_ids.append(hook_id)

        try:
            # Run inference
            _ = self.context.language_model._inference(
                texts,
                autocast=autocast,
                autocast_dtype=autocast_dtype,
            )

            # Collect activations from all detectors
            all_activations = {}
            for layer_sig, detector in detectors.items():
                activations = detector.get_captured()
                if activations is not None:
                    all_activations[layer_sig] = activations

            return all_activations

        finally:
            # Clean up all hooks
            for hook_id in hook_ids:
                self._cleanup_detector(hook_id)

    def save_activations_per_sequence(
            self,
            texts: Sequence[str],
            layer_signature: str | int,
            run_name: str | None = None,
            store: Store | None = None,
            *,
            dtype: torch.dtype | None = None,
            max_length: int | None = None,
            autocast: bool = True,
            autocast_dtype: torch.dtype | None = None,
            save_inputs: bool = True,
            verbose: bool = False,
    ) -> str:
        """
        Save activations per sequence (one string = one batch) for a single layer.
        
        Args:
            texts: Sequence of text strings to process
            layer_signature: Layer name or index to capture from
            run_name: Optional run name (auto-generated if not provided)
            store: Store to save to (uses model's store if not provided)
            dtype: Data type to save activations as
            max_length: Maximum sequence length
            autocast: Whether to use automatic mixed precision
            autocast_dtype: Data type for autocast
            save_inputs: Whether to save input_ids and attention_mask
            verbose: Print progress information
            
        Returns:
            The run_name used
        """
        model = self.context.model
        if model is None:
            raise ValueError("Model must be initialized before running")

        if store is None:
            store = self.context.store

        device = next(model.parameters()).device  # type: ignore[attr-defined]
        device_type = str(getattr(device, 'type', 'cpu'))

        logger = get_logger(__name__)

        # Prepare metadata
        options = {
            "dtype": str(dtype) if dtype is not None else None,
            "max_length": max_length,
            "save_inputs": bool(save_inputs),
        }
        run_name, meta = self._prepare_run_metadata(
            layer_signature, dataset=None, run_name=run_name, options=options
        )

        if verbose:
            logger.info(
                f"Starting save_activations_per_sequence: run={run_name}, layer={layer_signature}, "
                f"sequences={len(texts)}, device={device_type}"
            )

        # Save run metadata
        self._save_run_metadata(store, run_name, meta, verbose)

        # Setup detector
        detector, hook_id = self._setup_detector(layer_signature, f"save_{run_name}")

        try:
            with torch.inference_mode():
                for batch_index, text in enumerate(texts):
                    if not text:
                        continue

                    tok_kwargs = {}
                    if max_length is not None:
                        tok_kwargs["max_length"] = max_length

                    payload: Dict[str, torch.Tensor] = {}
                    res = self.context.language_model._inference(
                        [text],
                        tok_kwargs=tok_kwargs,
                        autocast=autocast,
                        autocast_dtype=autocast_dtype,
                    )
                    output, enc = res

                    if save_inputs:
                        inp_ids = enc.get("input_ids")
                        attn = enc.get("attention_mask")
                        if isinstance(inp_ids, torch.Tensor):
                            payload["input_ids"] = inp_ids
                        if isinstance(attn, torch.Tensor):
                            payload["attention_mask"] = attn
                    else:
                        inp_ids = enc.get("input_ids") if isinstance(enc.get("input_ids"), torch.Tensor) else None

                    # Process activations
                    payload = self._process_batch_activations(
                        detector, layer_signature, payload, inp_ids, dtype, device_type, save_inputs
                    )

                    store.put_run_batch(run_name, batch_index, payload)
                    if verbose:
                        logger.info(f"Saved batch {batch_index} for run={run_name} with keys={list(payload.keys())}")
                    del payload
        finally:
            # Save detector metadata and cleanup
            self._save_detector_metadata_and_cleanup(store, run_name, hook_id, verbose)
            if verbose:
                logger.info(f"Completed save_activations_per_sequence: run={run_name}, sequences_saved={len(texts)}")

        return run_name

    def save_activations_per_sequence_all_layers(
            self,
            texts: Sequence[str],
            layer_signatures: list[str | int] | None = None,
            run_name: str | None = None,
            store: Store | None = None,
            *,
            dtype: torch.dtype | None = None,
            max_length: int | None = None,
            autocast: bool = True,
            autocast_dtype: torch.dtype | None = None,
            save_inputs: bool = True,
            verbose: bool = False,
    ) -> str:
        """
        Save activations per sequence (one string = one batch) for all layers.
        
        Args:
            texts: Sequence of text strings to process
            layer_signatures: List of layer names/indices. If None, saves from all layers
            run_name: Optional run name (auto-generated if not provided)
            store: Store to save to (uses model's store if not provided)
            dtype: Data type to save activations as
            max_length: Maximum sequence length
            autocast: Whether to use automatic mixed precision
            autocast_dtype: Data type for autocast
            save_inputs: Whether to save input_ids and attention_mask
            verbose: Print progress information
            
        Returns:
            The run_name used
        """
        model = self.context.model
        if model is None:
            raise ValueError("Model must be initialized before running")

        if store is None:
            store = self.context.store

        # If no layer signatures provided, get all layers
        if layer_signatures is None:
            layer_signatures = self.context.language_model.layers.get_layer_names()

        device = next(model.parameters()).device  # type: ignore[attr-defined]
        device_type = str(getattr(device, 'type', 'cpu'))

        logger = get_logger(__name__)

        # Prepare metadata
        options = {
            "dtype": str(dtype) if dtype is not None else None,
            "max_length": max_length,
            "save_inputs": bool(save_inputs),
        }
        run_name, meta = self._prepare_run_metadata(
            layer_signatures, dataset=None, run_name=run_name, options=options
        )

        if verbose:
            logger.info(
                f"Starting save_activations_per_sequence_all_layers: run={run_name}, "
                f"layers={len(layer_signatures)}, sequences={len(texts)}, device={device_type}"
            )

        # Save run metadata
        self._save_run_metadata(store, run_name, meta, verbose)

        # Setup detectors for all layers
        detectors = {}
        hook_ids = []

        for layer_sig in layer_signatures:
            detector, hook_id = self._setup_detector(layer_sig, f"save_all_{layer_sig}_{run_name}")
            detectors[layer_sig] = detector
            hook_ids.append(hook_id)

        try:
            with torch.inference_mode():
                for batch_index, text in enumerate(texts):
                    if not text:
                        continue

                    tok_kwargs = {}
                    if max_length is not None:
                        tok_kwargs["max_length"] = max_length

                    payload: Dict[str, torch.Tensor] = {}
                    res = self.context.language_model._inference(
                        [text],
                        tok_kwargs=tok_kwargs,
                        autocast=autocast,
                        autocast_dtype=autocast_dtype,
                    )
                    output, enc = res

                    if save_inputs:
                        inp_ids = enc.get("input_ids")
                        attn = enc.get("attention_mask")
                        if isinstance(inp_ids, torch.Tensor):
                            payload["input_ids"] = inp_ids
                        if isinstance(attn, torch.Tensor):
                            payload["attention_mask"] = attn
                    else:
                        inp_ids = enc.get("input_ids") if isinstance(enc.get("input_ids"), torch.Tensor) else None

                    # Process activations from all layers
                    payload = self._process_batch_activations(
                        detectors, layer_signatures, payload, inp_ids, dtype, device_type, save_inputs
                    )

                    store.put_run_batch(run_name, batch_index, payload)
                    if verbose:
                        num_layers = len([k for k in payload.keys() if k.startswith('activations_')])
                        logger.info(
                            f"Saved batch {batch_index} for run={run_name} with {num_layers} layers"
                        )
                    del payload
        finally:
            # Save detector metadata and cleanup
            self._save_detector_metadata_and_cleanup(store, run_name, hook_ids, verbose)
            if verbose:
                logger.info(
                    f"Completed save_activations_per_sequence_all_layers: run={run_name}, "
                    f"sequences_saved={len(texts)}, layers={len(layer_signatures)}"
                )

        return run_name

    def save_activations_dataset(
            self,
            dataset: BaseDataset,
            layer_signature: str | int,
            run_name: str | None = None,
            store: Store | None = None,
            batch_size: int = 32,
            *,
            dtype: torch.dtype | None = None,
            max_length: int | None = None,
            autocast: bool = True,
            autocast_dtype: torch.dtype | None = None,
            save_inputs: bool = True,
            free_cuda_cache_every: int | None = 0,
            verbose: bool = False,
    ) -> str:
        """
        Save activations from a dataset for a single layer.
        
        Args:
            dataset: Dataset to process
            layer_signature: Layer name or index to capture from
            run_name: Optional run name (auto-generated if not provided)
            store: Store to save to (uses model's store if not provided)
            batch_size: Batch size for processing
            dtype: Data type to save activations as
            max_length: Maximum sequence length
            autocast: Whether to use automatic mixed precision
            autocast_dtype: Data type for autocast
            save_inputs: Whether to save input_ids and attention_mask
            free_cuda_cache_every: Free CUDA cache every N batches (0=disabled)
            verbose: Print progress information
            
        Returns:
            The run_name used
        """
        model = self.context.model
        if model is None:
            raise ValueError("Model must be initialized before running")

        if store is None:
            store = self.context.store

        device = next(model.parameters()).device
        device_type = str(getattr(device, 'type', 'cpu'))

        logger = get_logger(__name__)

        options = {
            "dtype": str(dtype) if dtype is not None else None,
            "max_length": max_length,
            "save_inputs": bool(save_inputs),
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

        # Save run metadata
        self._save_run_metadata(store, run_name, meta, verbose)

        # Setup detector
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

                    payload: Dict[str, torch.Tensor] = {}
                    res = self.context.language_model._inference(
                        texts,
                        tok_kwargs=tok_kwargs,
                        autocast=autocast,
                        autocast_dtype=autocast_dtype,
                    )
                    output, enc = res

                    if save_inputs:
                        inp_ids = enc.get("input_ids")
                        attn = enc.get("attention_mask")
                        if isinstance(inp_ids, torch.Tensor):
                            payload["input_ids"] = inp_ids
                        if isinstance(attn, torch.Tensor):
                            payload["attention_mask"] = attn
                    else:
                        inp_ids = enc.get("input_ids") if isinstance(enc.get("input_ids"), torch.Tensor) else None

                    # Process activations
                    payload = self._process_batch_activations(
                        detector, layer_signature, payload, inp_ids, dtype, device_type, save_inputs
                    )

                    store.put_run_batch(run_name, batch_index, payload)
                    if verbose:
                        logger.info(f"Saved batch {batch_index} for run={run_name} with keys={list(payload.keys())}")
                    del payload
                    batch_counter += 1

                    if device_type == "cuda" and free_cuda_cache_every and free_cuda_cache_every > 0:
                        if (batch_counter % free_cuda_cache_every) == 0:
                            torch.cuda.empty_cache()
                            if verbose:
                                logger.info("Emptied CUDA cache")
        finally:
            # Save detector metadata and cleanup
            self._save_detector_metadata_and_cleanup(store, run_name, hook_id, verbose)
            if verbose:
                logger.info(f"Completed save_activations_dataset: run={run_name}, batches_saved={batch_counter}")

        return run_name

    def save_activations_dataset_all_layers(
            self,
            dataset: TextSnippetDataset,
            layer_signatures: list[str | int] | None = None,
            run_name: str | None = None,
            store: Store | None = None,
            batch_size: int = 32,
            *,
            dtype: torch.dtype | None = None,
            max_length: int | None = None,
            autocast: bool = True,
            autocast_dtype: torch.dtype | None = None,
            save_inputs: bool = True,
            free_cuda_cache_every: int | None = 0,
            verbose: bool = False,
    ) -> str:
        """
        Save activations from a dataset for all layers.
        
        Args:
            dataset: Dataset to process
            layer_signatures: List of layer names/indices. If None, saves from all layers
            run_name: Optional run name (auto-generated if not provided)
            store: Store to save to (uses model's store if not provided)
            batch_size: Batch size for processing
            dtype: Data type to save activations as
            max_length: Maximum sequence length
            autocast: Whether to use automatic mixed precision
            autocast_dtype: Data type for autocast
            save_inputs: Whether to save input_ids and attention_mask
            free_cuda_cache_every: Free CUDA cache every N batches (0=disabled)
            verbose: Print progress information
            
        Returns:
            The run_name used
        """
        model = self.context.model
        if model is None:
            raise ValueError("Model must be initialized before running")

        if store is None:
            store = self.context.store

        # If no layer signatures provided, get all layers
        if layer_signatures is None:
            layer_signatures = self.context.language_model.layers.get_layer_names()

        device = next(model.parameters()).device  # type: ignore[attr-defined]
        device_type = str(getattr(device, 'type', 'cpu'))

        logger = get_logger(__name__)

        # Prepare metadata
        options = {
            "dtype": str(dtype) if dtype is not None else None,
            "max_length": max_length,
            "save_inputs": bool(save_inputs),
            "batch_size": int(batch_size),
        }
        run_name, meta = self._prepare_run_metadata(
            layer_signatures, dataset=dataset, run_name=run_name, options=options
        )

        if verbose:
            logger.info(
                f"Starting save_activations_dataset_all_layers: run={run_name}, "
                f"layers={len(layer_signatures)}, batch_size={batch_size}, device={device_type}"
            )

        # Save run metadata
        self._save_run_metadata(store, run_name, meta, verbose)

        # Setup detectors for all layers
        detectors = {}
        hook_ids = []

        for layer_sig in layer_signatures:
            detector, hook_id = self._setup_detector(layer_sig, f"save_all_{layer_sig}_{run_name}")
            detectors[layer_sig] = detector
            hook_ids.append(hook_id)

        batch_counter = 0
        try:
            with torch.inference_mode():
                for batch_index, texts in enumerate(dataset.iter_batches(batch_size)):
                    if not texts:
                        continue

                    tok_kwargs = {}
                    if max_length is not None:
                        tok_kwargs["max_length"] = max_length

                    payload: Dict[str, torch.Tensor] = {}
                    res = self.context.language_model._inference(
                        texts,
                        tok_kwargs=tok_kwargs,
                        autocast=autocast,
                        autocast_dtype=autocast_dtype,
                    )
                    output, enc = res

                    if save_inputs:
                        inp_ids = enc.get("input_ids")
                        attn = enc.get("attention_mask")
                        if isinstance(inp_ids, torch.Tensor):
                            payload["input_ids"] = inp_ids
                        if isinstance(attn, torch.Tensor):
                            payload["attention_mask"] = attn
                    else:
                        inp_ids = enc.get("input_ids") if isinstance(enc.get("input_ids"), torch.Tensor) else None

                    # Process activations from all layers
                    payload = self._process_batch_activations(
                        detectors, layer_signatures, payload, inp_ids, dtype, device_type, save_inputs
                    )

                    store.put_run_batch(run_name, batch_index, payload)
                    if verbose:
                        num_layers = len([k for k in payload.keys() if k.startswith('activations_')])
                        logger.info(
                            f"Saved batch {batch_index} for run={run_name} with {num_layers} layers"
                        )
                    del payload
                    batch_counter += 1

                    if device_type == "cuda" and free_cuda_cache_every and free_cuda_cache_every > 0:
                        if (batch_counter % free_cuda_cache_every) == 0:
                            torch.cuda.empty_cache()
                            if verbose:
                                logger.info("Emptied CUDA cache")
        finally:
            # Save detector metadata and cleanup
            self._save_detector_metadata_and_cleanup(store, run_name, hook_ids, verbose)
            if verbose:
                logger.info(
                    f"Completed save_activations_dataset_all_layers: run={run_name}, "
                    f"batches_saved={batch_counter}, layers={len(layer_signatures)}"
                )

        return run_name
