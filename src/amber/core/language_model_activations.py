from typing import TYPE_CHECKING, Sequence, Dict

from amber.adapters.text_snippet_dataset import TextSnippetDataset
from amber.store import Store
from amber.utils import get_logger
from amber.hooks.activation_saver import ActivationSaverDetector

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
    ) -> tuple[ActivationSaverDetector, str]:
        """
        Create and register an activation detector.
        
        Returns:
            Tuple of (detector, hook_id)
        """
        detector = ActivationSaverDetector(
            layer_signature=layer_signature,
            hook_id=f"detector_{hook_id_suffix}"
        )

        hook_id = self.context.language_model.layers.register_hook(
            layer_signature,
            detector
        )

        return detector, hook_id

    def _cleanup_detector(self, hook_id: str) -> None:
        """Unregister a detector hook."""
        try:
            self.context.language_model.layers.unregister_hook(hook_id)
        except Exception:
            pass

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

    def infer_and_save(
            self,
            dataset: TextSnippetDataset,
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
    ):

        model = self.context.model
        if model is None:
            raise ValueError("Model must be initialized before running")

        if store is None:
            store = self.context.store

        if run_name is None:
            import datetime
            run_name = f"activations_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        device = next(model.parameters()).device  # type: ignore[attr-defined]
        device_type = str(getattr(device, 'type', 'cpu'))

        logger = get_logger(__name__)
        if verbose:
            logger.info(
                f"Starting save_model_activations: run={run_name}, layer={layer_signature}, batch_size={batch_size}, device={device_type}")

        # Save run metadata (dataset, model, run info)
        try:
            ds_id = str(getattr(dataset, "cache_dir", ""))
            ds_len = int(len(dataset))
        except Exception:
            ds_id = ""
            ds_len = -1
        meta = {
            "run_name": run_name,
            "model": getattr(self.context.model, "model_name", self.context.model.__class__.__name__),
            "layer_signature": str(layer_signature),
            "dataset": {
                "cache_dir": ds_id,
                "length": ds_len,
            },
            "options": {
                "dtype": str(dtype) if dtype is not None else None,
                "max_length": max_length,
                "save_inputs": bool(save_inputs),
                "batch_size": int(batch_size),
            },
        }
        try:
            store.put_run_meta(run_name, meta)
        except Exception:
            # Non-fatal if metadata cannot be stored
            pass

        # Setup detector
        saver_detector, hook_id = self._setup_detector(layer_signature, f"save_{run_name}")
        batch_counter = 0
        try:
            with (torch.inference_mode()):
                for batch_index, texts in enumerate(dataset.iter_batches(batch_size)):
                    if not texts:
                        continue

                    tok_kwargs = {}
                    if max_length is not None:
                        tok_kwargs["max_length"] = max_length

                    payload: dict[str, torch.Tensor] = {}
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

                    # Get captured activations from detector
                    act = saver_detector.get_captured()
                    if act is not None:
                        # If the hook captured a 2D tensor [N, D] from an inner layer (e.g., flattened tokens),
                        # and we have input_ids with shape [B, T], reshape to [B, T, D] when possible.
                        try:
                            inp_ids = payload.get("input_ids")
                            if isinstance(act, torch.Tensor) and isinstance(inp_ids, torch.Tensor):
                                if act.dim() == 2 and inp_ids.dim() == 2:
                                    B, T = int(inp_ids.shape[0]), int(inp_ids.shape[1])
                                    N, D = int(act.shape[0]), int(act.shape[1])
                                    if B * T == N:
                                        act = act.view(B, T, D)
                        except Exception:
                            # Best-effort reshape; continue without raising
                            pass
                        if dtype is not None:
                            try:
                                act = act.to(dtype, copy=False)
                            except Exception:
                                act = act.to(dtype)
                        if device_type == "cuda":
                            act = act.to("cpu", non_blocking=True)
                        else:
                            act = act.to("cpu")
                        payload["activations"] = act
                        saver_detector.clear_captured()

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
            # Ensure hook is removed even if an exception occurs
            self._cleanup_detector(hook_id)
            if verbose:
                logger.info(f"Completed save_model_activations: run={run_name}, batches_saved={batch_counter}")

    def infer_and_save_all_layers(
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
    ):
        """
        Run inference on a dataset and save activations from multiple layers.
        
        Args:
            dataset: Dataset to process
            layer_signatures: List of layer names/indices. If None, saves from all layers
            run_name: Name for this run (auto-generated if not provided)
            store: Store to save to (uses model's store if not provided)
            batch_size: Batch size for processing
            dtype: Data type to save activations as
            max_length: Maximum sequence length
            autocast: Whether to use automatic mixed precision
            autocast_dtype: Data type for autocast
            save_inputs: Whether to save input_ids and attention_mask
            free_cuda_cache_every: Free CUDA cache every N batches (0=disabled)
            verbose: Print progress information
        """
        model = self.context.model
        if model is None:
            raise ValueError("Model must be initialized before running")

        if store is None:
            store = self.context.store

        if run_name is None:
            import datetime
            run_name = f"activations_all_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # If no layer signatures provided, get all layers
        if layer_signatures is None:
            layer_signatures = self.context.language_model.layers.get_layer_names()

        device = next(model.parameters()).device
        device_type = str(getattr(device, 'type', 'cpu'))

        logger = get_logger(__name__)
        if verbose:
            logger.info(
                f"Starting save_all_layers: run={run_name}, layers={len(layer_signatures)}, "
                f"batch_size={batch_size}, device={device_type}"
            )

        # Save run metadata
        try:
            ds_id = str(getattr(dataset, "cache_dir", ""))
            ds_len = int(len(dataset))
        except Exception:
            ds_id = ""
            ds_len = -1

        meta = {
            "run_name": run_name,
            "model": getattr(self.context.model, "model_name", self.context.model.__class__.__name__),
            "layer_signatures": [str(sig) for sig in layer_signatures],
            "num_layers": len(layer_signatures),
            "dataset": {
                "cache_dir": ds_id,
                "length": ds_len,
            },
            "options": {
                "dtype": str(dtype) if dtype is not None else None,
                "max_length": max_length,
                "save_inputs": bool(save_inputs),
                "batch_size": int(batch_size),
            },
        }

        try:
            store.put_run_meta(run_name, meta)
        except Exception:
            pass

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

                    payload: dict[str, torch.Tensor] = {}
                    res = self.context.language_model._inference(
                        texts,
                        tok_kwargs=tok_kwargs,
                        autocast=autocast,
                        autocast_dtype=autocast_dtype,
                    )

                    if save_inputs:
                        output, enc = res
                        inp_ids = enc.get("input_ids")
                        attn = enc.get("attention_mask")
                        if isinstance(inp_ids, torch.Tensor):
                            payload["input_ids"] = inp_ids
                        if isinstance(attn, torch.Tensor):
                            payload["attention_mask"] = attn

                    # Collect activations from all layers
                    for layer_sig, detector in detectors.items():
                        act = detector.get_captured()
                        if act is not None:
                            # Process activation
                            try:
                                inp_ids = payload.get("input_ids")
                                if isinstance(act, torch.Tensor) and isinstance(inp_ids, torch.Tensor):
                                    if act.dim() == 2 and inp_ids.dim() == 2:
                                        B, T = int(inp_ids.shape[0]), int(inp_ids.shape[1])
                                        N, D = int(act.shape[0]), int(act.shape[1])
                                        if B * T == N:
                                            act = act.view(B, T, D)
                            except Exception:
                                pass

                            if dtype is not None:
                                try:
                                    act = act.to(dtype, copy=False)
                                except Exception:
                                    act = act.to(dtype)

                            if device_type == "cuda":
                                act = act.to("cpu", non_blocking=True)
                            else:
                                act = act.to("cpu")

                            # Store with layer prefix
                            safe_layer_name = str(layer_sig).replace("/", "_")
                            payload[f"activations_{safe_layer_name}"] = act
                            detector.clear_captured()

                    store.put_run_batch(run_name, batch_index, payload)
                    if verbose:
                        logger.info(
                            f"Saved batch {batch_index} for run={run_name} "
                            f"with {len([k for k in payload.keys() if k.startswith('activations_')])} layers"
                        )
                    del payload
                    batch_counter += 1

                    if device_type == "cuda" and free_cuda_cache_every and free_cuda_cache_every > 0:
                        if (batch_counter % free_cuda_cache_every) == 0:
                            torch.cuda.empty_cache()
                            if verbose:
                                logger.info("Emptied CUDA cache")

        finally:
            # Clean up all hooks
            for hook_id in hook_ids:
                self._cleanup_detector(hook_id)

            if verbose:
                logger.info(
                    f"Completed save_all_layers: run={run_name}, batches_saved={batch_counter}, "
                    f"layers={len(layer_signatures)}"
                )
