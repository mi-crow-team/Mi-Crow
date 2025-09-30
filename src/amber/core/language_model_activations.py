from typing import TYPE_CHECKING

from amber.adapters.text_snippet_dataset import TextSnippetDataset
from amber.store import Store
from amber.utils import get_logger

import torch

if TYPE_CHECKING:
    from amber.core.language_model import LanguageModel


class LanguageModelActivations:
    def __init__(self, language_model: "LanguageModel"):
        self.lm = language_model

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
        """Run the model over the dataset and persist activations from a chosen layer.

        - Registers a forward hook on the specified layer (by name or index).
        - Iterates dataset batches of strings, tokenizes using model.tokenizer.
        - Executes a forward pass to trigger the hook and capture the layer output.
        - Saves per-batch tensors via the provided Store under {run_name}/batch_XXXXXX.safetensors.

        Saved keys per batch:
          - activations: Tensor captured from the layer hook (detached on CPU)
          - input_ids: Tokenized input IDs (CPU tensor)
          - attention_mask: Attention mask (CPU tensor), if present
        """
        model = self.lm
        if model.model is None:
            raise ValueError("Model must be initialized before running")

        if store is None:
            store = self.lm.store

        if run_name is None:
            import datetime
            run_name = f"activations_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        device = next(model.model.parameters()).device  # type: ignore[attr-defined]
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
            "model": getattr(self.lm, "model_name", self.lm.model.__class__.__name__),
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

        captured: dict[str, torch.Tensor] = {}

        def save_activations_hook(_module, _inputs, output):
            # Normalize output to a single tensor
            tensor: torch.Tensor | None = None
            if isinstance(output, torch.Tensor):
                tensor = output
            elif isinstance(output, (tuple, list)):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        tensor = item
                        break
            else:
                # Try common HF output objects
                if hasattr(output, "last_hidden_state"):
                    maybe = getattr(output, "last_hidden_state")
                    if isinstance(maybe, torch.Tensor):
                        tensor = maybe
            if tensor is None:
                # As a fallback, do nothing for this call
                return
            captured["activations"] = tensor.detach().to("cpu")

        # Register the hook on the desired layer and keep a handle for cleanup
        handle = model.layers.register_forward_hook_for_layer(layer_signature, save_activations_hook)

        model.model.eval()
        batch_counter = 0
        try:
            with torch.inference_mode():
                for batch_index, texts in enumerate(dataset.iter_batches(batch_size)):
                    if not texts:
                        continue

                    tok_kwargs = {
                        "padding": True,
                        "truncation": True,
                        "return_tensors": "pt",
                    }
                    if max_length is not None:
                        tok_kwargs["max_length"] = max_length

                    enc = model.lm_tokenizer.tokenize(texts, **tok_kwargs)

                    if verbose:
                        _ii = enc.get("input_ids")
                        seq_len = int(_ii.shape[-1]) if _ii is not None else -1
                        logger.info(f"Prepared batch {batch_index}: items={len(texts)}, seq_len={seq_len}")

                    # Move inputs to model device (non_blocking if possible)
                    if device_type == "cuda":
                        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
                    else:
                        enc = {k: v.to(device) for k, v in enc.items()}

                    # Optional autocast for faster matmul on CUDA
                    if autocast and device_type == "cuda":
                        amp_dtype = autocast_dtype or torch.float16
                        cm = torch.autocast(device_type, dtype=amp_dtype)  # type: ignore[arg-type]
                    else:
                        # noop context manager
                        class _Noop:
                            def __enter__(self):
                                return None

                            def __exit__(self, exc_type, exc, tb):
                                return False

                        cm = _Noop()

                    with cm:
                        _ = model(**enc)  # type: ignore[attr-defined]

                    # Prepare payload to store
                    payload: dict[str, torch.Tensor] = {}
                    if "activations" in captured:
                        act = captured.pop("activations")
                        if dtype is not None:
                            try:
                                act = act.to(dtype, copy=False)
                            except Exception:
                                act = act.to(dtype)
                        # Move to CPU for persistence (non_blocking if CUDA)
                        if device_type == "cuda":
                            act = act.to("cpu", non_blocking=True)
                        else:
                            act = act.to("cpu")
                        payload["activations"] = act

                    if save_inputs:
                        input_ids = enc.get("input_ids")
                        attn = enc.get("attention_mask")
                        if input_ids is not None:
                            payload["input_ids"] = input_ids.detach().to("cpu", non_blocking=(device_type == "cuda"))
                        if attn is not None:
                            payload["attention_mask"] = attn.detach().to("cpu", non_blocking=(device_type == "cuda"))

                    # Persist and cleanup to keep memory bounded
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
            try:
                handle.remove()
            except Exception:
                pass
            if verbose:
                logger.info(f"Completed save_model_activations: run={run_name}, batches_saved={batch_counter}")
