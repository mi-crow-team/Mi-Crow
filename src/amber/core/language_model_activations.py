from typing import TYPE_CHECKING

from amber.adapters.text_snippet_dataset import TextSnippetDataset
from amber.store import Store
from amber.utils import get_logger

import torch

if TYPE_CHECKING:
    from amber.core.language_model_context import LanguageModelContext


class LanguageModelActivations:
    def __init__(self, context: "LanguageModelContext"):
        self.context = context

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

        captured: dict[str, torch.Tensor] = {}

        def save_activations_hook(_module, _inputs, output):
            tensor = None
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
        handle = self.context.language_model.layers.register_forward_hook_for_layer(layer_signature,
                                                                                    save_activations_hook)
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
                        discard_output=True,
                        save_inputs=save_inputs,
                    )
                    if save_inputs:
                        inp_ids, attn = res
                        if isinstance(inp_ids, torch.Tensor):
                            payload["input_ids"] = inp_ids
                        if isinstance(attn, torch.Tensor):
                            payload["attention_mask"] = attn

                    if "activations" in captured:
                        act = captured.pop("activations")
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
