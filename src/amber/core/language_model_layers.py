from typing import Dict, List, Callable, TYPE_CHECKING

from torch import nn

from amber.mechanistic.autoencoder.autoencoder import Autoencoder

if TYPE_CHECKING:
    from amber.core.language_model import LanguageModel


class LanguageModelLayers:

    def __init__(
            self,
            lm: "LanguageModel",
            model: nn.Module
    ):
        self._model = model
        self._lm = lm
        self.name_to_layer: Dict[str, nn.Module] = {}
        self.idx_to_layer: Dict[int, nn.Module] = {}
        self._flatten_layer_names()

    def _flatten_layer_names(self):
        self.name_to_layer.clear()
        self.idx_to_layer.clear()

        def _recurse(module: nn.Module, prefix: str, idx: List[int]):
            for name, child in module.named_children():
                clean_name = f"{prefix}_{name}".replace(".", "_")
                idx_val = len(self.idx_to_layer)
                self.name_to_layer[clean_name] = child
                self.idx_to_layer[idx_val] = child
                _recurse(child, clean_name, idx)

        _recurse(self._model, self._model.__class__.__name__.lower(), [])

        return self.name_to_layer, self.idx_to_layer

    def _get_layer_by_name(self, layer_name: str):
        if not self.name_to_layer:
            self._flatten_layer_names()
        if layer_name not in self.name_to_layer:
            raise ValueError(f"Layer name '{layer_name}' not found in model.")
        return self.name_to_layer[layer_name]

    def _get_layer_by_index(self, layer_index: int):
        if not self.idx_to_layer:
            self._flatten_layer_names()
        if layer_index not in self.idx_to_layer:
            raise ValueError(f"Layer index '{layer_index}' not found in model.")
        return self.idx_to_layer[layer_index]

    def get_layer_names(self) -> List[str]:
        return list(self.name_to_layer.keys())

    def print_layer_names(self) -> None:
        # Print layer names with basic info; no return value
        names = self.get_layer_names()
        for name in names:
            layer = self.name_to_layer[name]
            print(f"{name}: {getattr(layer, 'weight', None).shape if hasattr(layer, 'weight') else 'No weight'}")

    def register_forward_hook_for_layer(
            self,
            layer_signature: str | int,
            hook: Callable,  # TODO: perhaps we could make some better signature
            hook_args: dict = None
    ):
        if isinstance(layer_signature, int):
            layer = self._get_layer_by_index(layer_signature)
        else:
            layer = self._get_layer_by_name(layer_signature)
        return layer.register_forward_hook(hook, **(hook_args or {}))

    def register_pre_forward_hook_for_layer(
            self,
            layer_signature: str | int,
            hook: Callable,  # TODO: perhaps we could make some better signature
            hook_args: dict = None
    ):
        if isinstance(layer_signature, int):
            layer = self._get_layer_by_index(layer_signature)
        else:
            layer = self._get_layer_by_name(layer_signature)
        return layer.register_forward_pre_hook(hook, **(hook_args or {}))

    def register_new_layer(
            self,
            layer_name: str,
            layer: nn.Module,
            after_layer_signature: str | int,
    ):
        """
        Attach a new layer under an existing module and *replace* the parent's output with the new layer's output.
        This is the only supported behavior.

        Usage:
            l1 -> m -> l2   # output of l1 becomes input to m; output of m is passed to l2
        """
        # Resolve target layer
        if isinstance(after_layer_signature, int):
            after_layer = self._get_layer_by_index(after_layer_signature)
        else:
            after_layer = self._get_layer_by_name(after_layer_signature)

        after_layer.add_module(layer_name, layer)
        self._flatten_layer_names()

        new_layer_signature = f"{after_layer_signature}_{layer_name}"

        if isinstance(layer, Autoencoder):
            layer.concepts.lm = self._lm
            layer.concepts.lm_layer_signature = new_layer_signature

        def _extract_main_tensor(output):
            import torch
            if isinstance(output, torch.Tensor):
                return output
            if isinstance(output, (tuple, list)):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        return item
                return None
            return None

        def _replace_output_hook(_module, _inputs, output):
            import torch
            x = _extract_main_tensor(output)
            if x is None:
                raise RuntimeError(
                    f"register_new_layer('{layer_name}'): could not extract a Tensor from parent output "
                    f"to feed into the new layer."
                )

            if isinstance(layer, Autoencoder):
                if x.dim() == 3:
                    b, t, d = x.shape
                    x_for_layer = x.reshape(b * t, d)  # [B*T, D]
                elif x.dim() == 2:
                    x_for_layer = x  # already [N, D]
                else:
                    raise RuntimeError(
                        f"register_new_layer('{layer_name}'): Autoencoder expected 2D or 3D tensor, got shape {tuple(x.shape)}"
                    )
                # Validate feature size if SAE exposes n_inputs
                n_inputs = getattr(layer, 'n_inputs', None)
                if n_inputs is not None and x_for_layer.shape[-1] != n_inputs:
                    raise RuntimeError(
                        f"register_new_layer('{layer_name}'): feature dim mismatch: SAE expects {n_inputs}, got {x_for_layer.shape[-1]}"
                    )
            else:
                x_for_layer = x

            try:
                try:
                    out = layer(x_for_layer, detach=True)
                except TypeError:
                    out = layer(x_for_layer)
            except Exception as e:
                raise RuntimeError(
                    f"register_new_layer('{layer_name}'): new layer forward failed with: {e}"
                ) from e

            # Ensure we return a Tensor, not a tuple/dict/object
            if isinstance(layer, Autoencoder):
                # SAE commonly returns: (recon, latents, recon_full, latents_full)
                recon = None
                recon_full = None
                if isinstance(out, (tuple, list)):
                    if len(out) > 0 and isinstance(out[0], torch.Tensor):
                        recon = out[0]
                    if len(out) > 2 and isinstance(out[2], torch.Tensor):
                        recon_full = out[2]
                elif isinstance(out, torch.Tensor):
                    recon = out
                elif hasattr(out, "reconstruction") and isinstance(out.reconstruction, torch.Tensor):
                    recon = out.reconstruction

                if recon_full is not None:
                    y = recon_full  # already matches [B, T, D] if parent was 3D
                elif recon is not None:
                    # If we flattened [B,T,D] -> [B*T,D], reshape back
                    try:
                        if 'b' in locals() and 't' in locals():
                            y = recon.view(b, t, -1)
                        else:
                            y = recon
                    except Exception:
                        # Fallback: keep recon as-is
                        y = recon
                else:
                    raise RuntimeError(
                        f"register_new_layer('{layer_name}'): SAE did not return a reconstruction tensor."
                    )
            else:
                # Generic layers: pick a tensor output deterministically
                if isinstance(out, torch.Tensor):
                    y = out
                elif isinstance(out, (tuple, list)):
                    y = None
                    for item in out:
                        if isinstance(item, torch.Tensor):
                            y = item
                            break
                    if y is None:
                        raise RuntimeError(
                            f"register_new_layer('{layer_name}'): non-SAE layer returned no tensor in tuple/list."
                        )
                elif hasattr(out, 'last_hidden_state') and isinstance(out.last_hidden_state, torch.Tensor):
                    y = out.last_hidden_state
                else:
                    raise RuntimeError(
                        f"register_new_layer('{layer_name}'): layer returned unsupported type {type(out)}."
                    )

            # For SAE, ensure the replacement has exactly the same shape as the parent's main tensor output
            if isinstance(layer, Autoencoder):
                orig = x  # original main tensor extracted from parent's output
                if isinstance(orig, torch.Tensor):
                    if y.shape != orig.shape:
                        # Try to safely reshape when numel matches and last dim aligns
                        reshaped = False
                        try:
                            if orig.dim() == 3 and y.dim() == 2 and y.shape[-1] == orig.shape[
                                -1] and y.numel() == orig.numel():
                                y = y.view_as(orig)
                                reshaped = True
                            elif orig.dim() == 2 and y.dim() == 3 and y.shape[-1] == orig.shape[
                                -1] and y.numel() == orig.numel():
                                y = y.reshape_as(orig)
                                reshaped = True
                        except Exception:
                            reshaped = False
                        if not reshaped and y.shape != orig.shape:
                            raise RuntimeError(
                                f"register_new_layer('{layer_name}'): SAE reconstruction shape {tuple(y.shape)} does not match parent output shape {tuple(orig.shape)}."
                            )

            return y  # <-- Replace parent's output with a Tensor

        # Returning a value from a forward hook replaces the module's output.
        return after_layer.register_forward_hook(_replace_output_hook)
