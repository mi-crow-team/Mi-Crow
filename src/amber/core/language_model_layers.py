from typing import Dict, List, Callable, TYPE_CHECKING

from torch import nn

from amber.hooks.hook import Hook, HookType
from amber.mechanistic.autoencoder.autoencoder import Autoencoder
from amber.hooks.detector import Detector
from amber.hooks.controller import Controller

if TYPE_CHECKING:
    from amber.core.language_model_context import LanguageModelContext


class LanguageModelLayers:

    def __init__(
            self,
            context: "LanguageModelContext",
    ):
        self.context = context
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

        _recurse(self.context.model, self.context.model.__class__.__name__.lower(), [])

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
        # Resolve target layer
        if isinstance(after_layer_signature, int):
            after_layer = self._get_layer_by_index(after_layer_signature)
        else:
            after_layer = self._get_layer_by_name(after_layer_signature)

        after_layer.add_module(layer_name, layer)
        self._flatten_layer_names()

        new_layer_signature = f"{after_layer_signature}_{layer_name}"

        if isinstance(layer, Autoencoder):
            layer.concepts.lm = self.context.model
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

    def register_hook(
            self,
            layer_signature: str | int,
            hook: Hook,
            hook_type: HookType | str | None = None
    ) -> str:
        """
        Register a hook on a layer.
        
        Args:
            layer_signature: Layer name or index
            hook: Hook instance to register
            hook_type: Type of hook (HookType.FORWARD or HookType.PRE_FORWARD). If None, uses hook.hook_type
            
        Returns:
            The hook's ID
            
        Raises:
            ValueError: If hook ID is not unique or if mixing hook types on same layer
        """
        # Resolve layer
        if isinstance(layer_signature, int):
            layer = self._get_layer_by_index(layer_signature)
        else:
            layer = self._get_layer_by_name(layer_signature)

        # Use hook's hook_type if not specified
        if hook_type is None:
            hook_type = hook.hook_type

        # Convert string to enum if needed for backward compatibility
        if isinstance(hook_type, str):
            # Validate that it's a valid HookType value
            if hook_type not in [ht.value for ht in HookType]:
                raise ValueError(
                    f"Invalid hook_type string '{hook_type}'. "
                    f"Must be one of: {[ht.value for ht in HookType]}"
                )
            hook_type = HookType(hook_type)
        
        # Ensure hook_type is a HookType enum
        if not isinstance(hook_type, HookType):
            raise ValueError(
                f"hook_type must be a HookType enum, got {type(hook_type)}: {hook_type}"
            )

        # Check for duplicate hook ID
        if hook.id in self.context._hook_id_map:
            raise ValueError(f"Hook with ID '{hook.id}' is already registered")

        # Initialize registry for this layer if needed
        if layer_signature not in self.context._hook_registry:
            self.context._hook_registry[layer_signature] = {}

        # Check if we're mixing hook types (Detector vs Controller) on the same layer
        existing_types = set()
        for existing_hook_type, hooks in self.context._hook_registry[layer_signature].items():
            if hooks:  # If there are hooks of this type
                # Check the first hook to determine the class type
                first_hook = hooks[0][0]
                if isinstance(first_hook, Detector):
                    existing_types.add("Detector")
                elif isinstance(first_hook, Controller):
                    existing_types.add("Controller")

        # Determine the type of the new hook
        new_hook_class = "Detector" if isinstance(hook, Detector) else "Controller"

        # Enforce: only one hook class type per layer
        if existing_types and new_hook_class not in existing_types:
            existing_type_str = ", ".join(existing_types)
            raise ValueError(
                f"Cannot register {new_hook_class} hook on layer '{layer_signature}': "
                f"layer already has {existing_type_str} hook(s). "
                f"Only one hook class type (Detector or Controller) per layer is allowed."
            )

        # Initialize list for this hook type if needed
        if hook_type not in self.context._hook_registry[layer_signature]:
            self.context._hook_registry[layer_signature][hook_type] = []

        # Get the PyTorch-compatible hook function
        torch_hook_fn = hook.get_torch_hook()

        # Register with PyTorch
        if hook_type == HookType.PRE_FORWARD:
            handle = layer.register_forward_pre_hook(torch_hook_fn)
        else:  # forward
            handle = layer.register_forward_hook(torch_hook_fn)

        # Store in our registry
        self.context._hook_registry[layer_signature][hook_type].append((hook, handle))
        self.context._hook_id_map[hook.id] = (layer_signature, hook_type, hook)

        return hook.id

    def unregister_hook(self, hook_or_id: Hook | str) -> bool:
        """
        Unregister a hook by Hook instance or ID.
        
        Args:
            hook_or_id: Hook instance or hook ID string
            
        Returns:
            True if hook was found and removed, False otherwise
        """
        # Get hook ID
        if isinstance(hook_or_id, Hook):
            hook_id = hook_or_id.id
        else:
            hook_id = hook_or_id

        # Look up hook
        if hook_id not in self.context._hook_id_map:
            return False

        layer_signature, hook_type, hook = self.context._hook_id_map[hook_id]

        # Find and remove from registry
        if layer_signature in self.context._hook_registry:
            if hook_type in self.context._hook_registry[layer_signature]:
                hooks_list = self.context._hook_registry[layer_signature][hook_type]
                for i, (h, handle) in enumerate(hooks_list):
                    if h.id == hook_id:
                        # Remove PyTorch hook
                        handle.remove()
                        # Remove from our list
                        hooks_list.pop(i)
                        break

        # Remove from ID map
        del self.context._hook_id_map[hook_id]
        return True

    def get_hooks(
            self,
            layer_signature: str | int | None = None,
            hook_type: HookType | str | None = None
    ) -> List[Hook]:
        """
        Get registered hooks, optionally filtered by layer and/or type.
        
        Args:
            layer_signature: Optional layer to filter by
            hook_type: Optional hook type to filter by (HookType.FORWARD or HookType.PRE_FORWARD)
            
        Returns:
            List of Hook instances
        """
        # Convert string to enum if needed for backward compatibility
        if isinstance(hook_type, str):
            hook_type = HookType(hook_type)
        hooks = []

        if layer_signature is not None:
            # Get hooks for specific layer
            if layer_signature in self.context._hook_registry:
                layer_hooks = self.context._hook_registry[layer_signature]
                if hook_type is not None:
                    # Specific layer and type
                    if hook_type in layer_hooks:
                        hooks.extend([h for h, _ in layer_hooks[hook_type]])
                else:
                    # All hooks on this layer
                    for type_hooks in layer_hooks.values():
                        hooks.extend([h for h, _ in type_hooks])
        else:
            # All hooks (optionally filtered by type)
            for layer_hooks in self.context._hook_registry.values():
                if hook_type is not None:
                    if hook_type in layer_hooks:
                        hooks.extend([h for h, _ in layer_hooks[hook_type]])
                else:
                    for type_hooks in layer_hooks.values():
                        hooks.extend([h for h, _ in type_hooks])

        return hooks

    def enable_hook(self, hook_id: str) -> bool:
        """
        Enable a specific hook by ID.
        
        Args:
            hook_id: Hook ID to enable
            
        Returns:
            True if hook was found and enabled, False otherwise
        """
        if hook_id in self.context._hook_id_map:
            _, _, hook = self.context._hook_id_map[hook_id]
            hook.enable()
            return True
        return False

    def disable_hook(self, hook_id: str) -> bool:
        """
        Disable a specific hook by ID.
        
        Args:
            hook_id: Hook ID to disable
            
        Returns:
            True if hook was found and disabled, False otherwise
        """
        if hook_id in self.context._hook_id_map:
            _, _, hook = self.context._hook_id_map[hook_id]
            hook.disable()
            return True
        return False

    def enable_all_hooks(self) -> None:
        """Enable all registered hooks."""
        for _, _, hook in self.context._hook_id_map.values():
            hook.enable()

    def disable_all_hooks(self) -> None:
        """Disable all registered hooks."""
        for _, _, hook in self.context._hook_id_map.values():
            hook.disable()

    def get_controllers(self) -> List[Controller]:
        """Get all registered Controller hooks."""
        return [hook for hook in self.get_hooks() if isinstance(hook, Controller)]

    def get_detectors(self) -> List[Detector]:
        """Get all registered Detector hooks."""
        return [hook for hook in self.get_hooks() if isinstance(hook, Detector)]
