from typing import Dict, List, Callable, TYPE_CHECKING, Any

from torch import nn

from mi_crow.hooks.hook import Hook, HookType
from mi_crow.hooks.detector import Detector
from mi_crow.hooks.controller import Controller

if TYPE_CHECKING:
    from mi_crow.language_model.context import LanguageModelContext


class LanguageModelLayers:
    """Manages layer access and hook registration for LanguageModel."""

    def __init__(
            self,
            context: "LanguageModelContext",
    ):
        """
        Initialize LanguageModelLayers.
        
        Args:
            context: LanguageModelContext instance
        """
        self.context = context
        self.name_to_layer: Dict[str, nn.Module] = {}
        self.idx_to_layer: Dict[int, nn.Module] = {}
        self._flatten_layer_names()

    def _flatten_layer_names(self) -> tuple[Dict[str, nn.Module], Dict[int, nn.Module]]:
        """
        Flatten model structure into name and index mappings.
        
        Returns:
            Tuple of (name_to_layer, idx_to_layer) dictionaries
            
        Raises:
            ValueError: If model is not initialized
        """
        if self.context.model is None:
            raise ValueError("Model must be initialized before accessing layers")
        
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

    def _get_layer_by_name(self, layer_name: str) -> nn.Module:
        """
        Get layer by name.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Layer module
            
        Raises:
            ValueError: If layer name not found
        """
        if not self.name_to_layer:
            self._flatten_layer_names()
        if layer_name not in self.name_to_layer:
            raise ValueError(f"Layer name '{layer_name}' not found in model.")
        return self.name_to_layer[layer_name]

    def _get_layer_by_index(self, layer_index: int) -> nn.Module:
        """
        Get layer by index.
        
        Args:
            layer_index: Index of the layer
            
        Returns:
            Layer module
            
        Raises:
            ValueError: If layer index not found
        """
        if not self.idx_to_layer:
            self._flatten_layer_names()
        if layer_index not in self.idx_to_layer:
            raise ValueError(f"Layer index '{layer_index}' not found in model.")
        return self.idx_to_layer[layer_index]

    def get_layer_names(self) -> List[str]:
        """
        Get all layer names.
        
        Returns:
            List of layer names
        """
        return list(self.name_to_layer.keys())

    def print_layer_names(self) -> None:
        """
        Print layer names with basic info.
        
        Useful for debugging and exploring model structure.
        """
        names = self.get_layer_names()
        for name in names:
            layer = self.name_to_layer[name]
            weight_shape = getattr(layer, 'weight', None)
            weight_info = weight_shape.shape if weight_shape is not None else 'No weight'
            print(f"{name}: {weight_info}")

    def register_forward_hook_for_layer(
            self,
            layer_signature: str | int,
            hook: Callable,
            hook_args: dict = None
    ) -> Any:
        """
        Register a forward hook directly on a layer.
        
        Args:
            layer_signature: Layer name or index
            hook: Hook callable
            hook_args: Optional arguments for register_forward_hook
            
        Returns:
            Hook handle
        """
        layer = self._resolve_layer(layer_signature)
        return layer.register_forward_hook(hook, **(hook_args or {}))

    def register_pre_forward_hook_for_layer(
            self,
            layer_signature: str | int,
            hook: Callable,
            hook_args: dict = None
    ) -> Any:
        """
        Register a pre-forward hook directly on a layer.
        
        Args:
            layer_signature: Layer name or index
            hook: Hook callable
            hook_args: Optional arguments for register_forward_pre_hook
            
        Returns:
            Hook handle
        """
        layer = self._resolve_layer(layer_signature)
        return layer.register_forward_pre_hook(hook, **(hook_args or {}))

    def _resolve_layer(self, layer_signature: str | int) -> nn.Module:
        """
        Resolve layer signature to actual layer module.
        
        Args:
            layer_signature: Layer name (str) or index (int)
            
        Returns:
            Layer module
        """
        if isinstance(layer_signature, int):
            return self._get_layer_by_index(layer_signature)
        return self._get_layer_by_name(layer_signature)

    def _get_hook_type_from_hook(self, hook: Hook) -> HookType:
        """Get hook type from hook instance, normalizing if needed.
        
        Args:
            hook: Hook instance
            
        Returns:
            HookType enum value
        """
        return hook.hook_type

    def _validate_hook_registration(self, layer_signature: str | int, hook: Hook) -> None:
        """
        Validate hook registration constraints.
        
        Args:
            layer_signature: Layer signature
            hook: Hook instance to register
            
        Raises:
            ValueError: If hook ID is not unique or mixing hook types on same layer
        """
        if hook.id in self.context._hook_id_map:
            raise ValueError(f"Hook with ID '{hook.id}' is already registered")

        if layer_signature not in self.context._hook_registry:
            return

        existing_types = self._get_existing_hook_types(layer_signature)
        
        # Check if new hook is both Controller and Detector
        is_both = hook._is_both_controller_and_detector()
        if is_both:
            # Dual-inheritance hooks are compatible with either type
            # They can be registered if there are no existing hooks, or if existing hooks are compatible
            if existing_types:
                # Check if existing hooks are also dual-inheritance or compatible type
                # If existing is Controller, dual hook is compatible (it's also Controller)
                # If existing is Detector, dual hook is compatible (it's also Detector)
                # If existing is both, they're compatible
                # So we allow registration in all cases
                pass
        else:
            # Single-type hook: check compatibility
            new_hook_class = "Detector" if isinstance(hook, Detector) else "Controller"
            if existing_types and new_hook_class not in existing_types:
                existing_type_str = ", ".join(existing_types)
                raise ValueError(
                    f"Cannot register {new_hook_class} hook on layer '{layer_signature}': "
                    f"layer already has {existing_type_str} hook(s). "
                    f"Only one hook class type (Detector or Controller) per layer is allowed, "
                    f"or use a hook that inherits from both."
                )

    def _get_existing_hook_types(self, layer_signature: str | int) -> set[str]:
        """Get set of existing hook class types for a layer.
        
        Args:
            layer_signature: Layer signature
            
        Returns:
            Set of hook class type names (e.g., {"Detector", "Controller"})
        """
        existing_types = set()
        for existing_hook_type, hooks in self.context._hook_registry[layer_signature].items():
            if hooks:
                first_hook = hooks[0][0]
                # Check if hook is both Controller and Detector
                if first_hook._is_both_controller_and_detector():
                    existing_types.add("Detector")
                    existing_types.add("Controller")
                elif isinstance(first_hook, Detector):
                    existing_types.add("Detector")
                elif isinstance(first_hook, Controller):
                    existing_types.add("Controller")
        return existing_types

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
            hook_type: Type of hook (HookType.FORWARD or HookType.PRE_FORWARD). 
                      If None, uses hook.hook_type
            
        Returns:
            The hook's ID
            
        Raises:
            ValueError: If hook ID is not unique or if mixing hook types on same layer
        """
        layer = self._resolve_layer(layer_signature)
        
        if hook_type is None:
            hook_type = self._get_hook_type_from_hook(hook)
        elif isinstance(hook_type, str):
            hook_type = HookType(hook_type)
        
        self._validate_hook_registration(layer_signature, hook)

        hook.layer_signature = layer_signature
        hook.set_context(self.context)

        if layer_signature not in self.context._hook_registry:
            self.context._hook_registry[layer_signature] = {}

        if hook_type not in self.context._hook_registry[layer_signature]:
            self.context._hook_registry[layer_signature][hook_type] = []

        torch_hook_fn = hook.get_torch_hook()

        if hook_type == HookType.PRE_FORWARD:
            handle = layer.register_forward_pre_hook(torch_hook_fn)
        else:
            handle = layer.register_forward_hook(torch_hook_fn)

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

        if layer_signature not in self.context._hook_registry:
            del self.context._hook_id_map[hook_id]
            return True
        
        hook_types = self.context._hook_registry[layer_signature]
        if hook_type not in hook_types:
            del self.context._hook_id_map[hook_id]
            return True
        
        hooks_list = hook_types[hook_type]
        for i, (h, handle) in enumerate(hooks_list):
            if h.id == hook_id:
                handle.remove()
                hooks_list.pop(i)
                break

        del self.context._hook_id_map[hook_id]
        return True

    def _get_hooks_from_registry(
            self,
            layer_signature: str | int | None,
            hook_type: HookType | None
    ) -> List[Hook]:
        """Get hooks from registry with optional filtering.
        
        Args:
            layer_signature: Optional layer to filter by
            hook_type: Optional hook type to filter by
            
        Returns:
            List of Hook instances
        """
        hooks = []
        
        if layer_signature is not None:
            # Filter by specific layer
            if layer_signature in self.context._hook_registry:
                layer_hooks = self.context._hook_registry[layer_signature]
                if hook_type is not None:
                    if hook_type in layer_hooks:
                        hooks.extend([h for h, _ in layer_hooks[hook_type]])
                else:
                    for type_hooks in layer_hooks.values():
                        hooks.extend([h for h, _ in type_hooks])
        else:
            # Get all hooks across all layers
            for layer_hooks in self.context._hook_registry.values():
                if hook_type is not None:
                    if hook_type in layer_hooks:
                        hooks.extend([h for h, _ in layer_hooks[hook_type]])
                else:
                    for type_hooks in layer_hooks.values():
                        hooks.extend([h for h, _ in type_hooks])

        return hooks

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
        # Normalize hook_type if string
        normalized_hook_type = None
        if hook_type is not None:
            if isinstance(hook_type, str):
                normalized_hook_type = HookType(hook_type)
            else:
                normalized_hook_type = hook_type
        
        return self._get_hooks_from_registry(layer_signature, normalized_hook_type)

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
        """
        Get all registered Controller hooks.
        
        Returns:
            List of Controller instances
        """
        return [hook for hook in self.get_hooks() if isinstance(hook, Controller)]

    def get_detectors(self) -> List[Detector]:
        """
        Get all registered Detector hooks.
        
        Returns:
            List of Detector instances
        """
        return [hook for hook in self.get_hooks() if isinstance(hook, Detector)]
