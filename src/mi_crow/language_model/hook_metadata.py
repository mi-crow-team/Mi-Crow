"""Hook metadata collection and serialization."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from mi_crow.language_model.context import LanguageModelContext


def collect_hooks_metadata(context: "LanguageModelContext") -> Dict[str, List[Dict[str, Any]]]:
    """
    Collect metadata from all registered hooks.
    
    Args:
        context: LanguageModelContext containing hook registry
        
    Returns:
        Dictionary mapping layer_signature to list of hook metadata dictionaries
    """
    hooks_info: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    for layer_signature, hook_types in context._hook_registry.items():
        layer_key = str(layer_signature)
        for hook_type, hooks_list in hook_types.items():
            for hook, _ in hooks_list:
                hook_info = {
                    "hook_id": hook.id,
                    "hook_type": hook.hook_type.value if hasattr(hook.hook_type, 'value') else str(hook.hook_type),
                    "layer_signature": str(hook.layer_signature) if hook.layer_signature is not None else None,
                    "hook_class": hook.__class__.__name__,
                    "enabled": hook.enabled,
                }
                hooks_info[layer_key].append(hook_info)
    
    return dict(hooks_info)

