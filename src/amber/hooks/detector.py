import abc
from typing import Any, TYPE_CHECKING, List, Dict
import torch

from amber.hooks.hook import Hook, HookType

if TYPE_CHECKING:
    from torch import nn
    from amber.store import Store


class Detector(Hook):
    """
    Abstract base class for detector hooks that collect metadata during inference.
    
    Detectors can accumulate data across batches and optionally save it to a Store.
    """
    
    def __init__(
        self,
        layer_signature: str | int,
        hook_type: HookType | str = HookType.FORWARD,
        hook_id: str | None = None,
        store: "Store | None" = None
    ):
        """
        Initialize a detector hook.
        
        Args:
            layer_signature: Layer to attach to
            hook_type: Type of hook (HookType.FORWARD or HookType.PRE_FORWARD)
            hook_id: Unique identifier
            store: Optional Store for saving metadata
        """
        super().__init__(layer_signature, hook_type, hook_id)
        self.store = store
        self._batch_metadata: List[Dict[str, Any]] = []
        self._current_batch_index = 0
    
    def _hook_fn(self, module: "nn.Module", inputs: tuple, output: Any) -> Any:
        """
        Internal hook function that collects metadata.
        
        This calls process_activations and collect_metadata.
        """
        if not self._enabled:
            return None
        
        try:
            # Process activations (subclass-specific logic)
            self.process_activations(module, inputs, output)
            
            # Collect metadata for this batch
            metadata = self.collect_metadata(module, inputs, output)
            if metadata is not None:
                self._batch_metadata.append(metadata)
        except Exception:
            # Don't let hook errors crash inference
            pass
        
        # Return None to not modify output
        return None
    
    @abc.abstractmethod
    def process_activations(self, module: "nn.Module", inputs: tuple, output: Any) -> None:
        """
        Process activations from the hooked layer.
        
        This is where detector-specific logic goes (e.g., tracking top activations,
        computing statistics, etc.).
        
        Args:
            module: The PyTorch module being hooked
            inputs: Tuple of inputs to the module
            output: Output from the module
        """
        pass

    def collect_metadata(self, module: "nn.Module", inputs: tuple, output: Any) -> Dict[str, Any] | None:
        """
        Collect metadata for the current batch.
        
        Override this to customize what metadata is collected per batch.
        By default, returns None (no per-batch metadata).
        
        The returned dict can contain any fields the user wants, for example:
        - 'layer_signature': str | int
        - 'batch_index': int
        - 'activations': torch.Tensor
        - Any other custom fields
        
        Args:
            module: The PyTorch module being hooked
            inputs: Tuple of inputs to the module
            output: Output from the module
            
        Returns:
            Dictionary with metadata or None
        """
        return None
    
    def get_accumulated_metadata(self) -> List[Dict[str, Any]]:
        """
        Get all accumulated metadata across batches.
        
        Returns:
            List of metadata dictionaries
        """
        return self._batch_metadata.copy()
    
    def reset_metadata(self) -> None:
        """Clear accumulated metadata."""
        self._batch_metadata.clear()
        self._current_batch_index = 0
    
    def save_metadata(self, run_name: str, store: "Store | None" = None) -> None:
        """
        Save accumulated metadata to a Store.
        
        Args:
            run_name: Name for this run/session
            store: Optional Store (uses self.store if not provided)
        """
        target_store = store or self.store
        if target_store is None:
            raise ValueError("No store available for saving metadata")
        
        # Convert metadata to saveable format
        metadata_list = []
        for meta in self._batch_metadata:
            # Convert tensors to CPU and make serializable
            meta_dict = {}
            for key, value in meta.items():
                if isinstance(value, torch.Tensor):
                    meta_dict[key] = value.detach().cpu()
                else:
                    meta_dict[key] = value
            metadata_list.append(meta_dict)
        
        # Save using store's metadata API
        save_data = {
            'hook_id': self.id,
            'layer_signature': str(self.layer_signature),
            'hook_type': self.hook_type,
            'num_batches': len(metadata_list),
            'metadata': metadata_list
        }
        
        try:
            target_store.put_run_meta(f"{run_name}_detector_{self.id}", save_data)
        except Exception:
            # If put_run_meta not available, try alternative
            pass
