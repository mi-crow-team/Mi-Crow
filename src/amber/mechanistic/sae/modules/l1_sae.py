from pathlib import Path
from typing import Any

import torch
from overcomplete import SAE as OvercompleteSAE
from amber.hooks.hook import HookType, HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT
from amber.mechanistic.sae.sae import Sae
from amber.mechanistic.sae.sae_trainer import SaeTrainingConfig
from amber.store.store import Store
from amber.utils import get_logger

logger = get_logger(__name__)


class L1SaeTrainingConfig(SaeTrainingConfig):
    """Training configuration for L1 SAE models.
    
    Extends SaeTrainingConfig with L1-specific training parameters.
    Currently, L1 SAE uses the same training parameters as the base SAE,
    but this class provides a clear extension point for future L1-specific options.
    """
    pass


class L1Sae(Sae):
    """L1 Sparse Autoencoder implementation.
    
    This is a placeholder implementation. The actual L1 SAE functionality
    will be implemented in the future.
    """
    
    def __init__(
            self,
            n_latents: int,
            n_inputs: int,
            hook_id: str | None = None,
            device: str = 'cpu',
            store: Store | None = None,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(n_latents, n_inputs, hook_id, device, store, *args, **kwargs)

    def _initialize_sae_engine(self) -> OvercompleteSAE:
        """Initialize the SAE engine.
        
        TODO: Implement L1 SAE engine initialization.
        """
        raise NotImplementedError("L1Sae is not yet implemented. This is a placeholder class.")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using sae_engine.
        
        Args:
            x: Input tensor of shape [batch_size, n_inputs]
            
        Returns:
            Encoded latents (L1 sparse activations)
        
        TODO: Implement L1 SAE encoding.
        """
        raise NotImplementedError("L1Sae is not yet implemented. This is a placeholder class.")

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latents using sae_engine.
        
        Args:
            x: Encoded tensor of shape [batch_size, n_latents]
            
        Returns:
            Reconstructed tensor of shape [batch_size, n_inputs]
        
        TODO: Implement L1 SAE decoding.
        """
        raise NotImplementedError("L1Sae is not yet implemented. This is a placeholder class.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using sae_engine.
        
        Args:
            x: Input tensor of shape [batch_size, n_inputs]
            
        Returns:
            Reconstructed tensor of shape [batch_size, n_inputs]
        
        TODO: Implement L1 SAE forward pass.
        """
        raise NotImplementedError("L1Sae is not yet implemented. This is a placeholder class.")

    def train(
            self,
            store: Store,
            run_id: str,
            layer_signature: str | int,
            config: L1SaeTrainingConfig | None = None,
            training_run_id: str | None = None
    ) -> dict[str, Any]:
        """
        Train the L1 SAE model.
        
        Args:
            store: Store instance containing activation data
            run_id: ID of the activation run
            layer_signature: Layer signature to train on
            config: Training configuration
            training_run_id: Optional training run ID
        
        Returns:
            Dictionary containing training results
        
        TODO: Implement L1 SAE training.
        """
        raise NotImplementedError("L1Sae is not yet implemented. This is a placeholder class.")

    @classmethod
    def load(cls, path: Path) -> "L1Sae":
        """
        Load a saved L1Sae model from disk.
        
        Args:
            path: Path to the saved model
        
        Returns:
            Loaded L1Sae instance
        
        TODO: Implement L1 SAE model loading.
        """
        raise NotImplementedError("L1Sae is not yet implemented. This is a placeholder class.")

