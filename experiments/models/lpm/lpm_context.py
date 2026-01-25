from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import torch


@dataclass
class LPMContext:
    """
    Context for Latent Prototype Moderator (LPM).
    Holds configuration and learned parameters (prototypes, covariance).
    """

    # Model & Layer info
    model_id: Optional[str] = None  # e.g., "speakleash/Bielik-1.5B-v3.0-Instruct"
    layer_signature: Optional[str] = None  # e.g., "llamaforcausallm_model_layers_27"
    layer_number: Optional[int] = None  # e.g., 27

    # Training Dataset info
    dataset_name: Optional[str] = None  # e.g., "wildguard_train"
    run_id: Optional[str] = None  # The run_id where activations were saved

    # LPM Parameters
    distance_metric: Literal["euclidean", "mahalanobis"] = "euclidean"
    aggregation_method: Literal["mean", "last_token", "last_token_prefix"] = "last_token"

    # Learned State
    # Maps class name (e.g., "harmful", "safe") to prototype tensor [hidden_dim]
    prototypes: Dict[str, torch.Tensor] = field(default_factory=dict)

    # Inverse covariance matrix (Precision matrix) [hidden_dim, hidden_dim]
    # Used for Mahalanobis distance. Shared across classes.
    precision_matrix: Optional[torch.Tensor] = None

    # Dimensionality
    hidden_dim: Optional[int] = None

    # Device
    device: str = "cpu"

    def to(self, device: str):
        """Move all tensors to device."""
        self.device = device
        for k, v in self.prototypes.items():
            self.prototypes[k] = v.to(device)

        if self.precision_matrix is not None:
            self.precision_matrix = self.precision_matrix.to(device)
