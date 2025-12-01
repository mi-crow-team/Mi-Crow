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
    model_id: Optional[str] = None
    layer_signature: Optional[str] = None

    # LPM Parameters
    distance_metric: Literal["euclidean", "mahalanobis"] = "euclidean"

    # Learned State
    # Maps class name (e.g., "harmful") to prototype tensor [hidden_dim]
    prototypes: Dict[str, torch.Tensor] = field(default_factory=dict)

    # Inverse covariance matrix (Precision matrix) [hidden_dim, hidden_dim]
    # Used for Mahalanobis distance. Shared across classes.
    precision_matrix: Optional[torch.Tensor] = None

    # Device
    device: str = "cpu"

    def to(self, device: str):
        """Move all tensors to device."""
        self.device = device
        for k, v in self.prototypes.items():
            self.prototypes[k] = v.to(device)

        if self.precision_matrix is not None:
            self.precision_matrix = self.precision_matrix.to(device)
