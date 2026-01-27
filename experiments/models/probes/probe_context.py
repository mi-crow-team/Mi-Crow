from dataclasses import dataclass
from typing import Literal, Optional

import torch


@dataclass
class ProbeContext:
    """
    Context for Linear Probe Classifier.
    Holds configuration and learned parameters (weights, bias).
    """

    # Model & Layer info
    model_id: Optional[str] = None  # e.g., "speakleash/Bielik-1.5B-v3.0-Instruct"
    layer_signature: Optional[str] = None  # e.g., "llamaforcausallm_model_layers_27"
    layer_number: Optional[int] = None  # e.g., 27

    # Training Dataset info
    dataset_name: Optional[str] = None  # e.g., "wgmix_train"
    run_id: Optional[str] = None  # The run_id where activations were saved

    # Probe Parameters
    aggregation_method: Literal["mean", "last_token", "last_token_prefix"] = "last_token"

    # Learned State (Linear Layer: y = W·x + b)
    weight: Optional[torch.Tensor] = None  # [hidden_dim] for binary classification
    bias: Optional[torch.Tensor] = None  # scalar

    # Training Hyperparameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 5  # Early stopping patience

    # Training History
    train_losses: list = None
    val_losses: list = None
    val_accuracies: list = None
    val_aucs: list = None
    best_epoch: Optional[int] = None
    num_train_samples: Optional[int] = None
    num_val_samples: Optional[int] = None

    # Dimensionality
    hidden_dim: Optional[int] = None

    # Device
    device: str = "cpu"

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.train_losses is None:
            self.train_losses = []
        if self.val_losses is None:
            self.val_losses = []
        if self.val_accuracies is None:
            self.val_accuracies = []
        if self.val_aucs is None:
            self.val_aucs = []

    def to(self, device: str):
        """Move all tensors to device."""
        self.device = device
        if self.weight is not None:
            self.weight = self.weight.to(device)
        if self.bias is not None:
            self.bias = self.bias.to(device)
