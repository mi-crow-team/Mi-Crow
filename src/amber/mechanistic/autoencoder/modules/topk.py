from typing import Callable

import torch
from torch import nn, Tensor
from pathlib import Path

from amber.mechanistic.autoencoder.sae_module import SaeModuleABC


# Soft cappping - przeskalowanie z 0 do inf, na razie nie

# forward_eval - w teorii model się nauczy brać k

# tied - decoder to odwrotność encodera. słabe

# nn.Parameter - rozdzielamy wagi i bias

# project_grads_decode - kolumny w decoderze wskazują kierunki konceptów. Upewniamy się że wektory będą kierunkowe. może trochę nie zadzialać

# scale_to_unit_norm - jak grads_decode nie zeskaluje dobrze to poprawia

# preprocesss -skip

# load model - dużo do wyjjebania

class TopK(SaeModuleABC):
    default_model_path: Path = Path("./models/topk")
    model_name: str = "topk"

    def __init__(self, k: int, act_fn: Callable = nn.Identity(), use_abs: bool = False) -> None:
        super().__init__()
        self.k = k
        self.act_fn = act_fn
        self.use_abs = use_abs

    def extra_repr(self) -> str:
        """Return string representation of module parameters."""
        return f"k={self.k}, act_fn={self.act_fn}, use_abs={self.use_abs}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use absolute values if requested
        values = torch.abs(x) if self.use_abs else x

        # Get indices of top-k elements
        _, indices = torch.topk(values, k=min(self.k, x.shape[-1]), dim=-1)

        # Gather original values at those indices
        top_values = torch.gather(x, -1, indices)

        # Apply activation function to those values
        activated_values = self.act_fn(top_values)

        # Create output tensor of zeros and place activated values at correct positions
        result = torch.zeros_like(x)
        result.scatter_(-1, indices, activated_values)

        # Verify that we have at most k non-zero elements per sample
        return result

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.abs(x) if self.use_abs else x
        x = self.act_fn(x)
        return x
