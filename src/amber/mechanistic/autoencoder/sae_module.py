import abc

import torch.nn as nn
from pathlib import Path


class SaeModuleABC(abc.ABC, nn.Module):
    default_model_path: Path = Path("./models/unknown")
    model_name: str = "sae_abc"

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def forward_eval(self, x):
        pass
