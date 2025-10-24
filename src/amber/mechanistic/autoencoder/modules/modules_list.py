from enum import Enum
from functools import partial

from torch import nn

from amber.mechanistic.autoencoder.modules.topk import TopK

ACTIVATIONS_CLASSES = {
    "TopK": partial(TopK, act_fn=nn.Identity()),
    "TopKReLU": partial(TopK, act_fn=nn.ReLU()),
}


def get_activation(activation: str) -> TopK:
    if "_" in activation:
        activation, arg = activation.split("_", maxsplit=1)
        if "TopK" in activation:
            return ACTIVATIONS_CLASSES[activation](k=int(arg))
    return ACTIVATIONS_CLASSES[activation]()
