import abc
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from overcomplete.sae import SAE as OvercompleteSAE
from torch import nn

from amber.hooks.controller import Controller
from amber.hooks.hook import HookType
from amber.mechanistic.autoencoder.autoencoder_context import AutoencoderContext
from amber.mechanistic.autoencoder.concepts.autoencoder_concepts import AutoencoderConcepts
from amber.mechanistic.autoencoder.concepts.concept_dictionary import ConceptDictionary
from amber.mechanistic.autoencoder.sae_trainer import SaeTrainer
from amber.utils import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class Sae(Controller, abc.ABC):
    def __init__(
            self,
            n_latents: int,
            n_inputs: int,
            hook_id: str | None = None,
            device: str = 'cpu',
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(HookType.FORWARD, hook_id)
        self.context = AutoencoderContext(
            autoencoder=self,
            n_latents=n_latents,
            n_inputs=n_inputs
        )
        self.context.device = device
        self.sae_engine: OvercompleteSAE = self._initialize_sae_engine()
        self.concepts = AutoencoderConcepts(self.context)

        # Text tracking flag
        self._text_tracking_enabled: bool = False

        # Training component
        self.trainer = SaeTrainer(self)

    @abc.abstractmethod
    def _initialize_sae_engine(self) -> OvercompleteSAE:
        raise NotImplementedError("Initialize SAE engine not implemented.")

    @abc.abstractmethod
    def modify_activations(
            self,
            module: nn.Module,
            inputs: torch.Tensor,
            output: torch.Tensor
    ) -> Any:
        raise NotImplementedError("Modify activations method not implemented.")

    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Encode method not implemented.")

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Decode method not implemented.")

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward method not implemented.")

    @abc.abstractmethod
    def save(self, name: str):
        raise NotImplementedError("Save method not implemented.")

    @staticmethod
    @abc.abstractmethod
    def load(path: Path):
        raise NotImplementedError("Load method not implemented.")

    def attach_dictionary(self, concept_dictionary: ConceptDictionary):
        self.concepts.dictionary = concept_dictionary
