import abc
from pathlib import Path
from typing import Any, TYPE_CHECKING, Literal

import torch
from torch import nn

from mi_crow.hooks.controller import Controller
from mi_crow.hooks.detector import Detector
from mi_crow.hooks.hook import Hook, HookType, HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT
from mi_crow.mechanistic.sae.autoencoder_context import AutoencoderContext
from mi_crow.mechanistic.sae.concepts.autoencoder_concepts import AutoencoderConcepts
from mi_crow.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary
from mi_crow.mechanistic.sae.sae_trainer import SaeTrainer
from mi_crow.store.store import Store
from mi_crow.utils import get_logger

from overcomplete.sae import SAE as OvercompleteSAE

if TYPE_CHECKING:
    pass

ActivationFn = Literal["relu", "linear"] | None

logger = get_logger(__name__)


class Sae(Controller, Detector, abc.ABC):
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
        # Initialize both Controller and Detector
        Controller.__init__(self, hook_type=HookType.FORWARD, hook_id=hook_id)
        Detector.__init__(self, hook_type=HookType.FORWARD, hook_id=hook_id, store=store)

        self._autoencoder_context = AutoencoderContext(
            autoencoder=self,
            n_latents=n_latents,
            n_inputs=n_inputs
        )
        self._autoencoder_context.device = device
        self.sae_engine: OvercompleteSAE = self._initialize_sae_engine()
        self.concepts = AutoencoderConcepts(self._autoencoder_context)

        # Text tracking flag
        self._text_tracking_enabled: bool = False

        # Training component
        self.trainer = SaeTrainer(self)

    @property
    def context(self) -> AutoencoderContext:
        """Get the AutoencoderContext associated with this SAE."""
        return self._autoencoder_context

    @context.setter
    def context(self, value: AutoencoderContext) -> None:
        """Set the AutoencoderContext for this SAE."""
        self._autoencoder_context = value

    def set_context(self, context: "LanguageModelContext") -> None:
        """Set the LanguageModelContext for this hook and sync to AutoencoderContext.
        
        When the hook is registered, this method is called with the LanguageModelContext.
        It automatically syncs relevant values to the AutoencoderContext, including device.
        
        Args:
            context: The LanguageModelContext instance from the LanguageModel
        """
        Hook.set_context(self, context)
        self._context = context
        if context is not None:
            self._autoencoder_context.lm = context.language_model
            if context.model_id is not None:
                self._autoencoder_context.model_id = context.model_id
            if context.store is not None and self._autoencoder_context.store is None:
                self._autoencoder_context.store = context.store
            if self.layer_signature is not None:
                self._autoencoder_context.lm_layer_signature = self.layer_signature
            if context.device is not None:
                self._autoencoder_context.device = context.device

    @abc.abstractmethod
    def _initialize_sae_engine(self) -> OvercompleteSAE:
        raise NotImplementedError("Initialize SAE engine not implemented.")

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

    @abc.abstractmethod
    def process_activations(
            self,
            module: torch.nn.Module,
            input: HOOK_FUNCTION_INPUT,
            output: HOOK_FUNCTION_OUTPUT
    ) -> None:
        """
        Process activations to save neuron activations in metadata.
        
        This implements the Detector interface. It extracts activations, encodes them
        to get neuron activations (latents), and saves metadata for each item in the batch
        individually, including nonzero latent indices and activations.
        
        Args:
            module: The PyTorch module being hooked
            input: Tuple of input tensors to the module
            output: Output tensor(s) from the module
        """
        raise NotImplementedError("process_activations method not implemented.")

    @abc.abstractmethod
    def modify_activations(
            self,
            module: nn.Module,
            inputs: torch.Tensor | None,
            output: torch.Tensor | None
    ) -> torch.Tensor | None:
        raise NotImplementedError("modify_activations method not implemented.")

    @staticmethod
    def _apply_activation_fn(
            tensor: torch.Tensor,
            activation_fn: ActivationFn
    ) -> torch.Tensor:
        """
        Apply activation function to tensor.
        
        Args:
            tensor: Input tensor
            activation_fn: Activation function to apply ("relu", "linear", or None)
            
        Returns:
            Tensor with activation function applied
        """
        if activation_fn == "relu":
            return torch.relu(tensor)
        elif activation_fn == "linear" or activation_fn is None:
            return tensor
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}. Use 'relu', 'linear', or None")
