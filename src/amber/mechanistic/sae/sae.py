import abc
from pathlib import Path
from typing import Any, TYPE_CHECKING, Literal

import torch
from torch import nn

from amber.hooks.controller import Controller
from amber.hooks.detector import Detector
from amber.hooks.hook import HookType, HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT
from amber.mechanistic.sae.autoencoder_context import AutoencoderContext
from amber.mechanistic.sae.concepts.autoencoder_concepts import AutoencoderConcepts
from amber.mechanistic.sae.concepts.concept_dictionary import ConceptDictionary
from amber.mechanistic.sae.sae_trainer import SaeTrainer
from amber.store.store import Store
from amber.utils import get_logger

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
            inputs: torch.Tensor | None,
            output: torch.Tensor | None
    ) -> torch.Tensor | None:
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
        # Extract tensor from input/output based on hook type
        if self.hook_type == HookType.FORWARD:
            tensor = output
        else:
            tensor = input[0] if len(input) > 0 else None

        if tensor is None or not isinstance(tensor, torch.Tensor):
            return

        original_shape = tensor.shape

        # Flatten to 2D if needed: (batch, seq_len, hidden) -> (batch * seq_len, hidden)
        needs_reshape = len(original_shape) > 2
        if needs_reshape:
            tensor_flat = tensor.reshape(-1, original_shape[-1])
        else:
            tensor_flat = tensor

        # Encode to get latents (neuron activations)
        latents = self.encode(tensor_flat)
        latents_cpu = latents.detach().cpu()

        # Save full neuron activations tensor to tensor_metadata for backward compatibility
        if needs_reshape:
            batch_size, seq_len = original_shape[:2]
            latents_reshaped = latents_cpu.reshape(batch_size, seq_len, -1)
            self.tensor_metadata['neurons'] = latents_reshaped
        else:
            self.tensor_metadata['neurons'] = latents_cpu

        # Process each item in the batch individually
        # latents_cpu shape: [batch*seq, n_latents] or [batch, n_latents]
        batch_items = []
        n_items = latents_cpu.shape[0]

        for item_idx in range(n_items):
            item_latents = latents_cpu[item_idx]  # [n_latents]
            
            # Find nonzero indices for this item
            nonzero_mask = item_latents != 0
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).flatten().tolist()
            
            # Create map of nonzero indices to activations
            activations_map = {
                int(idx): float(item_latents[idx].item())
                for idx in nonzero_indices
            }
            
            # Create item metadata
            item_metadata = {
                "nonzero_indices": nonzero_indices,
                "activations": activations_map
            }
            
            batch_items.append(item_metadata)

        # Save batch items metadata
        self.metadata['batch_items'] = batch_items

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
