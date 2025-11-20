from pathlib import Path
from typing import Any, Iterator, Optional
from dataclasses import dataclass

import torch
from torch import nn

from overcomplete import (
    TopKSAE as OvercompleteTopkSAE,
    SAE as OvercompleteSAE
)

from amber.hooks.hook import HookType
from amber.mechanistic.sae.sae import Sae
from amber.mechanistic.sae.sae_trainer import SaeTrainingConfig
from amber.mechanistic.sae.utils import (
    extract_activation_tensor,
    reshape_for_sae,
    reshape_from_sae,
    reconstruct_hook_output
)
from amber.store.store import Store
from amber.utils import get_logger

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        return iterable if iterable is not None else range(0)

logger = get_logger(__name__)

# TopKSAETrainingConfig is now SaeTrainingConfig in sae.py
# Keep alias for backward compatibility
TopKSAETrainingConfig = SaeTrainingConfig


class TopKSae(Sae):
    def __init__(
            self,
            n_latents: int,
            n_inputs: int,
            k: int,
            hook_id: str | None = None,
            device: str = 'cpu',
            *args: Any,
            **kwargs: Any
    ) -> None:
        self.k = k
        super().__init__(n_latents, n_inputs, hook_id, device)

    def _initialize_sae_engine(self) -> OvercompleteSAE:
        return OvercompleteTopkSAE(
            input_shape=self.context.n_inputs,
            nb_concepts=self.context.n_latents,
            top_k=self.k,
            device=self.context.device
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using sae_engine.
        
        Args:
            x: Input tensor of shape [batch_size, n_inputs]
            
        Returns:
            Encoded latents (TopK sparse activations)
        """
        # Overcomplete TopKSAE encode returns (pre_codes, codes)
        _, codes = self.sae_engine.encode(x)
        return codes

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latents using sae_engine.
        
        Args:
            x: Encoded tensor of shape [batch_size, n_latents]
            
        Returns:
            Reconstructed tensor of shape [batch_size, n_inputs]
        """
        return self.sae_engine.decode(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using sae_engine.
        
        Args:
            x: Input tensor of shape [batch_size, n_inputs]
            
        Returns:
            Reconstructed tensor of shape [batch_size, n_inputs]
        """
        # Overcomplete TopKSAE forward returns (pre_codes, codes, x_reconstructed)
        _, _, x_reconstructed = self.sae_engine.forward(x)
        return x_reconstructed

    def train(
            self,
            store: Store,
            run_id: str,
            layer_signature: str | int,
            config: SaeTrainingConfig | None = None
    ) -> dict[str, list[float]]:
        """
        Train TopKSAE using activations from a Store.
        
        This method delegates to the SaeTrainer composite class.
        
        Args:
            store: Store instance containing activations
            run_id: Run ID to train on
            config: Training configuration
            
        Returns:
            Dictionary with training history
        """
        return self.trainer.train(store, run_id, layer_signature, config)

    def modify_activations(
            self,
            module: "torch.nn.Module",
            inputs: torch.Tensor,
            output: torch.Tensor
    ) -> Any:
        """
        Modify activations using TopKSAE (Controller hook interface).
        
        Extracts tensor from inputs/output, applies SAE forward pass,
        and optionally applies concept manipulation.
        
        Args:
            module: The PyTorch module being hooked
            inputs: Tuple of inputs to the module
            output: Output from the module (None for pre_forward hooks)
            
        Returns:
            Modified activations with same shape as input
        """
        # Extract activation tensor from hook inputs/output
        tensor, original_value = extract_activation_tensor(self.hook_type, inputs, output)
        
        if tensor is None:
            return original_value

        # Process activations through SAE
        reconstructed = self._process_activations(tensor)

        # Reconstruct output in original format
        return reconstruct_hook_output(self.hook_type, reconstructed, original_value)

    def _process_activations(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Process activations through SAE: encode, apply concepts, decode.
        
        Args:
            tensor: Input activation tensor
            
        Returns:
            Reconstructed tensor with same shape as input
            
        Raises:
            ValueError: If tensor shape is invalid
            RuntimeError: If SAE processing fails
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        
        if tensor.numel() == 0:
            raise ValueError("Cannot process empty tensor")
        
        if len(tensor.shape) < 2:
            raise ValueError(f"Expected tensor with at least 2 dimensions, got shape {tensor.shape}")

        try:
            # Reshape for SAE processing (flatten if 3D)
            tensor_2d, original_shape, needs_reshape = reshape_for_sae(tensor)

            # Validate reshaped tensor matches expected input size
            if tensor_2d.shape[-1] != self.context.n_inputs:
                raise ValueError(
                    f"Tensor feature dimension {tensor_2d.shape[-1]} does not match "
                    f"SAE input size {self.context.n_inputs}"
                )

            # Encode to get latents
            latents = self.encode(tensor_2d)

            # Update text tracking if enabled
            self._update_text_tracking(tensor_2d, latents, original_shape)

            # Apply concept manipulation if needed
            latents = self._apply_concept_manipulation(latents)

            # Decode to get reconstruction
            reconstructed = self.decode(latents)

            # Reshape back to original shape
            return reshape_from_sae(reconstructed, original_shape, needs_reshape)
        except Exception as e:
            raise RuntimeError(f"Failed to process activations through SAE: {e}") from e

    def _update_text_tracking(
            self,
            tensor_2d: torch.Tensor,
            latents: torch.Tensor,
            original_shape: tuple[int, ...]
    ) -> None:
        """
        Update text tracking if enabled.
        
        Args:
            tensor_2d: Flattened input tensor [batch * seq_len, hidden]
            latents: Encoded latents (sparse TopK)
            original_shape: Original tensor shape before flattening
        """
        if not (self._text_tracking_enabled and self.context.lm is not None):
            return

        input_tracker = self.context.lm.get_input_tracker()
        if input_tracker is None:
            return

        texts = input_tracker.get_current_texts()
        if not texts:
            return

        # For text tracking, use full (non-sparse) activations, not sparse TopK
        # Overcomplete TopKSAE encode returns (pre_codes, codes) where codes are sparse
        # We need pre_codes (full activations) for accurate text tracking
        pre_codes, _ = self.sae_engine.encode(tensor_2d)
        full_latents = pre_codes

        # Update top texts using full latents and texts
        self.concepts.update_top_texts_from_latents(
            full_latents.detach().cpu(),
            texts,
            original_shape
        )

    def _apply_concept_manipulation(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Apply concept manipulation (multiplication and bias) if parameters differ from defaults.
        
        Args:
            latents: Encoded latents tensor
            
        Returns:
            Manipulated latents tensor
        """
        # Check if multiplication or bias differ from defaults (ones)
        multiplication_default = torch.allclose(
            self.concepts.multiplication,
            torch.ones_like(self.concepts.multiplication)
        )
        bias_default = torch.allclose(
            self.concepts.bias,
            torch.ones_like(self.concepts.bias)
        )

        if multiplication_default and bias_default:
            return latents

        # Apply manipulation: latents = latents * multiplication + bias
        return latents * self.concepts.multiplication + self.concepts.bias

    def save(self, name: str, path: str | Path | None = None) -> None:
        """
        Save model using overcomplete's state dict + our metadata.
        
        Args:
            name: Model name
            path: Directory path to save to (defaults to current directory)
        """
        if path is None:
            path = Path.cwd()
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{name}.pt"

        # Save overcomplete model state dict
        sae_state_dict = self.sae_engine.state_dict()

        amber_metadata = {
            "concepts_state": {
                'multiplication': self.concepts.multiplication.data,
                'bias': self.concepts.bias.data,
            },
            "n_latents": self.context.n_latents,
            "n_inputs": self.context.n_inputs,
            "k": self.k,
            "device": self.context.device,
            "layer_signature": self.context.lm_layer_signature,
            "model_id": self.context.model_id,
        }

        payload = {
            "sae_state_dict": sae_state_dict,
            "amber_metadata": amber_metadata,
        }

        torch.save(payload, save_path)
        logger.info(f"Saved TopKSAE to {save_path}")

    @staticmethod
    def load(path: Path) -> "TopKSae":
        """
        Load TopKSAE from saved file using overcomplete's load method + our metadata.
        
        Args:
            path: Path to saved model file
            
        Returns:
            Loaded TopKSAE instance
        """
        p = Path(path)

        # Load payload
        payload = torch.load(
            p,
            map_location='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Extract our metadata
        if "amber_metadata" not in payload:
            raise ValueError(f"Invalid TopKSAE save format: missing 'amber_metadata' key in {p}")

        amber_meta = payload["amber_metadata"]
        n_latents = int(amber_meta["n_latents"])
        n_inputs = int(amber_meta["n_inputs"])
        k = int(amber_meta["k"])
        device = amber_meta.get("device", "cpu")
        layer_signature = amber_meta.get("layer_signature")
        model_id = amber_meta.get("model_id")
        concepts_state = amber_meta.get("concepts_state", {})

        # Create TopKSAE instance
        topk_sae = TopKSae(
            n_latents=n_latents,
            n_inputs=n_inputs,
            k=k,
            device=device
        )

        # Load overcomplete model state dict
        if "sae_state_dict" in payload:
            topk_sae.sae_engine.load_state_dict(payload["sae_state_dict"])
        elif "model" in payload:
            # Backward compatibility with old format
            topk_sae.sae_engine.load_state_dict(payload["model"])
        else:
            # Assume payload is the state dict itself (backward compatibility)
            topk_sae.sae_engine.load_state_dict(payload)

        # Load concepts state
        if concepts_state:
            if "multiplication" in concepts_state:
                topk_sae.concepts.multiplication.data = concepts_state["multiplication"]
            if "bias" in concepts_state:
                topk_sae.concepts.bias.data = concepts_state["bias"]

        # Note: Top texts loading was removed as serialization methods were removed
        # Top texts should be exported/imported separately if needed

        # Set context metadata
        topk_sae.context.lm_layer_signature = layer_signature
        topk_sae.context.model_id = model_id

        params_str = f"n_latents={n_latents}, n_inputs={n_inputs}, k={k}"
        logger.info(f"\nLoaded TopKSAE from {p}\n{params_str}")

        return topk_sae
