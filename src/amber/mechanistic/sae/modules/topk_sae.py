from pathlib import Path
from typing import Any, Iterator, Optional
from dataclasses import dataclass

import torch
from torch import nn

from overcomplete import (
    TopKSAE as OvercompleteTopkSAE,
    SAE as OvercompleteSAE
)

from amber.hooks.hook import HookType, HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT
from amber.mechanistic.sae.sae import Sae
from amber.mechanistic.sae.sae_trainer import SaeTrainingConfig
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
            store: Store | None = None,
            *args: Any,
            **kwargs: Any
    ) -> None:
        self.k = k
        super().__init__(n_latents, n_inputs, hook_id, device, store, *args, **kwargs)

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
            inputs: torch.Tensor | None,
            output: torch.Tensor | None
    ) -> torch.Tensor | None:
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
        if self.hook_type == HookType.FORWARD:
            tensor = output
        else:
            tensor = inputs[0] if len(inputs) > 0 else None

        if tensor is None:
            return output if self.hook_type == HookType.FORWARD else inputs

        # Check if tensor is actually a torch.Tensor
        if not isinstance(tensor, torch.Tensor):
            # For FORWARD hooks with non-tensor output, try to extract tensor from inputs
            if self.hook_type == HookType.FORWARD and len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                tensor = inputs[0]
            else:
                # Can't process non-tensor, return original
                return output if self.hook_type == HookType.FORWARD else inputs

        original_shape = tensor.shape

        # Flatten to 2D if needed: (batch, seq_len, hidden) -> (batch * seq_len, hidden)
        needs_reshape = len(original_shape) > 2
        if needs_reshape:
            batch_size, seq_len = original_shape[:2]
            tensor = tensor.reshape(-1, original_shape[-1])

        # Encode to get latents
        latents = self.encode(tensor)

        # Update top texts if text tracking is enabled
        if self._text_tracking_enabled and self.context.lm is not None:
            input_tracker = self.context.lm.get_input_tracker()
            if input_tracker is not None:
                texts = input_tracker.get_current_texts()
                if texts:
                    # For text tracking, use full (non-sparse) activations, not sparse TopK
                    # Overcomplete TopKSAE encode returns (pre_codes, codes) where codes are sparse
                    # We need pre_codes (full activations) for accurate text tracking
                    # pre_codes are the raw activations before TopK sparsity
                    pre_codes, _ = self.sae_engine.encode(tensor)
                    # Use pre_codes directly (signed values) - the _text_tracking_negative flag
                    # in concepts will handle whether to track negative activations
                    full_latents = pre_codes

                    # Update top texts using full latents and texts
                    # Latents are already flattened, so pass original_shape for token index calculation
                    self.concepts.update_top_texts_from_latents(
                        full_latents.detach().cpu(),
                        texts,
                        original_shape
                    )

        # Apply concept manipulation if parameters are set
        # Check if multiplication or bias differ from defaults (ones)
        if not torch.allclose(self.concepts.multiplication, torch.ones_like(self.concepts.multiplication)) or \
                not torch.allclose(self.concepts.bias, torch.ones_like(self.concepts.bias)):
            # Apply manipulation: latents = latents * multiplication + bias
            latents = latents * self.concepts.multiplication + self.concepts.bias

        # Decode to get reconstruction
        reconstructed = self.decode(latents)

        # Reshape back to original shape if needed
        if needs_reshape:
            reconstructed = reconstructed.reshape(original_shape)

        # Return in appropriate format
        if self.hook_type == HookType.FORWARD:
            if isinstance(output, torch.Tensor):
                return reconstructed
            elif isinstance(output, (tuple, list)):
                # Replace first tensor in tuple/list
                result = list(output)
                for i, item in enumerate(result):
                    if isinstance(item, torch.Tensor):
                        result[i] = reconstructed
                        break
                return tuple(result) if isinstance(output, tuple) else result
            else:
                # For objects with attributes, try to set last_hidden_state
                if hasattr(output, "last_hidden_state"):
                    output.last_hidden_state = reconstructed
                return output
        else:  # PRE_FORWARD
            # Return modified inputs tuple
            result = list(inputs)
            if len(result) > 0:
                result[0] = reconstructed
            return tuple(result)

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

    def process_activations(
            self,
            module: torch.nn.Module,
            input: HOOK_FUNCTION_INPUT,
            output: HOOK_FUNCTION_OUTPUT
    ) -> None:
        """
        Override process_activations to save full neuron activations (pre_codes) 
        instead of sparse TopK codes, with per-item batch metadata.
        
        This ensures we capture all neuron values, not just the TopK active ones,
        and saves metadata for each item in the batch individually.
        
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
        
        # Get full activations (pre_codes) instead of sparse codes
        # Overcomplete TopKSAE encode returns (pre_codes, codes)
        pre_codes, _ = self.sae_engine.encode(tensor_flat)
        latents = pre_codes  # Use full activations
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
