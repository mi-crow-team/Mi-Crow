from pathlib import Path
from typing import Any

import torch
from overcomplete import SAE as OvercompleteSAE
from mi_crow.hooks.hook import HookType, HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT
from mi_crow.mechanistic.sae.sae import Sae
from mi_crow.mechanistic.sae.sae_trainer import SaeTrainingConfig
from mi_crow.store.store import Store
from mi_crow.utils import get_logger

logger = get_logger(__name__)


class L1SaeTrainingConfig(SaeTrainingConfig):
    """Training configuration for L1 SAE models.
    
    This class extends SaeTrainingConfig to provide a type-safe configuration
    interface specifically for L1 SAE models. While it currently uses the same
    training parameters as the base SaeTrainingConfig, this design allows for:
    
    1. **Type Safety**: Ensures that L1-specific training methods receive the
       correct configuration type, preventing accidental use of incompatible configs.
    
    2. **Future Extensibility**: Provides a clear extension point for L1-specific
       training parameters that may be needed in the future (e.g., L1 regularization
       scheduling, sparsity target parameters, etc.).
    
    3. **API Clarity**: Makes the intent explicit in the codebase - when you see
       L1SaeTrainingConfig, you know it's specifically for L1 SAE training.
    
    For now, you can use this class exactly like SaeTrainingConfig. All parameters
    from SaeTrainingConfig are available and work identically.
    """
    pass


class L1Sae(Sae):
    """L1 Sparse Autoencoder implementation.
    
    Uses L1 regularization to enforce sparsity in the latent activations.
    This implementation uses the base SAE class from the overcomplete library.
    """
    
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
        super().__init__(n_latents, n_inputs, hook_id, device, store, *args, **kwargs)

    def _initialize_sae_engine(self) -> OvercompleteSAE:
        """Initialize the SAE engine.
        
        Returns:
            OvercompleteSAE instance configured for L1 regularization
        """
        return OvercompleteSAE(
            input_shape=self.context.n_inputs,
            nb_concepts=self.context.n_latents,
            device=self.context.device
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using sae_engine.
        
        Args:
            x: Input tensor of shape [batch_size, n_inputs]
            
        Returns:
            Encoded latents (L1 sparse activations)
        """
        # Overcomplete SAE encode returns (pre_codes, codes)
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
        # Overcomplete SAE forward returns (pre_codes, codes, x_reconstructed)
        _, _, x_reconstructed = self.sae_engine.forward(x)
        return x_reconstructed

    def train(
            self,
            store: Store,
            run_id: str,
            layer_signature: str | int,
            config: L1SaeTrainingConfig | None = None,
            training_run_id: str | None = None
    ) -> dict[str, Any]:
        """
        Train L1SAE using activations from a Store.
        
        This method delegates to the SaeTrainer composite class.
        
        Args:
            store: Store instance containing activations
            run_id: Run ID to train on
            layer_signature: Layer signature to train on
            config: Training configuration
            training_run_id: Optional training run ID
            
        Returns:
            Dictionary with keys:
                - "history": Training history dictionary
                - "training_run_id": Training run ID where outputs were saved
        """
        if config is None:
            config = L1SaeTrainingConfig()
        return self.trainer.train(store, run_id, layer_signature, config, training_run_id)

    def modify_activations(
            self,
            module: "torch.nn.Module",
            inputs: torch.Tensor | None,
            output: torch.Tensor | None
    ) -> torch.Tensor | None:
        """
        Modify activations using L1SAE (Controller hook interface).
        
        Extracts tensor from inputs/output, applies SAE forward pass,
        and optionally applies concept manipulation.
        
        Args:
            module: The PyTorch module being hooked
            inputs: Tuple of inputs to the module
            output: Output from the module (None for pre_forward hooks)
            
        Returns:
            Modified activations with same shape as input
        """
        # Extract tensor from output/inputs, handling objects with last_hidden_state
        if self.hook_type == HookType.FORWARD:
            if isinstance(output, torch.Tensor):
                tensor = output
            elif hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
                tensor = output.last_hidden_state
            elif isinstance(output, (tuple, list)):
                # Try to find first tensor in tuple/list
                tensor = next((item for item in output if isinstance(item, torch.Tensor)), None)
            else:
                tensor = None
        else:
            tensor = inputs[0] if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor) else None

        if tensor is None or not isinstance(tensor, torch.Tensor):
            return output if self.hook_type == HookType.FORWARD else inputs

        original_shape = tensor.shape

        # Flatten to 2D for SAE processing: (batch, seq_len, hidden) -> (batch * seq_len, hidden)
        # or keep as 2D if already 2D: (batch, hidden)
        if len(original_shape) > 2:
            batch_size, seq_len = original_shape[:2]
            tensor_flat = tensor.reshape(-1, original_shape[-1])
        else:
            batch_size = original_shape[0]
            seq_len = 1
            tensor_flat = tensor

        # Get full activations (pre_codes) and sparse codes
        # Overcomplete SAE encode returns (pre_codes, codes)
        pre_codes, codes = self.sae_engine.encode(tensor_flat)
        
        # Save SAE activations (pre_codes) as 3D tensor: (batch, seq, n_latents)
        latents_cpu = pre_codes.detach().cpu()
        latents_3d = latents_cpu.reshape(batch_size, seq_len, -1)
        
        # Save to tensor_metadata
        self.tensor_metadata['neurons'] = latents_3d
        self.tensor_metadata['activations'] = latents_3d
        
        # Process each item in the batch individually for metadata
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

        # Use sparse codes for reconstruction
        latents = codes

        # Update top texts if text tracking is enabled
        if self._text_tracking_enabled and self.context.lm is not None:
            input_tracker = self.context.lm.get_input_tracker()
            if input_tracker is not None:
                texts = input_tracker.get_current_texts()
                if texts:
                    # Use pre_codes (full activations) for text tracking
                    self.concepts.update_top_texts_from_latents(
                        latents_cpu,
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

        # Reshape back to original shape
        if len(original_shape) > 2:
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

        mi_crow_metadata = {
            "concepts_state": {
                'multiplication': self.concepts.multiplication.data,
                'bias': self.concepts.bias.data,
            },
            "n_latents": self.context.n_latents,
            "n_inputs": self.context.n_inputs,
            "device": self.context.device,
            "layer_signature": self.context.lm_layer_signature,
            "model_id": self.context.model_id,
        }

        payload = {
            "sae_state_dict": sae_state_dict,
            "mi_crow_metadata": mi_crow_metadata,
        }

        torch.save(payload, save_path)
        logger.info(f"Saved L1SAE to {save_path}")

    @classmethod
    def load(cls, path: Path) -> "L1Sae":
        """
        Load L1SAE from saved file using overcomplete's load method + our metadata.
        
        Args:
            path: Path to saved model file
            
        Returns:
            Loaded L1Sae instance
        """
        p = Path(path)

        # Load payload
        if torch.cuda.is_available():
            map_location = 'cuda'
        elif torch.backends.mps.is_available():
            map_location = 'mps'
        else:
            map_location = 'cpu'
        payload = torch.load(p, map_location=map_location)

        # Extract our metadata
        if "mi_crow_metadata" not in payload:
            raise ValueError(f"Invalid L1SAE save format: missing 'mi_crow_metadata' key in {p}")

        mi_crow_meta = payload["mi_crow_metadata"]
        n_latents = int(mi_crow_meta["n_latents"])
        n_inputs = int(mi_crow_meta["n_inputs"])
        device = mi_crow_meta.get("device", "cpu")
        layer_signature = mi_crow_meta.get("layer_signature")
        model_id = mi_crow_meta.get("model_id")
        concepts_state = mi_crow_meta.get("concepts_state", {})

        # Create L1Sae instance
        l1_sae = L1Sae(
            n_latents=n_latents,
            n_inputs=n_inputs,
            device=device
        )

        # Load overcomplete model state dict
        if "sae_state_dict" in payload:
            l1_sae.sae_engine.load_state_dict(payload["sae_state_dict"])
        elif "model" in payload:
            # Backward compatibility with old format
            l1_sae.sae_engine.load_state_dict(payload["model"])
        else:
            # Assume payload is the state dict itself (backward compatibility)
            l1_sae.sae_engine.load_state_dict(payload)

        # Load concepts state
        if concepts_state:
            device = l1_sae.context.device
            if isinstance(device, str):
                device = torch.device(device)
            if "multiplication" in concepts_state:
                l1_sae.concepts.multiplication.data = concepts_state["multiplication"].to(device)
            if "bias" in concepts_state:
                l1_sae.concepts.bias.data = concepts_state["bias"].to(device)

        # Note: Top texts loading was removed as serialization methods were removed
        # Top texts should be exported/imported separately if needed

        # Set context metadata
        l1_sae.context.lm_layer_signature = layer_signature
        l1_sae.context.model_id = model_id

        params_str = f"n_latents={n_latents}, n_inputs={n_inputs}"
        logger.info(f"\nLoaded L1SAE from {p}\n{params_str}")

        return l1_sae

    def process_activations(
            self,
            module: torch.nn.Module,
            input: HOOK_FUNCTION_INPUT,
            output: HOOK_FUNCTION_OUTPUT
    ) -> None:
        """
        Process activations (Detector interface).
        
        Metadata saving is handled in modify_activations to avoid duplicate work.
        This method is kept for interface compatibility but does nothing since
        modify_activations already saves the metadata when called.
        
        Args:
            module: The PyTorch module being hooked
            input: Tuple of input tensors to the module
            output: Output tensor(s) from the module
        """
        # Metadata saving is done in modify_activations to avoid duplicate encoding
        pass

