from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from overcomplete import (
    TopKSAE as OvercompleteTopkSAE,
    SAE as OvercompleteSAE
)
from mi_crow.hooks.hook import HookType, HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT
from mi_crow.mechanistic.sae.sae import Sae
from mi_crow.mechanistic.sae.sae_trainer import SaeTrainingConfig
from mi_crow.store.store import Store
from mi_crow.utils import get_logger

logger = get_logger(__name__)


@dataclass
class TopKSaeTrainingConfig(SaeTrainingConfig):
    """Training configuration for TopK SAE models.
    
    This class extends SaeTrainingConfig to provide a type-safe configuration
    interface specifically for TopK SAE models. It adds the `k` parameter which
    specifies how many top activations to keep during encoding.
    
    Args:
        k: Number of top activations to keep (required for TopK SAE training)
    
    Note:
        All other parameters are inherited from SaeTrainingConfig.
    
    Attributes:
        k: Number of top activations to keep during TopK encoding
    
    Example:
        >>> config = TopKSaeTrainingConfig(
        ...     k=10,
        ...     epochs=100,
        ...     batch_size=1024,
        ...     lr=1e-3,
        ...     l1_lambda=1e-4
        ... )
    """
    k: int = 10


class TopKSae(Sae):
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
        """
        Initialize TopK SAE.
        
        Args:
            n_latents: Number of latent dimensions (concepts)
            n_inputs: Number of input dimensions
            hook_id: Optional hook identifier
            device: Device to run on ('cpu', 'cuda', 'mps')
            store: Optional store instance
            
        Note:
            The `k` parameter must be provided in TopKSaeTrainingConfig during training.
            For loaded models, `k` is restored from saved metadata.
            A temporary default k=1 is used for engine initialization and will be
            overridden with the actual k value from config during training.
        """
        super().__init__(n_latents, n_inputs, hook_id, device, store, *args, **kwargs)

    def _initialize_sae_engine(self, k: int = 1) -> OvercompleteSAE:
        """
        Initialize the SAE engine with the specified k value.
        
        Args:
            k: Number of top activations to keep (default: 1 for initialization)
        
        Note:
            k should be set from TopKSaeTrainingConfig during training.
            For loaded models, k is restored from saved metadata.
        """
        return OvercompleteTopkSAE(
            input_shape=self.context.n_inputs,
            nb_concepts=self.context.n_latents,
            top_k=k,
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
            config: TopKSaeTrainingConfig | None = None,
            training_run_id: str | None = None
    ) -> dict[str, Any]:
        """
        Train TopKSAE using activations from a Store.
        
        This method delegates to the SaeTrainer composite class.
        The SAE engine will be reinitialized with the k value from config.
        
        Args:
            store: Store instance containing activations
            run_id: Run ID to train on
            config: Training configuration (must include k parameter)
            training_run_id: Optional training run ID
            
        Returns:
            Dictionary with keys:
                - "history": Training history dictionary
                - "training_run_id": Training run ID where outputs were saved
                
        Raises:
            ValueError: If config is None or config.k is not set
        """
        if config is None:
            config = TopKSaeTrainingConfig()
        
        # Ensure k is set in config
        if not hasattr(config, 'k') or config.k is None:
            raise ValueError(
                "TopKSaeTrainingConfig must have k parameter set. "
                "Example: TopKSaeTrainingConfig(k=10, epochs=100, ...)"
            )
        
        # Reinitialize engine with k from config
        logger.info(f"Initializing SAE engine with k={config.k}")
        self.sae_engine = self._initialize_sae_engine(k=config.k)
        if hasattr(config, 'device') and config.device:
            device = torch.device(config.device)
            self.sae_engine.to(device)
            self.context.device = str(device)
        
        return self.trainer.train(store, run_id, layer_signature, config, training_run_id)

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
        # Overcomplete TopKSAE encode returns (pre_codes, codes)
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

    def save(self, name: str, path: str | Path | None = None, k: int | None = None) -> None:
        """
        Save model using overcomplete's state dict + our metadata.
        
        Args:
            name: Model name
            path: Directory path to save to (defaults to current directory)
            k: Top-K value to save (if None, attempts to get from engine or raises error)
        """
        if path is None:
            path = Path.cwd()
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{name}.pt"

        # Save overcomplete model state dict
        sae_state_dict = self.sae_engine.state_dict()

        # Get k value - prefer parameter, then try to get from engine
        if k is None:
            if hasattr(self.sae_engine, 'top_k'):
                k = self.sae_engine.top_k
            else:
                raise ValueError(
                    "k parameter must be provided to save() method. "
                    "The engine does not expose top_k attribute."
                )

        mi_crow_metadata = {
            "concepts_state": {
                'multiplication': self.concepts.multiplication.data,
                'bias': self.concepts.bias.data,
            },
            "n_latents": self.context.n_latents,
            "n_inputs": self.context.n_inputs,
            "k": k,
            "device": self.context.device,
            "layer_signature": self.context.lm_layer_signature,
            "model_id": self.context.model_id,
        }

        payload = {
            "sae_state_dict": sae_state_dict,
            "mi_crow_metadata": mi_crow_metadata,
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
        if torch.cuda.is_available():
            map_location = 'cuda'
        elif torch.backends.mps.is_available():
            map_location = 'mps'
        else:
            map_location = 'cpu'
        payload = torch.load(p, map_location=map_location)

        # Extract our metadata
        if "mi_crow_metadata" not in payload:
            raise ValueError(f"Invalid TopKSAE save format: missing 'mi_crow_metadata' key in {p}")

        mi_crow_meta = payload["mi_crow_metadata"]
        n_latents = int(mi_crow_meta["n_latents"])
        n_inputs = int(mi_crow_meta["n_inputs"])
        k = int(mi_crow_meta["k"])
        device = mi_crow_meta.get("device", "cpu")
        layer_signature = mi_crow_meta.get("layer_signature")
        model_id = mi_crow_meta.get("model_id")
        concepts_state = mi_crow_meta.get("concepts_state", {})

        # Create TopKSAE instance
        topk_sae = TopKSae(
            n_latents=n_latents,
            n_inputs=n_inputs,
            device=device
        )
        
        topk_sae.sae_engine = topk_sae._initialize_sae_engine(k=k)

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
            device = topk_sae.context.device
            if isinstance(device, str):
                device = torch.device(device)
            if "multiplication" in concepts_state:
                topk_sae.concepts.multiplication.data = concepts_state["multiplication"].to(device)
            if "bias" in concepts_state:
                topk_sae.concepts.bias.data = concepts_state["bias"].to(device)

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
