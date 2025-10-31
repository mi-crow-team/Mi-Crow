from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from torch import nn

from amber.mechanistic.autoencoder.concepts.autoencoder_concepts import AutoencoderConcepts
from amber.mechanistic.autoencoder.modules.modules_list import get_activation
from amber.mechanistic.autoencoder.modules.topk import TopK
from amber.mechanistic.autoencoder.sae_module import SaeModuleABC
from amber.mechanistic.autoencoder.autoencoder_context import AutoencoderContext
from amber.utils import get_logger

if TYPE_CHECKING:
    from amber.core.language_model import LanguageModel

logger = get_logger(__name__)


class Autoencoder(nn.Module):
    def __init__(
            self,
            n_latents: int,
            n_inputs: int,
            activation: str | nn.Module = nn.ReLU(),
            tied: bool = False,
            bias_init: float = 0.0,
            init_method: str = "kaiming",
            device: str = 'cpu',
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.context = AutoencoderContext(
            autoencoder=self,
            n_latents=n_latents,
            n_inputs=n_inputs
        )
        self.context.tied = tied
        self.context.bias_init = bias_init
        self.context.init_method = init_method
        self.context.device = device

        # Initialize components using context
        if isinstance(activation, str):
            activation = get_activation(activation)

        self.activation = activation
        self.metadata = None

        # Create concepts with context
        self.concepts = AutoencoderConcepts(self.context)

        self.pre_bias = nn.Parameter(torch.full((n_inputs,), bias_init))
        self.encoder = nn.Parameter(torch.zeros((n_latents, n_inputs)).t())
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))

        if tied:
            self.register_parameter('decoder', None)
        else:
            self.decoder = nn.Parameter(torch.zeros((n_latents, n_inputs)))

        self._init_weights()

        self.register_buffer(
            "latents_activation_frequency", torch.zeros(n_latents, dtype=torch.int64, requires_grad=False)
        )
        self.num_updates = 0
        self.dead_latents: list[int] = []

    def enable_text_tracking(self, k: int = 5, negative: bool = False):
        self.context.text_tracking_enabled = True
        self.context.text_tracking_k = k
        self.context.text_tracking_negative = negative
        self.concepts.enable_text_tracking()

    def disable_text_tracking(self):
        """Disable text tracking through context."""
        self.context.text_tracking_enabled = False
        if self.concepts.top_texts_tracker is not None:
            self.concepts.top_texts_tracker.detach()
            self.concepts.top_texts_tracker = None

    @torch.no_grad()
    def _init_weights(self, norm=0.1, neuron_indices: list[int] | None = None) -> None:
        valid_methods = ["kaiming", "xavier", "uniform", "normal"]
        if self.context.init_method not in valid_methods:
            raise ValueError(f"Invalid init_method: {self.context.init_method}. Choose from: {valid_methods}")

        # Get decoder reference (either tied to encoder or separate)
        if self.context.tied:
            decoder_weight = self.encoder.t()
        else:
            decoder_weight = self.decoder

        # Create new weights with requested initialization
        new_W_dec = torch.zeros_like(decoder_weight)

        if self.context.init_method == "kaiming":
            new_W_dec = nn.init.kaiming_uniform_(new_W_dec, nonlinearity='relu')
        elif self.context.init_method == "xavier":
            new_W_dec = nn.init.xavier_uniform_(new_W_dec, gain=nn.init.calculate_gain('relu'))
        elif self.context.init_method == "uniform":
            new_W_dec = nn.init.uniform_(new_W_dec, a=-1, b=1)
        elif self.context.init_method == "normal":
            new_W_dec = nn.init.normal_(new_W_dec)

        # Scale to target norm
        new_W_dec *= (norm / new_W_dec.norm(p=2, dim=-1, keepdim=True))

        # Create new latent biases (zeros)
        new_l_bias = torch.zeros_like(self.latent_bias)

        # Create encoder weights (transposed decoder weights)
        new_W_enc = new_W_dec.t().clone()

        # Update parameters, either all or only specified indices
        if neuron_indices is None:
            # Update all weights
            if not self.context.tied:
                self.decoder.data = new_W_dec
            self.encoder.data = new_W_enc
            self.latent_bias.data = new_l_bias
        else:
            # Update only specified neurons
            if not self.context.tied:
                self.decoder.data[neuron_indices] = new_W_dec[neuron_indices]
            self.encoder.data[:, neuron_indices] = new_W_enc[:, neuron_indices]
            self.latent_bias.data[neuron_indices] = new_l_bias[neuron_indices]

    @torch.no_grad()
    def project_grads_decode(self) -> None:
        """
        Project decoder gradients to enforce constraints.

        This ensures that each latent dimension's decoder weights
        maintain a specific norm during training.
        """
        if self.context.tied:
            weights = self.encoder.data.T
            grad = self.encoder.grad.T
        else:
            weights = self.decoder.data
            grad = self.decoder.grad

        if grad is None:
            return

        # Project gradients to maintain unit norm constraint
        # Calculate component of gradient parallel to weights
        grad_proj = (grad * weights).sum(dim=-1, keepdim=True) * weights

        # Subtract parallel component from gradients
        if self.context.tied:
            self.encoder.grad -= grad_proj.T
        else:
            self.decoder.grad -= grad_proj

    @torch.no_grad()
    def scale_to_unit_norm(self) -> None:
        """
        Scale decoder weights to have unit norm.

        This enforces a unit norm constraint on each latent dimension's
        decoder weights, which aids in training stability.
        """
        eps = torch.finfo(self.encoder.dtype).eps

        if self.context.tied:
            # For tied weights, normalize encoder's columns without assigning to .T
            # Compute column norms as shape [1, n_latents]
            norm = self.encoder.data.norm(p=2, dim=0, keepdim=True) + eps
            self.encoder.data /= norm
        else:
            # For untied weights, normalize decoder's rows
            norm = self.decoder.data.norm(p=2, dim=-1, keepdim=True) + eps
            self.decoder.data /= norm

            # Compensate in encoder to maintain same reconstruction
            self.encoder.data *= norm.t()

        # Scale latent bias to compensate
        self.latent_bias.data *= norm.squeeze()

    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activation latent values.

        Args:
            x: Input tensor of shape [batch_size, n_inputs]

        Returns:
            Pre-activation latent values
        """
        # Remove bias
        x_unbiased = x - self.pre_bias

        # Compute all latents
        latents_pre_act = x_unbiased @ self.encoder + self.latent_bias
        return latents_pre_act

    def encode(
            self,
            x: torch.Tensor,
            topk_number: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Encode input data to latent representation.

        Args:
            x: Input tensor of shape [batch_size, n_inputs]
            topk_number: Optional number of top activations to keep (for inference)

        Returns:
            - Encoded latents (with activation)
            - Full encoded latents (without TopK sparsity for inference)
            - Preprocessing info
        """

        # Calculate pre-activations
        pre_encoded = self.encode_pre_act(x)

        # Apply activation for training path
        encoded = self.activation(pre_encoded)

        # Calculate full activations for evaluation
        if isinstance(self.activation, TopK):
            # For TopK, use the non-sparse version during evaluation
            full_encoded = self.activation.forward_eval(pre_encoded)
        else:
            # For other activations, full activation is the same
            full_encoded = encoded.clone()

        # Apply TopK sparsity for inference if requested
        if topk_number is not None:
            # Get top-k values and their indices
            _, indices = torch.topk(full_encoded, k=min(topk_number, full_encoded.shape[-1]), dim=-1)
            values = torch.gather(full_encoded, -1, indices)

            # Create sparse output with only top-k values
            full_encoded_sparse = torch.zeros_like(full_encoded)
            full_encoded_sparse.scatter_(-1, indices, values)
            full_encoded = full_encoded_sparse

        # Apply soft capping to prevent exploding values
        # caped_encoded = self.latent_soft_cap(encoded)
        # capped_full_encoded = self.latent_soft_cap(full_encoded)

        return encoded, full_encoded, {}

    def decode(
            self,
            latents: torch.Tensor,
            info: dict[str, Any] | None = None,
            detach: bool = False,
    ) -> torch.Tensor:
        if info is None:
            info = {}

        # Apply decoder weights
        if self.context.tied:
            reconstructed = latents @ self.encoder.t() + self.pre_bias
        else:
            reconstructed = latents @ self.decoder + self.pre_bias

        if detach:
            reconstructed = reconstructed.detach()
        return reconstructed

    @torch.no_grad()
    def update_latent_statistics(self, latents: torch.Tensor) -> None:
        """
        Update activation statistics for latent neurons.

        Args:
            latents: Activated latent tensor of shape [batch_size, n_latents]
        """
        batch_size = latents.shape[0]
        self.num_updates += batch_size

        # Count how many times each neuron was active in this batch
        current_activation_frequency = (latents != 0).to(torch.int64).sum(dim=0)
        self.latents_activation_frequency += current_activation_frequency

    def forward(
            self,
            x: torch.Tensor,
            detach: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape [batch_size, n_inputs]

        Returns:
            - Input tensor (potentially normalized)
            - Latent activations
            - Reconstructed tensor
            - Full latent activations (without TopK sparsity)
        """
        # Encode to latent space
        latents, full_latents, info = self.encode(x)

        # Update activation statistics
        self.update_latent_statistics(latents)

        # Decode back to input space both with and without TopK sparsity
        reconstructed_full = self.decode(full_latents, info)
        reconstructed = self.decode(latents, info)
        # if isinstance(self.activation, JumpReLU) and self.training:
        #     # For JumpReLU, apply custom training path
        #     latents = self.activation.forward_train(latents)
        if detach:
            latents = latents.detach()
            full_latents = full_latents.detach()
            reconstructed = reconstructed.detach()
            reconstructed_full = reconstructed_full.detach()
        return reconstructed, latents, reconstructed_full, full_latents

    def save(
            self,
            name: str,
            path: str | Path,
            *,
            dataset_normalize: bool | None = None,
            dataset_target_norm: Any | None = None,
            dataset_mean: Any | None = None,
            run_metadata: dict[str, Any] | None = None,
    ) -> None:
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{name}.pt"

        state = self.state_dict()

        def _activation_to_str(act: Any) -> str:
            if isinstance(act, str):
                return act
            try:
                from amber.mechanistic.autoencoder.modules.topk import TopK  # local import to avoid cycles
            except Exception:
                TopK = None  # type: ignore
            if TopK is not None and isinstance(act, TopK):
                base = "TopKReLU" if isinstance(act.act_fn, nn.ReLU) else "TopK"
                return f"{base}_{getattr(act, 'k', 0)}"
            return act.__class__.__name__

        activation_name = _activation_to_str(self.activation)

        # Always save a rich payload with core metadata
        payload: dict[str, Any] = {
            "model": state,
            "n_latents": self.context.n_latents,
            "n_inputs": self.context.n_inputs,
            "activation": activation_name,
            "tied": self.context.tied,
            "bias_init": self.context.bias_init,
            "init_method": self.context.init_method,
            "layer_signature": self.context.lm_layer_signature,
            "model_id": self.context.model_id
        }
        if dataset_normalize is not None:
            payload["dataset_normalize"] = dataset_normalize
        if dataset_target_norm is not None:
            payload["dataset_target_norm"] = dataset_target_norm
        if dataset_mean is not None:
            payload["dataset_mean"] = dataset_mean
        if run_metadata is not None:
            payload["run_metadata"] = run_metadata

        torch.save(payload, save_path)

    def load(self, name: str, path: str | Path | None = None):
        if path is None:
            if isinstance(self.activation, SaeModuleABC):
                path = self.activation.default_model_path
            else:
                path = Path("./models/relu")
        load_path = Path(path) / f"{name}.pt"
        payload = torch.load(load_path, map_location=self.context.device)
        # Support both raw state_dict and dict payloads
        if isinstance(payload, dict) and all(k in payload for k in ("model",)):
            self.load_state_dict(payload["model"])
        else:
            self.load_state_dict(payload)

    @staticmethod
    def load_model(path: str | Path) -> tuple["Autoencoder", bool, bool, torch.Tensor]:
        """
        Load a saved autoencoder model from a path. Prefer metadata inside the file; fallback to filename parsing for legacy files.

        Args:
            path: Path to saved model file

        Returns:
            - Loaded model
            - Whether data was mean-centered
            - Whether data was normalized
            - Scaling factor (mean or norm tensor)
        """
        p = Path(path)

        # Load payload (can be full dict or raw state_dict)
        payload = torch.load(
            p,
            map_location='cuda' if torch.cuda.is_available() else 'cpu'
        )

        n_latents = int(payload["n_latents"])  # type: ignore
        n_inputs = int(payload["n_inputs"])  # type: ignore
        activation = payload["activation"]  # type: ignore
        tied = bool(payload["tied"])  # type: ignore

        # Create model with new constructor
        model = Autoencoder(
            n_latents=n_latents,
            n_inputs=n_inputs,
            activation=activation,
            tied=tied
        )

        # Set layer signature if available
        model.load_state_dict(payload["model"])  # type: ignore[arg-type]
        model.context.lm_layer_signature = payload.get('layer_signature')
        dataset_normalize = bool(payload.get("dataset_normalize", False))
        dataset_target_norm = bool(payload.get("dataset_target_norm", False))
        dataset_mean = payload.get("dataset_mean", torch.zeros(n_inputs))  # type: ignore[assignment]
        model.metadata = payload['run_metadata'] if 'run_metadata' in payload else None

        params_str = f"n_latents={n_latents}, n_inputs={n_inputs}, activation={activation}, tied={tied}"
        logger.info(f"\nLoaded model from {p}\n{params_str}")

        return model, dataset_normalize, dataset_target_norm, dataset_mean
