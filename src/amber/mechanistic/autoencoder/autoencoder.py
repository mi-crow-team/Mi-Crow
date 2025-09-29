from pathlib import Path
from typing import Any

import torch
from torch import nn

from amber.mechanistic.autoencoder.modules.modules_list import get_activation
from amber.mechanistic.autoencoder.modules.topk import TopK
from amber.mechanistic.autoencoder.sae_module import SaeModuleABC
from amber.utils import get_logger
logger = get_logger(__name__)


class Autoencoder(nn.Module):
    def __init__(
            self,
            n_latents: int,
            n_inputs: int,
            activation: str | nn.Module = nn.ReLU(),
            tied: bool = False,
            bias_init: torch.Tensor | float = 0.0,
            init_method: str = "kaiming",
            device: str = 'cpu',
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(activation, str):
            activation = get_activation(activation)

        self.tied = tied
        self.n_latents = n_latents
        self.n_inputs = n_inputs
        self.init_method = init_method
        self.bias_init = bias_init
        self.activation = activation
        self.device = device

        self.pre_bias = nn.Parameter(
            torch.full((n_inputs,), bias_init) if isinstance(bias_init, float) else bias_init.clone()
        )
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

    @torch.no_grad()
    def _init_weights(self, norm=0.1, neuron_indices: list[int] | None = None) -> None:
        """
        Initialize weights for encoder and decoder.

        Args:
            norm: Target norm for initialized weights
            neuron_indices: Indices of neurons to reinitialize (None for all)

        Raises:
            ValueError: If init_method is invalid
        """
        valid_methods = ["kaiming", "xavier", "uniform", "normal"]
        if self.init_method not in valid_methods:
            raise ValueError(f"Invalid init_method: {self.init_method}. Choose from: {valid_methods}")

        # Get decoder reference (either tied to encoder or separate)
        if self.tied:
            decoder_weight = self.encoder.t()
        else:
            decoder_weight = self.decoder

        # Create new weights with requested initialization
        new_W_dec = torch.zeros_like(decoder_weight)

        if self.init_method == "kaiming":
            new_W_dec = nn.init.kaiming_uniform_(new_W_dec, nonlinearity='relu')
        elif self.init_method == "xavier":
            new_W_dec = nn.init.xavier_uniform_(new_W_dec, gain=nn.init.calculate_gain('relu'))
        elif self.init_method == "uniform":
            new_W_dec = nn.init.uniform_(new_W_dec, a=-1, b=1)
        elif self.init_method == "normal":
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
            if not self.tied:
                self.decoder.data = new_W_dec
            self.encoder.data = new_W_enc
            self.latent_bias.data = new_l_bias
        else:
            # Update only specified neurons
            if not self.tied:
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
        if self.tied:
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
        if self.tied:
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

        if self.tied:
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
        if self.tied:
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

    def save(self, name: str, path: str | None = None):
        if path is None:
            if isinstance(self.activation, SaeModuleABC):
                path = self.activation.default_model_path
            else:
                path = Path("./models/relu")
        torch.save(self.state_dict(), f"{path}/{name}.pt")

    def load(self, name: str, path: str | None = None):
        if path is None:
            if isinstance(self.activation, SaeModuleABC):
                path = self.activation.default_model_path
            else:
                path = Path("./models/relu")
        state_dict = torch.load(f"{path}/{name}.pt", map_location=self.device)
        self.load_state_dict(state_dict)

    @staticmethod
    def load_model(path: str) -> tuple["Autoencoder", bool, bool, torch.Tensor]:
        """
        Load a saved autoencoder model from a path, inferring configuration from filename.

        Args:
            path: Path to saved model file

        Returns:
            - Loaded model
            - Whether data was mean-centered
            - Whether data was normalized
            - Scaling factor
        """
        # Extract model configuration from filename
        path_head = path.split("/")[-1]
        path_name = path_head[:path_head.find(".pt")]
        path_name_split = path_name.split("_")

        n_latents = int(path_name_split.pop(0))
        n_inputs = int(path_name_split.pop(0))
        activation = path_name_split.pop(0)
        if "JumpReLU" in activation or "TopK" in activation:
            activation += "_" + path_name_split.pop(0)
        tied = True if "True" == path_name_split.pop(0) else False

        model = Autoencoder(
            n_latents,
            n_inputs,
            activation,
            tied=tied,
        )

        # Load state dictionary
        model_state_dict = torch.load(
            path,
            map_location='cuda' if torch.cuda.is_available() else 'cpu'
        )
        model.load_state_dict(model_state_dict['model'])
        dataset_normalize = model_state_dict['dataset_normalize']
        dataset_target_norm = model_state_dict['dataset_target_norm']
        dataset_mean = model_state_dict['dataset_mean']

        params_str = f"n_latents={n_latents}, n_inputs={n_inputs}, activation={activation}, tied={tied}"
        logger.info(f"\nLoaded model from {path}\n{params_str}")

        return model, dataset_normalize, dataset_target_norm, dataset_mean
