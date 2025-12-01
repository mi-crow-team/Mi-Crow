from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from tqdm.auto import tqdm

from amber.datasets import ClassificationDataset
from amber.hooks.detector import Detector
from amber.hooks.hook import HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT, HookType
from amber.store.store import Store
from amber.utils import get_logger

from .lpm_context import LPMContext

logger = get_logger(__name__)


class LPM(Detector):
    """
    Latent Prototype Moderator (LPM) Detector.

    Implements a training-free moderation method based on latent space prototypes.
    Classifies inputs based on their distance (Euclidean or Mahalanobis) to
    pre-calculated class prototypes (e.g., 'safe' vs 'harmful').
    """

    def __init__(
        self,
        layer_signature: str,
        distance_metric: Literal["euclidean", "mahalanobis"] = "euclidean",
        hook_id: str | None = None,
        device: str = "cpu",
    ):
        super().__init__(hook_type=HookType.FORWARD, hook_id=hook_id, layer_signature=layer_signature)
        self.context = LPMContext(layer_signature=layer_signature, distance_metric=distance_metric, device=device)
        self.aggregation_strategy = "last_token"  # Currently only last_token is supported

    def fit(  # noqa: C901
        self,
        store: Store,
        run_id: str,
        dataset: ClassificationDataset,
        category_field: str = "category",
        batch_size: int = 32,
        max_samples: Optional[int] = None,
    ):
        """
        Calculate prototypes (and covariance) from stored activations.

        Args:
            store: Store containing activations.
            run_id: Run ID of the saved activations.
            dataset: ClassificationDataset with labels.
            category_field: Field name for labels in dataset.
            batch_size: Batch size for processing.
            max_samples: Limit number of samples to use.
        """
        logger.info(f"Starting LPM fit (metric={self.context.distance_metric})...")

        # 1. Collect activations and labels
        # We need to load all relevant activations to compute means and covariance efficiently.
        # For very large datasets, an online algorithm (Welford's) would be better,
        # but for typical LPM use cases (thousands of samples), in-memory is fine.

        activations_by_class: Dict[str, List[torch.Tensor]] = {}
        all_activations: List[torch.Tensor] = []

        # Get batch indices
        batch_indices = store.list_run_batches(run_id)

        current_idx = 0
        samples_processed = 0

        for batch_idx in tqdm(batch_indices, desc="Loading activations"):
            if max_samples is not None and samples_processed >= max_samples:
                break

            # Load batch
            try:
                batch_data = store.get_run_batch(run_id, batch_idx)
            except Exception as e:
                logger.warning(f"Skipping batch {batch_idx}: {e}")
                continue

            # Extract tensor
            if isinstance(batch_data, dict):
                # Try common keys
                if "activations" in batch_data:
                    acts = batch_data["activations"]
                elif self.context.layer_signature in batch_data:
                    acts = batch_data[self.context.layer_signature]
                else:
                    # Fallback: take first tensor value
                    acts = next((v for v in batch_data.values() if isinstance(v, torch.Tensor)), None)
            elif isinstance(batch_data, list):
                # Assume list of tensors [seq_len, hidden]
                acts = torch.stack(batch_data)
            else:
                acts = batch_data

            if acts is None:
                continue

            # Ensure acts is [batch, seq, hidden] or [batch, hidden]
            if isinstance(acts, list):
                acts = torch.stack(acts)

            # Apply aggregation (Last Token)
            # If [batch, seq, hidden] -> [batch, hidden]
            if acts.dim() == 3:
                acts = acts[:, -1, :]

            batch_len = acts.shape[0]

            # Get corresponding labels
            # We assume strict sequential correspondence between store batches and dataset
            batch_labels = []
            for i in range(batch_len):
                if current_idx + i < len(dataset):
                    item = dataset[current_idx + i]
                    batch_labels.append(item[category_field])
                else:
                    break

            if len(batch_labels) != batch_len:
                logger.warning(
                    f"Batch size mismatch at index {current_idx}. Acts: {batch_len}, Labels: {len(batch_labels)}"
                )
                acts = acts[: len(batch_labels)]

            # Group by class
            for i, label in enumerate(batch_labels):
                if label not in activations_by_class:
                    activations_by_class[label] = []

                vec = acts[i].to(self.context.device)
                activations_by_class[label].append(vec)

                if self.context.distance_metric == "mahalanobis":
                    all_activations.append(vec)

            current_idx += batch_len
            samples_processed += batch_len

        # 2. Calculate Prototypes (Means)
        logger.info("Calculating prototypes...")
        for label, vectors in activations_by_class.items():
            if not vectors:
                continue
            stacked = torch.stack(vectors)
            prototype = torch.mean(stacked, dim=0)
            self.context.prototypes[label] = prototype
            logger.info(f"  Class '{label}': {len(vectors)} samples")

        # 3. Calculate Covariance (if Mahalanobis)
        if self.context.distance_metric == "mahalanobis":
            logger.info("Calculating precision matrix (inverse covariance)...")
            if not all_activations:
                raise ValueError("No activations found for covariance calculation")

            # Stack all data [N, hidden]
            X = torch.stack(all_activations)
            N, D = X.shape

            # Calculate centered covariance
            # We assume a shared covariance matrix across classes (as per LPM paper)
            # But strictly speaking, we should center each class by its own mean first?
            # The paper says: "In most experiments we assume all classes share the same covariance matrix...
            # \hat{\Sigma} is the empirical covariance matrix."
            # Usually, for LDA/GDA with shared covariance, we compute the within-class scatter matrix.
            # Let's compute the pooled covariance matrix:
            # \Sigma = (1 / (N - K)) * \sum_c \sum_i (x_i^c - \mu_c)(x_i^c - \mu_c)^T

            X_centered_list = []
            for label, vectors in activations_by_class.items():
                mu = self.context.prototypes[label]
                vecs = torch.stack(vectors)
                X_centered_list.append(vecs - mu)

            X_centered = torch.cat(X_centered_list, dim=0)

            # Empirical Covariance
            # cov = (X_centered.T @ X_centered) / (N - 1)
            # Using torch.cov is easier but expects (Variables, Observations)
            cov = torch.cov(X_centered.T)

            # Regularization (shrinkage) is often needed for high dimensions to ensure invertibility
            # \hat{\Sigma}_{reg} = (1 - \alpha) \hat{\Sigma} + \alpha I
            # Or simply adding a small epsilon to diagonal
            epsilon = 1e-5
            cov = cov + torch.eye(D, device=cov.device) * epsilon

            # Inverse
            try:
                precision = torch.linalg.inv(cov)
                self.context.precision_matrix = precision
            except RuntimeError as e:
                logger.error(f"Failed to invert covariance matrix: {e}")
                raise

        logger.info("LPM fit complete.")

    def process_activations(
        self, module: torch.nn.Module, input: HOOK_FUNCTION_INPUT, output: HOOK_FUNCTION_OUTPUT
    ) -> None:
        """
        Inference hook: Calculate distances to prototypes for the current batch.
        """
        # Extract tensor
        # Similar logic to TopKSae.modify_activations but we don't modify, just read
        if isinstance(output, torch.Tensor):
            tensor = output
        elif isinstance(output, (tuple, list)) and len(output) > 0:
            tensor = output[0]
        elif hasattr(output, "last_hidden_state"):
            tensor = output.last_hidden_state
        else:
            return  # Cannot process

        if not isinstance(tensor, torch.Tensor):
            return

        # Aggregation: Last Token
        # Tensor shape: [batch, seq, hidden]
        if tensor.dim() == 3:
            # Take last token
            # Note: This assumes simple padding or no padding.
            # For correct handling of padded sequences, we would need attention mask.
            # For now, we assume the last element in dim 1 is the last token.
            vectors = tensor[:, -1, :]
        elif tensor.dim() == 2:
            vectors = tensor
        else:
            return

        vectors = vectors.to(self.context.device)

        # Calculate distances
        results = self._calculate_distances(vectors)

        # Save to metadata
        # We accumulate results in self.metadata
        # Structure: list of dicts per batch? Or just append?
        # Detector.metadata is Dict[str, Any].

        if "predictions" not in self.metadata:
            self.metadata["predictions"] = []

        # Convert to python objects for metadata
        batch_results = []
        for i in range(vectors.shape[0]):
            res = {k: v[i].item() if isinstance(v[i], torch.Tensor) else v[i] for k, v in results.items()}
            batch_results.append(res)

        self.metadata["predictions"].extend(batch_results)

    def _calculate_distances(self, vectors: torch.Tensor) -> Dict[str, Any]:
        """
        Calculate distances to all prototypes.
        Returns dict with scores and predicted class.
        """
        distances = {}

        for label, prototype in self.context.prototypes.items():
            prototype = prototype.to(vectors.device)
            diff = vectors - prototype  # [batch, hidden]

            if self.context.distance_metric == "euclidean":
                # L2 norm
                dist = torch.norm(diff, p=2, dim=1)  # [batch]

            elif self.context.distance_metric == "mahalanobis":
                if self.context.precision_matrix is None:
                    raise RuntimeError("Precision matrix not initialized for Mahalanobis distance")

                P = self.context.precision_matrix.to(vectors.device)
                # d = sqrt( (x-mu)^T P (x-mu) )
                # Batch wise: diag( diff @ P @ diff.T )
                # Optimized: sum( (diff @ P) * diff, dim=1 )

                left = torch.matmul(diff, P)  # [batch, hidden]
                dist_sq = torch.sum(left * diff, dim=1)  # [batch]
                dist = torch.sqrt(torch.clamp(dist_sq, min=1e-10))

            distances[label] = dist

        # Determine predicted class (min distance)
        # Stack distances: [batch, num_classes]
        labels = list(distances.keys())
        dist_stack = torch.stack([distances[label] for label in labels], dim=1)

        min_indices = torch.argmin(dist_stack, dim=1)
        predicted_labels = [labels[idx] for idx in min_indices]

        # Prepare output
        output = {
            "predicted_label": predicted_labels,
        }
        for label in labels:
            output[f"distance_{label}"] = distances[label]

        return output

    def save(self, path: Union[str, Path]):
        """Save LPM model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "prototypes": self.context.prototypes,
            "precision_matrix": self.context.precision_matrix,
            "config": {
                "layer_signature": self.context.layer_signature,
                "distance_metric": self.context.distance_metric,
                "model_id": self.context.model_id,
                "aggregation_strategy": self.aggregation_strategy,
            },
        }
        torch.save(state, path)
        logger.info(f"Saved LPM model to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "LPM":
        """Load LPM model."""
        path = Path(path)
        state = torch.load(path, map_location=device)

        config = state["config"]
        lpm = cls(layer_signature=config["layer_signature"], distance_metric=config["distance_metric"], device=device)
        lpm.context.model_id = config.get("model_id")
        lpm.aggregation_strategy = config.get("aggregation_strategy", "last_token")

        lpm.context.prototypes = state["prototypes"]
        lpm.context.precision_matrix = state["precision_matrix"]

        # Move to device
        lpm.context.to(device)

        return lpm
