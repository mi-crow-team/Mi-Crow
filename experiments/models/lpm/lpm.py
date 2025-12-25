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
from experiments.predictors.predictor import Predictor

from .lpm_context import LPMContext

logger = get_logger(__name__)


class LPM(Detector, Predictor):
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
        positive_label: str = "harmful",
    ):
        Detector.__init__(self, hook_type=HookType.FORWARD, hook_id=hook_id, layer_signature=layer_signature)
        Predictor.__init__(
            self,
            model_id=f"lpm_{distance_metric}_{layer_signature}",
            config={
                "layer_signature": layer_signature,
                "distance_metric": distance_metric,
                "device": device,
                "positive_label": positive_label,
            },
        )
        self.lpm_context = LPMContext(layer_signature=layer_signature, distance_metric=distance_metric, device=device)
        self.aggregation_strategy = "last_token"  # Currently only last_token is supported
        self.positive_label = positive_label
        self._seen_samples = 0

    def clear_predictions(self) -> None:
        super().clear_predictions()
        self._seen_samples = 0

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
        logger.info(f"Starting LPM fit (metric={self.lpm_context.distance_metric})...")

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
                elif self.lpm_context.layer_signature in batch_data:
                    acts = batch_data[self.lpm_context.layer_signature]
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

                vec = acts[i].to(self.lpm_context.device)
                activations_by_class[label].append(vec)

                if self.lpm_context.distance_metric == "mahalanobis":
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
            self.lpm_context.prototypes[label] = prototype
            logger.info(f"  Class '{label}': {len(vectors)} samples")

        # 3. Calculate Covariance (if Mahalanobis)
        if self.lpm_context.distance_metric == "mahalanobis":
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
                mu = self.lpm_context.prototypes[label]
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
                self.lpm_context.precision_matrix = precision
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

        vectors = vectors.to(self.lpm_context.device)

        # Calculate distances
        results = self._calculate_distances(vectors)

        # Convert to python objects for Predictor accumulation.
        batch_results: List[Dict[str, Any]] = []
        for i in range(vectors.shape[0]):
            predicted_class = results["predicted_class"][i]
            predicted_label = 1 if str(predicted_class).lower() == str(self.positive_label).lower() else 0

            res: Dict[str, Any] = {
                "sample_index": self._seen_samples + i,
                "predicted_label": predicted_label,
                "predicted_class": predicted_class,
            }

            for k, v in results.items():
                if k == "predicted_class":
                    continue
                res[k] = v[i].item() if isinstance(v[i], torch.Tensor) else v[i]

            batch_results.append(res)

        self._seen_samples += len(batch_results)

        # Keep detector metadata lightweight: overwrite per-batch for debugging/compatibility.
        self.metadata["predictions"] = batch_results

        # Standardized accumulation for final save.
        self.add_predictions(batch_results)

    def _calculate_distances(self, vectors: torch.Tensor) -> Dict[str, Any]:
        """
        Calculate distances to all prototypes.
        Returns dict with scores and predicted class.
        """
        distances = {}

        for label, prototype in self.lpm_context.prototypes.items():
            prototype = prototype.to(vectors.device)
            diff = vectors - prototype  # [batch, hidden]

            if self.lpm_context.distance_metric == "euclidean":
                # L2 norm
                dist = torch.norm(diff, p=2, dim=1)  # [batch]

            elif self.lpm_context.distance_metric == "mahalanobis":
                if self.lpm_context.precision_matrix is None:
                    raise RuntimeError("Precision matrix not initialized for Mahalanobis distance")

                P = self.lpm_context.precision_matrix.to(vectors.device)
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
        predicted_classes = [labels[idx] for idx in min_indices]

        # Prepare output
        output = {
            "predicted_class": predicted_classes,
        }
        for label in labels:
            output[f"distance_{label}"] = distances[label]

        return output

    def get_config(self) -> Dict[str, Any]:
        return dict(self.config)

    def save(self, path: Union[str, Path]):
        """Save LPM model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "prototypes": self.lpm_context.prototypes,
            "precision_matrix": self.lpm_context.precision_matrix,
            "config": {
                "layer_signature": self.lpm_context.layer_signature,
                "distance_metric": self.lpm_context.distance_metric,
                "model_id": self.lpm_context.model_id,
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
        lpm.lpm_context.model_id = config.get("model_id")
        lpm.aggregation_strategy = config.get("aggregation_strategy", "last_token")

        lpm.lpm_context.prototypes = state["prototypes"]
        lpm.lpm_context.precision_matrix = state["precision_matrix"]

        # Move to device
        lpm.lpm_context.to(device)

        return lpm
