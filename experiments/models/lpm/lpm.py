# ruff: noqa
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from tqdm.auto import tqdm

from experiments.predictors.predictor import Predictor
from experiments.utils.activation_aggregation import aggregate_activations_batch
from mi_crow.datasets import ClassificationDataset
from mi_crow.hooks.detector import Detector
from mi_crow.hooks.hook import HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT, HookType
from mi_crow.store.store import Store
from mi_crow.utils import get_logger

from .lpm_context import LPMContext

logger = get_logger(__name__)


class LPM(Detector, Predictor):
    """
    Latent Prototype Moderator (LPM) Detector.

    Implements a training-free moderation method based on latent space prototypes.
    Classifies inputs based on their distance (Euclidean or Mahalanobis) to
    pre-calculated class prototypes (e.g., 'safe' vs 'harmful').

    The model is fitted on pre-saved activations loaded batch-by-batch to avoid OOM.
    Supports sequence aggregation methods: 'mean', 'last_token', 'last_token_prefix'.
    """

    def __init__(
        self,
        layer_signature: str | None = None,
        layer_number: int | None = None,
        distance_metric: Literal["euclidean", "mahalanobis"] = "euclidean",
        aggregation_method: Literal["mean", "last_token", "last_token_prefix"] = "last_token",
        hook_id: str | None = None,
        device: str = "cpu",
        positive_label: str = "harmful",
    ):
        """
        Initialize LPM detector.

        Args:
            layer_signature: Full layer signature (e.g., "llamaforcausallm_model_layers_27")
            layer_number: Layer number (e.g., 27) - used if layer_signature is None
            distance_metric: "euclidean" or "mahalanobis"
            aggregation_method: How to aggregate sequence dimension ("mean", "last_token", "last_token_prefix")
            hook_id: Optional hook identifier
            device: Device for computations
            positive_label: Label for positive class (e.g., "harmful")
        """
        # Construct layer signature if only layer_number is provided
        if layer_signature is None and layer_number is not None:
            layer_signature = f"llamaforcausallm_model_layers_{layer_number}"
        elif layer_signature is None:
            raise ValueError("Either layer_signature or layer_number must be provided")

        # Extract layer number from signature if not provided
        if layer_number is None and layer_signature is not None:
            try:
                layer_number = int(layer_signature.split("_")[-1])
            except (ValueError, IndexError):
                logger.warning(f"Could not extract layer number from {layer_signature}")
                layer_number = None

        Detector.__init__(self, hook_type=HookType.FORWARD, hook_id=hook_id, layer_signature=layer_signature)
        Predictor.__init__(
            self,
            model_id=f"lpm_{distance_metric}_{aggregation_method}_layer{layer_number}",
            config={
                "layer_signature": layer_signature,
                "layer_number": layer_number,
                "distance_metric": distance_metric,
                "aggregation_method": aggregation_method,
                "device": device,
                "positive_label": positive_label,
            },
        )
        self.lpm_context = LPMContext(
            layer_signature=layer_signature,
            layer_number=layer_number,
            distance_metric=distance_metric,
            aggregation_method=aggregation_method,
            device=device,
        )
        self.positive_label = positive_label
        self._seen_samples = 0
        self._current_batch_idx = 0  # Track actual batch index for attention mask lookup
        self._inference_attention_masks: Optional[Dict[int, torch.Tensor]] = None  # batch_idx -> attention_mask

    def clear_predictions(self) -> None:
        super().clear_predictions()
        self._seen_samples = 0
        self._current_batch_idx = 0

    def _get_aggregation_type(self) -> Literal["last", "mean"]:
        """
        Convert LPM aggregation method to the format expected by aggregate_activations_batch.

        LPM uses:
        - 'last_token': Select last non-special token → maps to 'last'
        - 'last_token_prefix': Same as last_token (prefix is added before tokenization) → maps to 'last'
        - 'mean': Average over non-special tokens → maps to 'mean'
        """
        if self.lpm_context.aggregation_method in ["last_token", "last_token_prefix"]:
            return "last"
        elif self.lpm_context.aggregation_method == "mean":
            return "mean"
        else:
            raise ValueError(f"Unknown aggregation method: {self.lpm_context.aggregation_method}")

    def load_inference_attention_masks(self, store: Store, run_id: str) -> None:
        """
        Load attention masks from a saved run for use during inference.

        This enables proper sequence aggregation during inference by using the actual
        attention masks from the input data. Attention masks should be saved using
        ModelInputDetector with save_attention_mask=True.

        Args:
            store: Store containing the saved attention masks
            run_id: Run ID where attention masks were saved (e.g., from ModelInputDetector)

        Example:
            # Before inference, load attention masks
            lpm.load_inference_attention_masks(store, "test_attention_masks_run")
            # Then run inference normally
            lm.detectors.add_detector(lpm)
            ...
        """
        logger.info(f"Loading attention masks from run: {run_id}")
        self._inference_attention_masks = {}

        batch_indices = store.list_run_batches(run_id)
        logger.info(f"Found {len(batch_indices)} batches with attention masks")

        for batch_idx in batch_indices:
            try:
                # Try to load from input_ids layer (ModelInputDetector format)
                attention_mask = store.get_detector_metadata_by_layer_by_key(
                    run_id, batch_idx, "input_ids", "attention_mask"
                )
                self._inference_attention_masks[batch_idx] = attention_mask
            except Exception:
                # Fallback: try attention_masks layer (old format)
                try:
                    attention_mask = store.get_detector_metadata_by_layer_by_key(
                        run_id, batch_idx, "attention_masks", "attention_mask"
                    )
                    self._inference_attention_masks[batch_idx] = attention_mask
                except Exception as e:
                    logger.warning(f"Could not load attention mask for batch {batch_idx}: {e}")

        logger.info(f"Loaded attention masks for {len(self._inference_attention_masks)} batches")

    def clear_inference_attention_masks(self) -> None:
        """Clear loaded attention masks to free memory."""
        self._inference_attention_masks = None

    def fit(  # noqa: C901
        self,
        store: Store,
        run_id: str,
        dataset: ClassificationDataset,
        model_id: str,
        dataset_name: str = None,
        category_field: str = "category",
        max_samples: Optional[int] = None,
    ):
        """
        Calculate prototypes (and covariance/precision) from stored activations.

        Loads activations batch-by-batch to avoid OOM. For each batch:
        1. Load activations and attention masks
        2. Filter out special tokens using attention masks
        3. Apply sequence aggregation (mean, last_token, etc.)
        4. Accumulate aggregated activations in memory

        Then calculates class prototypes and (optionally) precision matrix.

        Args:
            store: Store containing activations.
            run_id: Run ID of the saved activations.
            dataset: ClassificationDataset with labels.
            model_id: Model ID (e.g., "speakleash/Bielik-1.5B-v3.0-Instruct")
            category_field: Field name for labels in dataset.
            max_samples: Limit number of samples to use.
        """
        logger.info(f"Starting LPM fit (metric={self.lpm_context.distance_metric})...")
        logger.info(f"Aggregation method: {self.lpm_context.aggregation_method}")

        # Store metadata
        self.lpm_context.model_id = model_id
        self.lpm_context.run_id = run_id
        self.lpm_context.dataset_name = (
            dataset_name if dataset_name is not None else getattr(dataset, "name", "unknown")
        )

        # Get batch indices
        batch_indices = store.list_run_batches(run_id)
        logger.info(f"Found {len(batch_indices)} batches in run {run_id}")

        # Prepare to accumulate aggregated activations
        aggregated_activations_by_class: Dict[str, List[torch.Tensor]] = {}
        all_aggregated_activations: List[torch.Tensor] = []
        all_labels: List[str] = []

        current_sample_idx = 0
        samples_processed = 0

        for batch_idx in tqdm(batch_indices, desc="Loading and aggregating activations"):
            if max_samples is not None and samples_processed >= max_samples:
                break

            # Load activations and attention masks
            try:
                # Get activations
                activations = store.get_detector_metadata_by_layer_by_key(
                    run_id, batch_idx, str(self.lpm_context.layer_signature), "activations"
                )
                # Get attention masks
                attention_mask = store.get_detector_metadata_by_layer_by_key(
                    run_id, batch_idx, "attention_masks", "attention_mask"
                )
            except Exception as e:
                logger.warning(f"Failed to load batch {batch_idx}: {e}")
                continue

            # DEBUG NaN means and cov - Detailed NaN analysis
            nan_mask = torch.isnan(activations)
            inf_mask = torch.isinf(activations)

            if nan_mask.any():
                total_elements = activations.numel()
                nan_count = nan_mask.sum().item()
                nan_percentage = (nan_count / total_elements) * 100
                logger.warning(
                    f"Batch {batch_idx}: Raw activations contain {nan_count}/{total_elements} NaNs ({nan_percentage:.2f}%)"
                )

                # Check if entire tensor is NaN
                if nan_count == total_elements:
                    logger.warning(f"Batch {batch_idx}: ENTIRE activation tensor is NaN!")
                else:
                    # Sample some non-NaN values if they exist
                    non_nan_values = activations[~nan_mask]
                    if non_nan_values.numel() > 0:
                        logger.info(f"Batch {batch_idx}: Non-NaN values exist - sample: {non_nan_values[:5].tolist()}")
                        logger.info(
                            f"Batch {batch_idx}: Non-NaN stats - mean: {non_nan_values.mean().item():.4f}, std: {non_nan_values.std().item():.4f}"
                        )

                # Check NaN pattern per sample in batch
                batch_size = activations.shape[0]
                for sample_idx in range(min(3, batch_size)):  # Check first 3 samples
                    sample_nans = nan_mask[sample_idx].sum().item()
                    sample_total = nan_mask[sample_idx].numel()
                    logger.info(f"Batch {batch_idx}, Sample {sample_idx}: {sample_nans}/{sample_total} NaNs")

            if inf_mask.any():
                inf_count = inf_mask.sum().item()
                logger.warning(f"Batch {batch_idx}: Raw activations contain {inf_count} Infs!")

            # Log normal statistics for comparison (even if NaNs present)
            if not nan_mask.all():  # If not all NaN
                logger.info(
                    f"Batch {batch_idx}: Raw activation stats - mean: {activations.nanmean().item():.4f}, std: {activations.std().item():.4f}"
                )

            # activations: [batch_size, seq_len, hidden_dim]
            # attention_mask: [batch_size, seq_len]
            if activations.dim() != 3 or attention_mask.dim() != 2:
                logger.warning(
                    f"Unexpected tensor shapes in batch {batch_idx}: "
                    f"activations={activations.shape}, attention_mask={attention_mask.shape}"
                )
                continue

            batch_size = activations.shape[0]

            # Apply sequence aggregation (filtering is done inside aggregate_activations_batch)
            # Result: [batch_size, hidden_dim]
            # DEBUG NaN means and cov
            logger.info(
                f"Batch {batch_idx}: Before aggregation - activations shape: {activations.shape}, device: {activations.device}"
            )
            logger.info(
                f"Batch {batch_idx}: Before aggregation - attention_mask shape: {attention_mask.shape}, device: {attention_mask.device}"
            )

            try:
                aggregated = aggregate_activations_batch(
                    activations,  # Keep on CPU to save memory
                    attention_mask,
                    agg=self._get_aggregation_type(),
                )
                aggregated = aggregated.cpu()

                # DEBUG NaN means and cov
                logger.info(
                    f"Batch {batch_idx}: After aggregation - shape: {aggregated.shape}, device: {aggregated.device}"
                )
                if torch.isnan(aggregated).any():
                    logger.warning(f"Batch {batch_idx}: Aggregated activations contain NaNs!")
                if torch.isinf(aggregated).any():
                    logger.warning(f"Batch {batch_idx}: Aggregated activations contain Infs!")

                # DEBUG NaN means and cov
                logger.info(
                    f"Batch {batch_idx}: Aggregated stats - mean: {aggregated.mean().item():.4f}, std: {aggregated.std().item():.4f}, min: {aggregated.min().item():.4f}, max: {aggregated.max().item():.4f}"
                )

                # DEBUG NaN means and cov - Detailed inspection of batch 1, sample 0
                if batch_idx == 1 and aggregated.shape[0] > 0:
                    sample_vec = aggregated[0]  # [hidden_dim]
                    logger.info(f"Batch {batch_idx}, Sample 0: First 10 values: {sample_vec[:10].tolist()}")
                    logger.info(
                        f"Batch {batch_idx}, Sample 0: Contains NaN: {torch.isnan(sample_vec).any().item()}, Contains Inf: {torch.isinf(sample_vec).any().item()}"
                    )

            except Exception as e:
                logger.warning(f"Failed to aggregate batch {batch_idx}: {e}")
                continue

            # Store hidden_dim
            if self.lpm_context.hidden_dim is None:
                self.lpm_context.hidden_dim = aggregated.shape[1]

            # Match with dataset labels
            for i in range(batch_size):
                if max_samples is not None and samples_processed >= max_samples:
                    break

                sample_idx = current_sample_idx + i
                if sample_idx >= len(dataset):
                    logger.warning(f"Sample index {sample_idx} out of dataset bounds")
                    break

                # Get label from dataset
                sample_data = dataset[sample_idx]
                if isinstance(sample_data, dict):
                    label = sample_data.get(category_field)
                else:
                    logger.warning(f"Dataset sample {sample_idx} is not a dict")
                    continue

                if label is None:
                    logger.warning(f"No label found for sample {sample_idx}")
                    continue

                # Convert label to string
                label = str(label)

                # Store aggregated activation
                all_aggregated_activations.append(aggregated[i])
                all_labels.append(label)

                # Store by class
                if label not in aggregated_activations_by_class:
                    aggregated_activations_by_class[label] = []
                aggregated_activations_by_class[label].append(aggregated[i])

                samples_processed += 1

            current_sample_idx += batch_size

            # Clear batch tensors to free memory
            del activations
            del attention_mask
            del aggregated
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        logger.info(f"Processed {samples_processed} samples across {len(aggregated_activations_by_class)} classes")

        # Convert to tensors
        all_aggregated_tensor = torch.stack(all_aggregated_activations)  # [N, hidden_dim]
        label_to_idx = {label: idx for idx, label in enumerate(aggregated_activations_by_class.keys())}
        all_labels_tensor = torch.tensor([label_to_idx[label] for label in all_labels])

        # Calculate prototypes (means)
        logger.info("Calculating class prototypes...")
        for label, vectors_list in aggregated_activations_by_class.items():
            vectors_tensor = torch.stack(vectors_list)  # [N_class, hidden_dim]

            # DEBUG NaN means and cov
            logger.info(f"  Class '{label}': Stacked tensor shape: {vectors_tensor.shape}")
            if torch.isnan(vectors_tensor).any():
                logger.warning(f"  Class '{label}': Stacked vectors contain NaNs!")
            if torch.isinf(vectors_tensor).any():
                logger.warning(f"  Class '{label}': Stacked vectors contain Infs!")
            logger.info(
                f"  Class '{label}': Stacked stats - mean: {vectors_tensor.mean().item():.4f}, std: {vectors_tensor.std().item():.4f}"
            )

            mean_vector = vectors_tensor.mean(dim=0)  # [hidden_dim]
            self.lpm_context.prototypes[label] = mean_vector

            # DEBUG NaN means and cov
            logger.info(f"  Class '{label}': {len(vectors_list)} samples, mean shape {mean_vector.shape}")
            logger.info(
                f"  Class '{label}': Prototype stats - mean: {mean_vector.mean().item():.4f}, std: {mean_vector.std().item():.4f}, min: {mean_vector.min().item():.4f}, max: {mean_vector.max().item():.4f}"
            )
            if torch.isnan(mean_vector).any():
                logger.warning(f"  Class '{label}': Prototype contains NaNs!")
            if torch.isinf(mean_vector).any():
                logger.warning(f"  Class '{label}': Prototype contains Infs!")

        # Calculate covariance and precision matrix (if Mahalanobis)
        if self.lpm_context.distance_metric == "mahalanobis":
            logger.info("Calculating pooled sample covariance and precision matrix...")
            self.lpm_context.precision_matrix = self._calculate_precision_matrix(
                all_aggregated_tensor, all_labels_tensor, list(aggregated_activations_by_class.keys())
            )
            logger.info(f"Precision matrix shape: {self.lpm_context.precision_matrix.shape}")

        # Move to target device
        self.lpm_context.to(self.lpm_context.device)

        logger.info("LPM fit complete.")

    def _calculate_precision_matrix(
        self, activations: torch.Tensor, labels: torch.Tensor, label_names: List[str]
    ) -> torch.Tensor:
        """
        Calculate precision matrix using Bayes ridge estimator.

        Following the paper's approach:
        1. Center activations using class-specific means
        2. Calculate pooled sample covariance: Σ̃ = (1/(N-K)) * Z^T * Z
        3. Apply Bayes ridge estimator: Σ̃^{-1} = d * ((N-1)*Σ̃ + tr(Σ̃)*I_d)^{-1}

        Args:
            activations: [N, d_model] aggregated activations
            labels: [N] class labels (as integers)
            label_names: List of class names

        Returns:
            precision_matrix: [d_model, d_model]
        """
        N, d = activations.shape
        K = len(label_names)  # Number of classes

        # Center activations using class-specific means
        centered_activations = []
        for class_idx, label_name in enumerate(label_names):
            mask = labels == class_idx
            class_activations = activations[mask]  # [N_class, d]
            class_mean = class_activations.mean(dim=0)  # [d]
            centered = class_activations - class_mean  # [N_class, d]
            centered_activations.append(centered)

        # Concatenate all centered activations
        Z = torch.cat(centered_activations, dim=0)  # [N, d]

        # Calculate pooled sample covariance
        # Σ̃ = (1/(N-K)) * Z^T * Z
        cov = (Z.T @ Z) / (N - K)  # [d, d]

        # DEBUG NaN means and cov
        logger.info(f"Covariance matrix stats - mean: {cov.mean().item():.4f}, std: {cov.std().item():.4f}")
        logger.info(f"Covariance matrix stats - min: {cov.min().item():.4f}, max: {cov.max().item():.4f}")
        if torch.isnan(cov).any():
            logger.warning("Covariance matrix contains NaNs!")
        if torch.isinf(cov).any():
            logger.warning("Covariance matrix contains Infs!")

        # Apply Bayes ridge estimator
        # Σ̃^{-1} = d * ((N-1)*Σ̃ + tr(Σ̃)*I_d)^{-1}
        trace = torch.trace(cov)  # scalar
        # DEBUG NaN means and cov
        logger.info(f"Trace of covariance: {trace.item():.4f}")

        regularized = (N - 1) * cov + trace * torch.eye(d, device=cov.device)  # [d, d]

        # DEBUG NaN means and cov
        logger.info(
            f"Regularized matrix stats - mean: {regularized.mean().item():.4f}, std: {regularized.std().item():.4f}"
        )
        if torch.isnan(regularized).any():
            logger.warning("Regularized matrix contains NaNs!")
        if torch.isinf(regularized).any():
            logger.warning("Regularized matrix contains Infs!")

        precision = d * torch.inverse(regularized)  # [d, d]

        # DEBUG NaN means and cov
        logger.info(f"Precision matrix stats - mean: {precision.mean().item():.4f}, std: {precision.std().item():.4f}")
        if torch.isnan(precision).any():
            logger.warning("Precision matrix contains NaNs!")
        if torch.isinf(precision).any():
            logger.warning("Precision matrix contains Infs!")

        return precision

    def process_activations(  # noqa: C901
        self, module: torch.nn.Module, input: HOOK_FUNCTION_INPUT, output: HOOK_FUNCTION_OUTPUT
    ) -> None:
        """
        Inference hook: Calculate distances to prototypes for the current batch.

        At inference time, we need to:
        1. Extract activations from hook output
        2. Apply same sequence aggregation as during training
        3. Calculate distances to prototypes
        4. Store predictions
        """
        # Extract tensor from output
        if isinstance(output, torch.Tensor):
            tensor = output
        elif isinstance(output, (tuple, list)) and len(output) > 0:
            tensor = output[0]
        elif hasattr(output, "last_hidden_state"):
            tensor = output.last_hidden_state
        else:
            logger.warning("Could not extract tensor from hook output")
            return

        if not isinstance(tensor, torch.Tensor):
            logger.warning("Extracted output is not a tensor")
            return

        # Tensor shape: [batch, seq, hidden]
        if tensor.dim() == 3:
            batch_size, seq_len, hidden_dim = tensor.shape

            # Try to get attention mask from loaded inference masks
            attention_mask = None
            if self._inference_attention_masks is not None:
                # Use explicit batch counter (not computed from _seen_samples)
                if self._current_batch_idx in self._inference_attention_masks:
                    attention_mask = self._inference_attention_masks[self._current_batch_idx]
                    # Ensure it's on the correct device
                    if attention_mask.device != tensor.device:
                        attention_mask = attention_mask.to(tensor.device)
                    logger.info(f"Using loaded attention mask for batch {self._current_batch_idx}")

            # Fallback: create dummy attention mask (all ones - treat all tokens as valid)
            if attention_mask is None:
                logger.info("No attention mask available, assuming all tokens are valid")
                attention_mask = torch.ones((batch_size, seq_len), device=tensor.device, dtype=tensor.dtype)

            # Apply sequence aggregation
            vectors = aggregate_activations_batch(
                tensor,
                attention_mask,
                agg=self._get_aggregation_type(),
            )
        elif tensor.dim() == 2:
            # Already aggregated
            vectors = tensor
        else:
            logger.warning(f"Unexpected tensor dimensionality: {tensor.dim()}")
            return

        vectors = vectors.to(self.lpm_context.device)

        # Calculate distances
        results = self._calculate_distances(vectors)

        # Convert to python objects for Predictor accumulation
        batch_results: List[Dict[str, Any]] = []
        for i in range(vectors.shape[0]):
            result_dict = {
                "predicted_class": results["predicted_class"][i],
            }
            # Add distances for each class
            for label in self.lpm_context.prototypes.keys():
                result_dict[f"distance_{label}"] = results[f"distance_{label}"][i].item()

            batch_results.append(result_dict)

        self._seen_samples += len(batch_results)
        self._current_batch_idx += 1  # Increment batch counter

        # Keep detector metadata lightweight
        self.metadata["predictions"] = batch_results

        # Standardized accumulation for final save
        self.add_predictions(batch_results)

    def _calculate_distances(self, vectors: torch.Tensor) -> Dict[str, Any]:
        """
        Calculate distances to all prototypes.

        Args:
            vectors: [batch_size, d_model]

        Returns:
            dict with keys:
                - predicted_class: list of predicted class names
                - distance_{label}: [batch_size] distances to each class
        """
        distances = {}

        if self.lpm_context.distance_metric == "euclidean":
            # Euclidean distance: ||x - μ||
            for label, prototype in self.lpm_context.prototypes.items():
                diff = vectors - prototype.unsqueeze(0)  # [batch, d]
                dist = torch.norm(diff, dim=1)  # [batch]
                distances[label] = dist

        elif self.lpm_context.distance_metric == "mahalanobis":
            # Mahalanobis distance: sqrt((x - μ)^T Σ^{-1} (x - μ))
            if self.lpm_context.precision_matrix is None:
                raise ValueError("Precision matrix not calculated. Run fit() first.")

            for label, prototype in self.lpm_context.prototypes.items():
                diff = vectors - prototype.unsqueeze(0)  # [batch, d]
                # (x - μ)^T Σ^{-1} (x - μ)
                mahal_sq = torch.sum(diff @ self.lpm_context.precision_matrix * diff, dim=1)  # [batch]
                dist = torch.sqrt(mahal_sq)  # [batch]
                distances[label] = dist

        # Determine predicted class (minimum distance)
        labels = list(distances.keys())
        dist_stack = torch.stack([distances[label] for label in labels], dim=1)  # [batch, num_classes]

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

    def save(self, store: Store, relative_path: Union[str, Path]):
        """
        Save LPM model with all context to LocalStore.

        Args:
            store: LocalStore instance
            relative_path: Relative path within store (e.g., "models/lpm_layer27.pt")

        Saves:
        - Prototypes
        - Precision matrix (if Mahalanobis)
        - All configuration and metadata

        Example:
            lpm.save(store, "models/lpm_bielik_layer27_mahalanobis.pt")
        """
        full_path = Path(store.base_path) / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "prototypes": self.lpm_context.prototypes,
            "precision_matrix": self.lpm_context.precision_matrix,
            "config": {
                "model_id": self.lpm_context.model_id,
                "layer_signature": self.lpm_context.layer_signature,
                "layer_number": self.lpm_context.layer_number,
                "distance_metric": self.lpm_context.distance_metric,
                "aggregation_method": self.lpm_context.aggregation_method,
                "dataset_name": self.lpm_context.dataset_name,
                "run_id": self.lpm_context.run_id,
                "hidden_dim": self.lpm_context.hidden_dim,
                "positive_label": self.positive_label,
            },
        }
        torch.save(state, full_path)
        logger.info(f"Saved LPM model to {full_path}")
        logger.info(f"  Relative path in store: {relative_path}")

    @classmethod
    def load(cls, store: Store, relative_path: Union[str, Path], device: str = "cpu") -> "LPM":
        """
        Load LPM model from LocalStore.

        Args:
            store: LocalStore instance
            relative_path: Relative path within store (e.g., "models/lpm_layer27.pt")
            device: Device to load model onto

        Returns:
            LPM instance

        Example:
            lpm = LPM.load(store, "models/lpm_bielik_layer27_mahalanobis.pt", device="cuda")
        """
        full_path = Path(store.base_path) / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"Model file not found: {full_path}")

        state = torch.load(full_path, map_location=device)
        config = state["config"]

        # Create instance
        lpm = cls(
            layer_signature=config["layer_signature"],
            layer_number=config.get("layer_number"),
            distance_metric=config["distance_metric"],
            aggregation_method=config["aggregation_method"],
            device=device,
            positive_label=config.get("positive_label", "harmful"),
        )

        # Restore learned parameters
        lpm.lpm_context.prototypes = state["prototypes"]
        lpm.lpm_context.precision_matrix = state["precision_matrix"]
        lpm.lpm_context.model_id = config.get("model_id")
        lpm.lpm_context.dataset_name = config.get("dataset_name")
        lpm.lpm_context.run_id = config.get("run_id")
        lpm.lpm_context.hidden_dim = config.get("hidden_dim")

        # Move to device
        lpm.lpm_context.to(device)

        logger.info(f"Loaded LPM model from {full_path}")
        logger.info(f"  Relative path in store: {relative_path}")
        logger.info(f"  Model: {lpm.lpm_context.model_id}")
        logger.info(f"  Layer: {lpm.lpm_context.layer_signature}")
        logger.info(f"  Distance metric: {lpm.lpm_context.distance_metric}")
        logger.info(f"  Aggregation: {lpm.lpm_context.aggregation_method}")
        logger.info(f"  Classes: {list(lpm.lpm_context.prototypes.keys())}")

        return lpm
