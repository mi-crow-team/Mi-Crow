# ruff: noqa
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from experiments.predictors.predictor import Predictor
from experiments.utils.activation_aggregation import aggregate_activations_batch
from mi_crow.datasets import ClassificationDataset
from mi_crow.hooks.detector import Detector
from mi_crow.hooks.hook import HOOK_FUNCTION_INPUT, HOOK_FUNCTION_OUTPUT, HookType
from mi_crow.store.store import Store
from mi_crow.utils import get_logger

from .probe_context import ProbeContext

logger = get_logger(__name__)


class LinearProbe(Detector, Predictor):
    """
    Linear Probe Classifier for binary content moderation.

    Learns a linear decision boundary (logistic regression) on LLM activations.
    Uses PyTorch for training with early stopping and validation.

    Similar to LPM but uses supervised learning instead of distance to prototypes.
    Loads activations batch-by-batch during fit() to avoid OOM.
    """

    def __init__(
        self,
        layer_signature: str | None = None,
        layer_number: int | None = None,
        aggregation_method: Literal["mean", "last_token", "last_token_prefix"] = "last_token",
        hook_id: str | None = None,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 50,
        patience: int = 5,
        positive_label: str = "harmful",
    ):
        """
        Initialize Linear Probe.

        Args:
            layer_signature: Full layer signature (e.g., "llamaforcausallm_model_layers_27")
            layer_number: Layer number (e.g., 27) - used if layer_signature is None
            aggregation_method: How to aggregate token-level activations
            hook_id: Optional hook identifier
            device: "cpu" or "cuda"
            learning_rate: AdamW learning rate
            weight_decay: L2 regularization
            batch_size: Training batch size
            max_epochs: Maximum training epochs
            patience: Early stopping patience (epochs)
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
            model_id=f"linear_probe_{aggregation_method}_layer{layer_number}",
            config={
                "layer_signature": layer_signature,
                "layer_number": layer_number,
                "aggregation_method": aggregation_method,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "patience": patience,
                "device": device,
                "positive_label": positive_label,
            },
        )

        self.context = ProbeContext(
            layer_signature=layer_signature,
            layer_number=layer_number,
            aggregation_method=aggregation_method,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
        )

        self.positive_label = positive_label
        self.linear = None  # Will be initialized during fit
        self._seen_samples = 0
        self._current_batch_idx = 0
        self._inference_attention_masks: Optional[Dict[int, torch.Tensor]] = None

    def clear_predictions(self) -> None:
        super().clear_predictions()
        self._seen_samples = 0
        self._current_batch_idx = 0

    def _get_aggregation_type(self) -> Literal["last", "mean"]:
        """Convert aggregation method to format expected by aggregate_activations_batch."""
        if self.context.aggregation_method in ["last_token", "last_token_prefix"]:
            return "last"
        elif self.context.aggregation_method == "mean":
            return "mean"
        else:
            raise ValueError(f"Unknown aggregation method: {self.context.aggregation_method}")

    def load_inference_attention_masks(self, store: Store, run_id: str) -> None:
        """
        Load attention masks from a saved run for use during inference.

        Args:
            store: Store containing the saved attention masks
            run_id: Run ID where attention masks were saved
        """
        logger.info(f"Loading attention masks from run: {run_id}")
        self._inference_attention_masks = {}

        batch_indices = store.list_run_batches(run_id)
        logger.info(f"Found {len(batch_indices)} batches with attention masks")

        for batch_idx in batch_indices:
            try:
                attention_mask = store.get_detector_metadata_by_layer_by_key(
                    run_id, batch_idx, "attention_masks", "attention_mask"
                )
                if attention_mask is not None:
                    self._inference_attention_masks[batch_idx] = attention_mask
                    logger.debug(f"Loaded attention mask for batch {batch_idx}: {attention_mask.shape}")
            except Exception as e:
                logger.warning(f"Could not load attention mask for batch {batch_idx}: {e}")
                continue

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
        Train the linear probe on stored activations.

        Loads activations batch-by-batch to avoid OOM. For each batch:
        1. Load activations and attention masks
        2. Apply sequence aggregation (mean, last_token, etc.)
        3. Accumulate aggregated activations in memory
        4. Split into train/val
        5. Train linear layer with early stopping

        Args:
            store: Store containing activations
            run_id: Run ID of the saved activations
            dataset: ClassificationDataset with labels
            model_id: Model ID
            dataset_name: Name of dataset
            category_field: Field name for labels in dataset
            max_samples: Limit number of samples to use
        """
        logger.info("=" * 80)
        logger.info("FITTING LINEAR PROBE")
        logger.info("=" * 80)
        logger.info(f"Aggregation method: {self.context.aggregation_method}")

        # Store metadata
        self.context.model_id = model_id
        self.context.run_id = run_id
        self.context.dataset_name = dataset_name if dataset_name is not None else getattr(dataset, "name", "unknown")

        # Validate binary labels
        all_categories = [item[category_field] for item in dataset]
        unique_labels = set(all_categories)
        if len(unique_labels) != 2:
            raise ValueError(
                f"LinearProbe only supports binary classification. "
                f"Found labels: {unique_labels}. Expected exactly 2 labels."
            )

        # Map labels to 0/1
        label_to_idx = {}
        for label in sorted(unique_labels):
            if label == self.positive_label:
                label_to_idx[label] = 1
            else:
                label_to_idx[label] = 0
        logger.info(f"Label mapping: {label_to_idx}")

        # Get batch indices
        batch_indices = store.list_run_batches(run_id)
        logger.info(f"Found {len(batch_indices)} batches in run {run_id}")

        # FIRST PASS: Count total available activations
        logger.info("Counting available activations...")
        total_available_activations = 0
        for batch_idx in batch_indices:
            try:
                activations = store.get_detector_metadata_by_layer_by_key(
                    run_id, batch_idx, str(self.context.layer_signature), "activations"
                )
                batch_size = activations.shape[0]
                total_available_activations += batch_size
                del activations
            except Exception as e:
                logger.warning(f"Could not count samples in batch {batch_idx}: {e}")
                continue

        logger.info(f"Total available activations: {total_available_activations}")
        logger.info(f"Dataset size: {len(dataset)}")

        # Determine effective max_samples
        effective_max_samples = min(len(dataset), total_available_activations)
        if max_samples is not None:
            effective_max_samples = min(effective_max_samples, max_samples)
            logger.info(f"User-specified max_samples: {max_samples}")

        logger.info(f"Effective max_samples for training: {effective_max_samples}")

        if effective_max_samples < len(dataset):
            logger.warning(
                f"Using only {effective_max_samples} samples out of {len(dataset)} in dataset "
                f"(limited by available activations: {total_available_activations})"
            )

        # SECOND PASS: Load and aggregate activations
        logger.info("Loading and aggregating activations...")
        all_aggregated_activations: List[torch.Tensor] = []
        all_labels: List[int] = []

        current_sample_idx = 0
        samples_processed = 0

        for batch_idx in tqdm(batch_indices, desc="Loading and aggregating activations"):
            if samples_processed >= effective_max_samples:
                break

            try:
                # Load activations
                activations = store.get_detector_metadata_by_layer_by_key(
                    run_id, batch_idx, str(self.context.layer_signature), "activations"
                )

                # Load attention masks (if available)
                try:
                    attention_mask = store.get_detector_metadata_by_layer_by_key(
                        run_id, batch_idx, "attention_masks", "attention_mask"
                    )
                except Exception:
                    attention_mask = None
                    logger.debug(f"No attention mask for batch {batch_idx}, using all-ones")

                # Create dummy mask if not available
                if attention_mask is None:
                    batch_size, seq_len = activations.shape[:2]
                    attention_mask = torch.ones(
                        (batch_size, seq_len), device=activations.device, dtype=activations.dtype
                    )

                # Aggregate activations
                aggregated = aggregate_activations_batch(
                    activations,
                    attention_mask,
                    agg=self._get_aggregation_type(),
                )

                # Initialize hidden_dim from first batch
                if self.context.hidden_dim is None:
                    self.context.hidden_dim = aggregated.shape[1]
                    logger.info(f"Detected hidden_dim: {self.context.hidden_dim}")

                # Get batch size
                actual_batch_size = aggregated.shape[0]
                samples_to_take = min(actual_batch_size, effective_max_samples - samples_processed)

                # Get labels for this batch
                for i in range(samples_to_take):
                    sample_idx = current_sample_idx + i
                    if sample_idx >= len(dataset):
                        break

                    category = dataset[sample_idx][category_field]
                    label = label_to_idx[category]

                    all_aggregated_activations.append(aggregated[i].cpu())
                    all_labels.append(label)
                    samples_processed += 1

                current_sample_idx += actual_batch_size

                # Cleanup
                del activations, attention_mask, aggregated
                gc.collect()

            except Exception as e:
                logger.error(f"Error loading batch {batch_idx}: {e}")
                break

        logger.info(f"Processed {samples_processed} samples")

        if not all_aggregated_activations:
            raise RuntimeError("No training activations loaded!")

        # Convert to tensors
        X_all = torch.stack(all_aggregated_activations)  # [N, hidden_dim]
        y_all = torch.tensor(all_labels, dtype=torch.float32)  # [N]

        # Split into train/val (80/20)
        dataset_size = len(X_all)
        num_val = int(dataset_size * 0.2)
        num_train = dataset_size - num_val

        # Shuffle indices
        indices = torch.randperm(dataset_size)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        X_train = X_all[train_indices].to(self.context.device)
        y_train = y_all[train_indices].to(self.context.device)
        X_val = X_all[val_indices].to(self.context.device) if num_val > 0 else None
        y_val = y_all[val_indices].to(self.context.device) if num_val > 0 else None

        self.context.num_train_samples = num_train
        self.context.num_val_samples = num_val

        logger.info(f"Train samples: {num_train}, Val samples: {num_val}")
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        # Initialize linear layer
        self.linear = nn.Linear(self.context.hidden_dim, 1).to(self.context.device)
        logger.info(f"Initialized linear layer: {self.context.hidden_dim} -> 1")

        # Train the probe
        self._train_probe(X_train, y_train, X_val, y_val)

        # Store learned parameters in context
        self.context.weight = self.linear.weight.data.squeeze().cpu()
        self.context.bias = self.linear.bias.data.cpu()

        logger.info("=" * 80)
        logger.info("LINEAR PROBE TRAINING COMPLETE")
        logger.info("=" * 80)

    def _train_probe(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
    ):
        """Train the linear probe with early stopping."""
        # Create DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.context.batch_size,
            shuffle=True,
        )

        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            self.linear.parameters(),
            lr=self.context.learning_rate,
            weight_decay=self.context.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Early stopping
        best_val_auc = 0.0
        patience_counter = 0
        best_state = None

        logger.info(f"Starting training: max_epochs={self.context.max_epochs}, patience={self.context.patience}")

        for epoch in range(self.context.max_epochs):
            # Training
            self.linear.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = self.linear(X_batch).squeeze()
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(X_batch)

            train_loss /= len(X_train)
            self.context.train_losses.append(train_loss)

            # Validation
            if X_val is not None and y_val is not None:
                self.linear.eval()
                with torch.no_grad():
                    val_logits = self.linear(X_val).squeeze()
                    val_loss = criterion(val_logits, y_val).item()
                    val_probs = torch.sigmoid(val_logits).cpu().numpy()
                    val_preds = (val_probs >= 0.5).astype(int)
                    y_val_np = y_val.cpu().numpy()

                    # Metrics
                    val_acc = accuracy_score(y_val_np, val_preds)
                    val_auc = roc_auc_score(y_val_np, val_probs)

                    self.context.val_losses.append(val_loss)
                    self.context.val_accuracies.append(val_acc)
                    self.context.val_aucs.append(val_auc)

                    logger.info(
                        f"Epoch {epoch + 1}/{self.context.max_epochs} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_acc:.4f} | "
                        f"Val AUC: {val_auc:.4f}"
                    )

                    # Early stopping check
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        self.context.best_epoch = epoch + 1
                        patience_counter = 0
                        # Save best model state
                        best_state = {
                            "weight": self.linear.weight.data.clone(),
                            "bias": self.linear.bias.data.clone(),
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= self.context.patience:
                            logger.info(f"Early stopping at epoch {epoch + 1} (patience={self.context.patience})")
                            # Restore best model
                            if best_state is not None:
                                self.linear.weight.data = best_state["weight"]
                                self.linear.bias.data = best_state["bias"]
                            break
            else:
                logger.info(f"Epoch {epoch + 1}/{self.context.max_epochs} | Train Loss: {train_loss:.4f}")

        logger.info(f"Training complete. Best epoch: {self.context.best_epoch}, Best Val AUC: {best_val_auc:.4f}")

    def process_activations(  # noqa: C901
        self, module: torch.nn.Module, input: HOOK_FUNCTION_INPUT, output: HOOK_FUNCTION_OUTPUT
    ) -> None:
        """
        Inference hook: Apply learned linear layer to current batch.

        At inference time:
        1. Extract activations from hook output
        2. Apply same sequence aggregation as during training
        3. Apply linear layer and sigmoid to get probabilities
        4. Store predictions
        """
        if self.linear is None:
            logger.warning("Probe not trained yet. Skipping inference.")
            return

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
                if self._current_batch_idx in self._inference_attention_masks:
                    attention_mask = self._inference_attention_masks[self._current_batch_idx]
                    # Ensure it's on the correct device
                    if attention_mask.device != tensor.device:
                        attention_mask = attention_mask.to(tensor.device)

                    # Validate dimension compatibility
                    if attention_mask.shape[0] != batch_size or attention_mask.shape[1] != seq_len:
                        logger.warning(
                            f"Attention mask shape mismatch for batch {self._current_batch_idx}: "
                            f"activations=[{batch_size}, {seq_len}, {hidden_dim}], "
                            f"attention_mask={list(attention_mask.shape)}. "
                            f"Falling back to all-ones mask."
                        )
                        attention_mask = None
                    else:
                        logger.info(f"Using loaded attention mask for batch {self._current_batch_idx}")

            # Fallback: create dummy attention mask
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

        vectors = vectors.to(self.context.device)

        # Apply linear layer
        self.linear.eval()
        with torch.no_grad():
            logits = self.linear(vectors).squeeze()
            probs = torch.sigmoid(logits)

        # Convert to python objects for Predictor accumulation
        batch_results: List[Dict[str, Any]] = []
        for i in range(vectors.shape[0]):
            prob = probs[i].item() if probs.dim() > 0 else probs.item()
            result_dict = {
                "probability_harmful": prob,
                "predicted_class": self.positive_label if prob >= 0.5 else "safe",
            }
            batch_results.append(result_dict)

        self._seen_samples += len(batch_results)
        self._current_batch_idx += 1

        # Keep detector metadata lightweight
        self.metadata["predictions"] = batch_results

        # Standardized accumulation for final save
        self.add_predictions(batch_results)

    def get_config(self) -> Dict[str, Any]:
        return self.config

    def save(self, store: Store, relative_path: Union[str, Path]):
        """Save probe to store."""
        from datetime import datetime

        path = Path(relative_path)

        # Save context
        context_data = {
            "model_id": self.context.model_id,
            "layer_signature": self.context.layer_signature,
            "layer_number": self.context.layer_number,
            "dataset_name": self.context.dataset_name,
            "run_id": self.context.run_id,
            "aggregation_method": self.context.aggregation_method,
            "weight": self.context.weight,
            "bias": self.context.bias,
            "learning_rate": self.context.learning_rate,
            "weight_decay": self.context.weight_decay,
            "batch_size": self.context.batch_size,
            "max_epochs": self.context.max_epochs,
            "patience": self.context.patience,
            "train_losses": self.context.train_losses,
            "val_losses": self.context.val_losses,
            "val_accuracies": self.context.val_accuracies,
            "val_aucs": self.context.val_aucs,
            "best_epoch": self.context.best_epoch,
            "num_train_samples": self.context.num_train_samples,
            "num_val_samples": self.context.num_val_samples,
            "hidden_dim": self.context.hidden_dim,
            "device": self.context.device,
            "saved_at": datetime.now().isoformat(),
        }

        # Save to store
        context_path = path / "probe_context.pt"
        store.save_file(str(context_path), context_data)
        logger.info(f"Saved probe context to {context_path}")

    @classmethod
    def load(cls, store: Store, relative_path: Union[str, Path], device: str = "cpu") -> "LinearProbe":
        """Load probe from store."""
        path = Path(relative_path)
        context_path = path / "probe_context.pt"

        # Load context
        context_data = store.load_file(str(context_path))

        # Create probe instance
        probe = cls(
            layer_signature=context_data["layer_signature"],
            layer_number=context_data["layer_number"],
            aggregation_method=context_data["aggregation_method"],
            learning_rate=context_data["learning_rate"],
            weight_decay=context_data["weight_decay"],
            batch_size=context_data["batch_size"],
            max_epochs=context_data["max_epochs"],
            patience=context_data["patience"],
            device=device,
        )

        # Restore context
        probe.context.model_id = context_data.get("model_id")
        probe.context.dataset_name = context_data.get("dataset_name")
        probe.context.run_id = context_data.get("run_id")
        probe.context.weight = context_data["weight"]
        probe.context.bias = context_data["bias"]
        probe.context.train_losses = context_data.get("train_losses", [])
        probe.context.val_losses = context_data.get("val_losses", [])
        probe.context.val_accuracies = context_data.get("val_accuracies", [])
        probe.context.val_aucs = context_data.get("val_aucs", [])
        probe.context.best_epoch = context_data.get("best_epoch")
        probe.context.num_train_samples = context_data.get("num_train_samples")
        probe.context.num_val_samples = context_data.get("num_val_samples")
        probe.context.hidden_dim = context_data["hidden_dim"]

        # Recreate linear layer
        if probe.context.hidden_dim is not None:
            probe.linear = nn.Linear(probe.context.hidden_dim, 1).to(device)
            probe.linear.weight.data = context_data["weight"].unsqueeze(0).to(device)
            probe.linear.bias.data = context_data["bias"].to(device)

        logger.info(f"Loaded probe from {context_path}")
        return probe
