# Linear Probe Classifiers

This module implements linear probe classifiers for content moderation on LLM activations.

## Overview

Linear probes learn a linear decision boundary (logistic regression) to classify content as safe or harmful based on hidden layer activations. Unlike the LPM (Latent Prototype Moderator) which uses distance to pre-computed prototypes, probes learn optimal weights through supervised training with backpropagation.

## Architecture

### ProbeContext
Data class holding configuration and learned parameters:
- **Model info**: model_id, layer_signature, layer_number
- **Training config**: learning_rate, weight_decay, batch_size, max_epochs, patience
- **Learned parameters**: weight, bias (from linear layer)
- **Training history**: losses, accuracies, AUCs across epochs

### LinearProbe
Main classifier class inheriting from `Detector` and `Predictor`:

**Training:**
- 80/20 train/validation split
- PyTorch nn.Linear layer (hidden_dim → 1)
- BCEWithLogitsLoss (binary cross-entropy)
- AdamW optimizer with weight decay
- Early stopping on validation AUC (patience=5 epochs)
- Batch-by-batch activation loading (memory efficient)

**Inference:**
- Loads activations from store
- Aggregates token-level activations (mean/last_token/last_token_prefix)
- Applies learned linear layer
- Returns probability scores via sigmoid

## Usage

### Training a Probe

```python
from mi_crow.store import LocalStore
from experiments.models.probes import LinearProbe

# Initialize store
store = LocalStore(base_path="store/runs/activation_run_id")

# Create probe
probe = LinearProbe(
    store=store,
    run_id="activation_run_id",
    aggregation_method="last_token",
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=32,
    max_epochs=50,
    patience=5,
    device="cpu",
    model_id="speakleash/Bielik-1.5B-v3.0-Instruct",
    layer_signature="llamaforcausallm_model_layers_31",
    layer_number=31,
)

# Train
probe.fit(dataset=train_items, max_samples=1000)

# Evaluate
metrics = probe.evaluate(dataset=test_items)
print(metrics)  # accuracy, precision, recall, F1, ROC-AUC, PR-AUC

# Save
probe.save("experiments/results/probes/my_probe")
```

### Running Experiments

#### Single Experiment
```bash
uv run python -m experiments.scripts.run_probe_experiment_oom \
    --model speakleash/Bielik-1.5B-v3.0-Instruct \
    --train-dataset wgmix_train \
    --test-dataset wgmix_test \
    --aggregation last_token \
    --layer 31 \
    --learning-rate 1e-3 \
    --weight-decay 1e-4 \
    --batch-size 32 \
    --max-epochs 50 \
    --patience 5
```

#### Full Experiment Matrix (SLURM)
```bash
sbatch slurm/run_probe_experiments.sh
```

Runs 36 experiments:
- 3 models × 2 datasets × 3 aggregations × 2 test sets
- Models: Bielik-1.5B, Bielik-4.5B, Llama-3.2-3B
- Datasets: wgmix (English), plmix (Polish)
- Aggregations: mean, last_token, last_token_prefix
- Tests: same-language + cross-lingual

## Hyperparameters

**Defaults** (validated by user requirements):
- `learning_rate`: 1e-3
- `weight_decay`: 1e-4 (L2 regularization)
- `batch_size`: 32
- `max_epochs`: 50
- `patience`: 5 (early stopping)
- `train_split`: 0.8 (80/20 train/val)

**Rationale:**
- Learning rate: Standard for Adam/AdamW
- Weight decay: Light regularization prevents overfitting
- Batch size: Balances memory and gradient stability
- Max epochs: Upper bound, early stopping typically terminates earlier
- Patience: Allows 5 epochs without improvement before stopping

## Metrics

Comprehensive binary classification metrics:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **F1**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve

## Memory Optimization

Same memory management as LPM:
- **Batch-by-batch loading**: Activations loaded incrementally
- **Two-pass approach**: Count available activations first, then load
- **Aggressive cleanup**: Double GC passes, circular reference breaking
- **Partial activation handling**: Gracefully handles incomplete SLURM runs

## File Structure

```
experiments/models/probes/
├── __init__.py            # Module exports
├── probe_context.py       # Context dataclass
├── linear_probe.py        # Main classifier
└── README.md              # This file

experiments/scripts/
├── run_probe_experiment_oom.py   # Experiment runner
└── test_probe.sh                  # Quick test script

slurm/
└── run_probe_experiments.sh      # Array job for full matrix
```

## Comparison: Probe vs LPM

| Feature | Linear Probe | LPM |
|---------|-------------|-----|
| **Training** | Supervised learning | Training-free |
| **Method** | Learned linear boundary | Distance to prototypes |
| **Parameters** | Weights + bias | Class means (+ covariance for Mahalanobis) |
| **Optimization** | Gradient descent | Analytical computation |
| **Complexity** | O(epochs × samples) | O(samples) |
| **Flexibility** | Learns optimal boundary | Fixed distance metric |
| **Interpretability** | Weight magnitudes | Prototype locations |

## Results Storage

Each experiment saves:
- `{run_name}_results.json`: Metrics, hyperparameters, training history
- `{run_name}_predictions.csv`: Per-sample predictions and labels
- `{run_name}_confusion_matrix.png`: Confusion matrix plot
- `{run_name}_classification_report.txt`: Detailed metrics
- `{run_name}_probe_model/`: Saved probe (weights, bias, context)

## Loading Saved Probes

```python
from mi_crow.store import LocalStore
from experiments.models.probes import LinearProbe

store = LocalStore(base_path="store/runs/test_run_id")
probe = LinearProbe.load(
    path="experiments/results/probes/my_probe_model",
    store=store,
    device="cpu"
)

# Use for inference
predictions = probe.predict(dataset=new_data)
```

## Testing

Quick validation test:
```bash
bash experiments/scripts/test_probe.sh
```

Runs on 200 train / 100 test samples with max 20 epochs for fast validation.

## Implementation Notes

1. **Binary classification only**: Raises ValueError if non-binary labels detected
2. **PyTorch-based**: Uses nn.Linear for flexibility and GPU support
3. **Early stopping**: Monitors validation AUC, restores best weights
4. **Dimension validation**: Checks attention mask shapes before aggregation
5. **Dtype safety**: Same as LPM (handles bfloat16 gracefully)

## Dependencies

- PyTorch: Neural network layers and optimization
- scikit-learn: Metrics (ROC-AUC, precision, recall, etc.)
- mi_crow: Store, hooks, detector/predictor interfaces
- numpy, pandas: Data manipulation

## Future Enhancements

Potential improvements:
- [ ] Multi-class support (softmax + cross-entropy)
- [ ] L1 regularization option (feature selection)
- [ ] Learning rate scheduling (reduce on plateau)
- [ ] Class balancing (weighted loss for imbalanced data)
- [ ] Non-linear probes (MLP with hidden layers)
- [ ] Visualization: weight distributions, decision boundaries
