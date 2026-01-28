# Efficiency Comparison Analysis

A comprehensive tool for comparing mechanistic interpretability methods with baseline approaches in terms of computational efficiency (FLOPs) and performance (F1 score).

## Purpose

This analysis demonstrates that **mechanistic interpretability methods** (Linear Probes and LPMs) can achieve **competitive performance** while being **orders of magnitude faster** than traditional LLM-based safety approaches. This is critical for production deployments where computational efficiency matters.

## Quick Start

```bash
# Run from the Mi-Crow directory
cd /Volumes/SanDiskData/Inzynierka

# Generate the full efficiency comparison table
python experiments/results/analysis_scripts/generate_efficiency_table.py

# Or run the demo with sample data
python experiments/.llm_context/demo/efficiency_comparison_demo.py
```

## Output

The script generates two files in `experiments/results/tables/`:

1. **efficiency_comparison.csv** - CSV format table
2. **efficiency_comparison.tex** - LaTeX format for thesis/papers

### Table Structure

| Method | FLOPs | F1 | Category |
|--------|-------|----|----|
| Linear Probe(Bielik-1.5B) | 2.30e+05 | 85.67% | MI Method |
| LPM(Bielik-1.5B) Euclidean | 2.40e+05 | 87.23% | MI Method |
| ... | ... | ... | ... |
| Llama-Guard-3-1B | 3.01e+11 | 72.45% | Baseline Guard |
| Bielik-4.5B-Prompted | 1.39e+12 | 81.34% | Direct Prompting |

## Methods Compared

### Mechanistic Interpretability (MI Methods)
- **Linear Probe(Bielik-1.5B)** - Simple linear classifier on activations
- **Linear Probe(Llama-3B)** - Simple linear classifier on activations
- **LPM(Bielik-1.5B) Euclidean** - Linear Probing with Mean using Euclidean distance
- **LPM(Bielik-1.5B) Mahalanobis** - Linear Probing with Mean using Mahalanobis distance
- **LPM(Llama-3B) Euclidean** - Linear Probing with Mean using Euclidean distance
- **LPM(Llama-3B) Mahalanobis** - Linear Probing with Mean using Mahalanobis distance

### Baseline Guard Models
- **Llama-Guard-3-1B** - Meta's safety classifier (1B parameters)
- **Bielik-Guard-0.1B** - Polish safety classifier (124M parameters)

### Direct Prompting
- **Bielik-4.5B-Prompted** - Bielik-4.5B with safety prompts (averaged across 4 prompts)
- **Llama-3.2-3B-Prompted** - Llama-3.2-3B with safety prompts (averaged across 4 prompts)

## Aggregation Strategy

### For MI Methods
- **Only `mean` aggregation** is used
- FLOPs calculations assume mean aggregation (averaging across sequence)
- F1 scores filtered to match this assumption

### For Prompted Models
- **Averaged across all prompts** (prompts 0-3)
- Each prompt is a different safety instruction formulation
- Shows robustness across prompt variations

### For All Methods
- **Averaged across both test datasets**: `plmix_test` and `wgmix_test`
- Provides dataset-agnostic performance metric

## Key Insights

### Computational Efficiency
- **MI methods: ~10⁵ FLOPs** (hundreds of thousands)
- **Baseline guards: ~10¹¹ FLOPs** (hundreds of billions)
- **Prompted LLMs: ~10¹² FLOPs** (trillions)

**→ MI methods are 6-7 orders of magnitude faster!**

### Performance
- MI methods achieve **competitive F1 scores** (typically 80-90%)
- Often within **5-10% of full LLM approaches**
- Best MI methods can **match or exceed** some baselines

### Trade-off
- **Small performance cost** (5-10% F1 reduction in some cases)
- **Massive efficiency gain** (1,000,000x faster)
- **Viable for production** where latency matters

## Implementation Details

### File Structure

```
experiments/results/
├── analysis_code/
│   ├── baseline_loader.py      # Load baseline & prompting results
│   ├── efficiency_table.py     # Create efficiency comparison table
│   └── result_loader.py        # Load LPM & probe results
├── analysis_scripts/
│   └── generate_efficiency_table.py  # Main execution script
└── tables/
    ├── efficiency_comparison.csv     # Generated CSV
    └── efficiency_comparison.tex     # Generated LaTeX

experiments/.llm_context/
├── demo/
│   └── efficiency_comparison_demo.py  # Demo script
└── docs/
    └── efficiency_comparison_analysis.md  # This documentation
```

### Data Sources

#### MI Method Results
- **Location**: `store/lpm_*/` and `store/probe_*/`
- **Files**: `runs/inference_*/analysis/metrics.json`
- **Loader**: `result_loader.py` (existing utilities)

#### Baseline Guard Results
- **Location**: `store/runs/baseline_*/`
- **Files**: `analysis/metrics.json`
- **Run IDs**: Listed in `where_baseline_results_are_saved.md`

#### Direct Prompting Results
- **Location**: `store/runs/direct_prompting_*/`
- **Files**: `analysis/metrics.json`
- **Multiple runs**: One per prompt variant (0-3)

#### FLOPs Data
- **Location**: `experiments/.llm_context/experiments/flops.json`
- **Contains**: Pre-calculated FLOPs for all methods
- **Format**: Scientific notation (e.g., 2.30e+05)

### FLOPs Calculation Methodology

#### Baseline LLMs
```python
# Full forward pass through the model
FLOPs = 2 * parameters * sequence_length
```

For example, Llama-3.2-3B with sequence length 150:
```
FLOPs ≈ 2 * 3.21e9 * 150 ≈ 9.67e11
```

#### MI Methods
```python
# Only the classifier logic (activations pre-computed)
Linear Probe: D * S  # Matrix-vector multiplication
LPM Euclidean: D * (S + 6)  # Distance calculation
LPM Mahalanobis: D * (S + 4*D + 6)  # With covariance
```

Where:
- `D` = hidden dimension (e.g., 3072 for Llama-3B)
- `S` = sequence length (150)

**Key assumption**: Activations are pre-computed and cached. The MI method only performs the classification logic, not the full forward pass.

## Usage Examples

### Basic Usage

```python
from pathlib import Path
from experiments.results.analysis_code.efficiency_table import (
    create_efficiency_comparison_table,
    load_flops_data,
)
from experiments.results.analysis_code.result_loader import (
    load_lpm_results,
    load_probe_results,
)
from experiments.results.analysis_code.baseline_loader import (
    load_baseline_results,
    load_direct_prompting_results,
)

# Load data
store_path = Path("store")
flops_data = load_flops_data("experiments/.llm_context/experiments/flops.json")

# Load results
lpm_df = load_lpm_results(store_path, LPM_RUN_IDS)
probe_df = load_probe_results(store_path, PROBE_RUN_IDS)
baseline_df = load_baseline_results(store_path, BASELINE_RUN_IDS)
prompting_df = load_direct_prompting_results(store_path, PROMPTING_RUN_IDS)

# Create table
table = create_efficiency_comparison_table(
    lpm_df, probe_df, baseline_df, prompting_df, flops_data,
    output_path="tables/efficiency.csv"
)
```

### Custom Analysis

```python
# Filter for specific methods
mi_only = table[table["Category"] == "MI Method"]

# Calculate speedup
avg_mi_flops = mi_only["FLOPs_numeric"].mean()
avg_baseline_flops = table[table["Category"] != "MI Method"]["FLOPs_numeric"].mean()
speedup = avg_baseline_flops / avg_mi_flops
print(f"MI methods are {speedup:.1e}x faster on average")

# Compare performance
print(f"MI avg F1: {mi_only['F1_numeric'].mean()*100:.1f}%")
print(f"Baseline avg F1: {table[table['Category'] != 'MI Method']['F1_numeric'].mean()*100:.1f}%")
```

## Troubleshooting

### "Run directory not found"
- Check that `store_path` points to the correct directory
- Verify run IDs in `where_baseline_results_are_saved.md` and `where_results_are_saved.md`

### "No inference run found"
- For LPM/Probe: Ensure `runs/inference_*` subdirectory exists
- Script automatically finds the most recent inference run

### "Metrics file not found"
- Check that `analysis/metrics.json` exists in the run directory
- Ensure experiments completed successfully

### Wrong accuracy values
- **Note**: Probe accuracy in `metrics.json` can be incorrect
- Script automatically recalculates from confusion matrix (tp, tn, fp, fn)

## References

- **FLOPs calculations**: `experiments/.llm_context/experiments/flops.json`
- **Result locations**: `experiments/.llm_context/experiments/where_results_are_saved.md`
- **Baseline locations**: `experiments/.llm_context/experiments/where_baseline_results_are_saved.md`
