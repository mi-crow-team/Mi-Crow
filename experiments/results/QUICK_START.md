# Quick Start Guide: Enhanced Visualizations

## Generate All Figures

```bash
cd /Volumes/SanDiskData/Inzynierka
python experiments/results/analysis_scripts/analyze_lpm_probe_results.py
```

**Output:**
- `experiments/results/visualizations/fig1_lpm_metric_comparison_all.png`
- `experiments/results/visualizations/fig3_method_comparison.png`
- `experiments/results/visualizations/fig4_aggregation_consolidated.png`
- `experiments/results/visualizations/fig5_method_comparison_detailed.png`
- `experiments/results/visualizations/aggregation_impact_*.png` (4 files)
- `experiments/results/tables/*.csv` (3 files)
- `experiments/results/tables/*.tex` (3 files)

## Test New Visualizations

```bash
python experiments/results/analysis_scripts/test_new_visualizations.py
```

Creates test plots in `experiments/results/visualizations/test/` with mock data.

## Individual Plot Generation

If you want to generate just one figure:

```python
from pathlib import Path
import sys
sys.path.insert(0, "experiments/results/analysis_code")

from result_loader import load_lpm_results, load_probe_results
from new_visualizations import plot_aggregation_impact_consolidated

# Load data
store_path = Path("store")
lpm_run_ids = [...]  # Your run IDs
probe_run_ids = [...]

lpm_df = load_lpm_results(lpm_run_ids, store_path)
probe_df = load_probe_results(probe_run_ids, store_path)

# Generate plot
output_path = Path("my_figure.png")
plot_aggregation_impact_consolidated(lpm_df, probe_df, output_path)
```

## LaTeX Integration

### Figure 1 (Metric Comparison with Whiskers)

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{experiments/results/visualizations/fig1_lpm_metric_comparison_all.png}
    \caption{LPM Metric Comparison: Euclidean vs. Mahalanobis Distance. 
             Bars show mean F1 across all aggregation methods (mean, last token, 
             last token with prefix). Error bars indicate minimum and maximum 
             performance across aggregations, revealing sensitivity to aggregation 
             method choice.}
    \label{fig:lpm_metrics}
\end{figure}
```

### Figure 3 (Method Comparison with Stability)

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{experiments/results/visualizations/fig3_method_comparison.png}
    \caption{Method Comparison: LPM vs. Linear Probe. Bars represent best 
             configuration F1 scores for each method on each dataset. Dashed 
             horizontal lines show mean performance across all configurations, 
             indicating method stability. The gap between best and mean reveals 
             sensitivity to hyperparameter choices.}
    \label{fig:method_comparison}
\end{figure}
```

### Figure 4 (Consolidated Aggregation Impact)

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{experiments/results/visualizations/fig4_aggregation_consolidated.png}
    \caption{Aggregation Method Impact Analysis. 
             Left: F1 score range (max - min) across aggregation methods for both 
             techniques on both datasets, showing sensitivity to aggregation choice. 
             Right: Detailed breakdown of Linear Probe on WGMix dataset, the 
             configuration most affected by aggregation method selection.}
    \label{fig:aggregation_impact}
\end{figure}
```

### Figure 5 (Detailed Comparison - Appendix)

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{experiments/results/visualizations/fig5_method_comparison_detailed.png}
    \caption{Comprehensive Method Comparison: LPM vs. Linear Probe across all 
             configurations. Each row represents a unique combination of dataset, 
             model, and aggregation method. LPM uses Mahalanobis distance metric 
             (best performing). This detailed view reveals performance patterns 
             across all experimental conditions.}
    \label{fig:method_detailed}
\end{figure}
```

## Interpreting the Enhancements

### Figure 1 Whiskers

**Example:**
```
Bielik-1.5B, PLMix, Euclidean:
  Bar = 0.73 (mean F1)
  Upper whisker = 0.80 (best aggregation)
  Lower whisker = 0.66 (worst aggregation)
```

**Interpretation:**
- Range of 0.14 suggests aggregation choice significantly impacts performance
- Mean of 0.73 provides expected performance across aggregations
- Whiskers help identify when aggregation tuning is critical

### Figure 3 Stability Lines

**Example:**
```
LPM:
  Best F1 = 0.82 (bar)
  Mean F1 = 0.73 (dashed line)
  Gap = 0.09

Linear Probe:
  Best F1 = 0.78 (bar)
  Mean F1 = 0.75 (dashed line)
  Gap = 0.03
```

**Interpretation:**
- Linear Probe more stable (gap = 0.03)
- LPM more variable but higher peak (gap = 0.09, best = 0.82)
- Trade-off: Stability vs. peak performance

### Figure 4 Analysis

**Left Panel (Aggregation Sensitivity):**
```
PLMix:
  LPM range: 0.05
  Probe range: 0.03

WGMix:
  LPM range: 0.04
  Probe range: 0.12  ← Largest!
```

**Interpretation:**
- Most configs: aggregation doesn't matter much (ranges < 0.05)
- Probe/WGMix: aggregation matters significantly (range = 0.12)
- This justifies showing detailed breakdown in right panel

**Right Panel (Probe/WGMix Detail):**
- Shows which aggregation works best for each model
- Validates that this is the "special case" worth discussing

## Thesis Writing Tips

### Reporting Small Differences

**Good approach:**
```
"We evaluated the impact of aggregation method choice across all 
configurations. As shown in Figure 4 (left), most combinations 
exhibited small sensitivity (F1 range < 0.05), suggesting robust 
performance regardless of aggregation choice. However, Linear Probe 
on the WGMix dataset showed substantial variation (range = 0.12), 
indicating that aggregation method is critical for this specific 
configuration (Figure 4, right)."
```

**Why this works:**
- Shows you did comprehensive analysis
- Honest about mostly null result
- Highlights the interesting exception
- Demonstrates scientific rigor

### Discussing Stability

**Example:**
```
"While LPM achieved higher peak performance (F1 = 0.82), it exhibited 
greater variability across configurations (mean F1 = 0.73, gap = 0.09). 
In contrast, Linear Probe showed more consistent performance (mean F1 = 
0.75, gap = 0.03), suggesting better stability with reduced hyperparameter 
sensitivity. This trade-off between peak performance and stability is an 
important consideration for deployment scenarios."
```

## Checklist for Thesis Integration

Before including figures:

- [ ] Generated with real data (not test data)
- [ ] Reviewed for clarity and readability
- [ ] Captions written (concise but informative)
- [ ] Referenced in main text before figure appears
- [ ] Discussed/interpreted in results section
- [ ] LaTeX compiles without errors
- [ ] Figures visible in grayscale (print preview)
- [ ] File sizes reasonable (< 5MB each)
- [ ] Figure numbers consistent with thesis outline

## Troubleshooting

### Plots not generating

Check that you have actual results in `store/` directory:
```bash
ls -la store/ | grep -E "(lpm_|probe_)"
```

### Import errors

Make sure you're running from repository root:
```bash
cd /Volumes/SanDiskData/Inzynierka
python experiments/results/analysis_scripts/analyze_lpm_probe_results.py
```

### Font issues

If serif fonts not working, check matplotlib backend:
```python
import matplotlib
print(matplotlib.get_backend())  # Should be 'agg' for script generation
```

### Missing data

Verify run_ids match actual directory names in store:
```python
from pathlib import Path
store = Path("store")
lpm_dirs = [d.name for d in store.glob("lpm_*")]
print(f"Found {len(lpm_dirs)} LPM directories")
```

## Support

For issues or questions:
1. Check `experiments/results/README.md` for detailed documentation
2. Review `experiments/.llm_context/experiments/visualization_improvements.md`
3. Run test script to verify setup: `test_new_visualizations.py`
4. Check that data loading works: `test_analysis_setup.py`
