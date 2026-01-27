# LPM and Linear Probe Results Analysis

This directory contains code for analyzing and visualizing results from LPM and Linear Probe experiments.

## Directory Structure

```
experiments/results/
├── analysis_code/          # Python modules for analysis
│   ├── result_loader.py    # Load results from disk
│   ├── visualizations.py   # Create thesis-ready plots
│   └── tables.py           # Generate result tables
├── analysis_scripts/       # Executable scripts
│   └── analyze_lpm_probe_results.py
├── visualizations/         # Generated plots (PNG, high-res)
├── tables/                 # Generated tables (CSV, LaTeX)
└── misc/                   # Other analysis artifacts
```

## Usage

### Run Complete Analysis

From the repository root (`Mi-Crow/`):

```bash
python experiments/results/analysis_scripts/analyze_lpm_probe_results.py
```

Or with custom store path:

```bash
python experiments/results/analysis_scripts/analyze_lpm_probe_results.py --store-path /path/to/store
```

This will:
1. Load all LPM and Linear Probe results
2. Create 3 main figures for the thesis
3. Generate comprehensive result tables (CSV + LaTeX)
4. Print summary statistics

### Generated Visualizations

#### Figure 1: LPM Metric Comparison
- **File**: `visualizations/fig1_lpm_metric_comparison_all.png`
- **Description**: Compares Euclidean vs. Mahalanobis distance metrics for LPM
- **Format**: Grouped bar chart, faceted by dataset (PLMix, WGMix)
- **Aggregation**: Shows mean across all 3 aggregation methods with error bars (whiskers) indicating min/max performance
- **Enhancement**: Whiskers provide insight into aggregation sensitivity

#### Figure 2a: LPM Aggregation Impact
- **Files**: 
  - `visualizations/aggregation_impact_lpm_plmix.png`
  - `visualizations/aggregation_impact_lpm_wgmix.png`
- **Description**: Shows impact of aggregation methods (mean, last_token, last_token_prefix)
- **Metric**: Uses Mahalanobis distance only

#### Figure 2b: Linear Probe Aggregation Impact
- **Files**:
  - `visualizations/aggregation_impact_linear_probe_plmix.png`
  - `visualizations/aggregation_impact_linear_probe_wgmix.png`
- **Description**: Shows impact of aggregation methods for Linear Probes

#### Figure 3: Method Comparison (with Stability Indicators)
- **File**: `visualizations/fig3_method_comparison.png`
- **Description**: Compares best LPM vs. best Linear Probe for each dataset
- **Enhancement**: Includes horizontal dashed lines showing mean performance across ALL configurations
- **Purpose**: Gap between best (bar) and mean (line) indicates method stability
  - Large gap = unstable (performance varies a lot)
  - Small gap = stable (consistently good)

#### Figure 4: Consolidated Aggregation Impact
- **File**: `visualizations/fig4_aggregation_consolidated.png`
- **Description**: Two-panel visualization showing aggregation impact across all experiments
- **Left Panel**: Aggregation sensitivity (F1 range) for both methods on both datasets
- **Right Panel**: Detailed breakdown of Probe/WGMix (where differences are largest)
- **Purpose**: Shows which method×dataset combinations are most/least sensitive to aggregation choice

#### Figure 5: Detailed Method Comparison (All Configurations)
- **File**: `visualizations/fig5_method_comparison_detailed.png`
- **Description**: Comprehensive horizontal bar chart showing LPM vs. Probe for all 18 configurations
- **Format**: Shows all model×dataset×aggregation combinations side-by-side
- **Purpose**: Complete performance overview for deep analysis and pattern identification

### Generated Tables

#### LPM Results Table
- **Files**:
  - `tables/lpm_results.csv` (for spreadsheets)
  - `tables/lpm_results.tex` (for LaTeX thesis)
- **Columns**: Model, Aggregation, Metric, PLMix (F1, Precision, Recall, Accuracy), WGMix (F1, Precision, Recall, Accuracy)
- **Rows**: All 36 LPM experiment configurations

#### Linear Probe Results Table
- **Files**:
  - `tables/probe_results.csv`
  - `tables/probe_results.tex`
- **Columns**: Model, Aggregation, PLMix (F1, Precision, Recall, Accuracy), WGMix (F1, Precision, Recall, Accuracy)
- **Rows**: All 18 probe experiment configurations

#### Best Results Summary
- **Files**:
  - `tables/best_results_summary.csv`
  - `tables/best_results_summary.tex`
- **Description**: Shows top-performing configuration for each method and dataset
- **Includes**: Full metrics and configuration details

## Thesis Integration

### Plotting Style

All visualizations are configured for thesis requirements:

- **Font**: LaTeX Computer Modern (serif) for consistency
- **Context**: Seaborn "paper" context
- **Palette**: Colorblind-safe colors (distinguishable in grayscale)
- **Resolution**: 300 DPI (print quality)
- **Format**: PNG with tight bounding boxes

### LaTeX Integration

To include figures in your thesis:

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{experiments/results/visualizations/fig1_lpm_metric_comparison_mean.png}
    \caption{LPM: Euclidean vs. Mahalanobis Distance Comparison}
    \label{fig:lpm_metric_comparison}
\end{figure}
```

To include tables:

```latex
\input{experiments/results/tables/lpm_results.tex}
```

## Implementation Details

### Result Loading

The analysis automatically:
- Parses run_id strings to extract experiment parameters
- Finds the most recent `inference_[timestamp]` directory for each run
- Loads `analysis/metrics.json` files
- **Recalculates accuracy** for Linear Probes (original values in metrics.json are incorrect)

### Data Processing

- Results are loaded into pandas DataFrames
- Missing experiments are logged but don't stop analysis
- All metrics are validated before plotting

### Error Handling

The script is robust to:
- Missing run directories
- Missing metrics files
- Parsing errors
- LaTeX unavailability (falls back to standard fonts)

## Dependencies

Required Python packages:
- pandas
- matplotlib
- seaborn
- numpy

Optional:
- LaTeX installation (for LaTeX fonts in plots)

## Notes

### Accuracy Recalculation

**Important**: The `accuracy` field in `metrics.json` for Linear Probe runs is incorrect. The analysis code automatically recalculates it from the confusion matrix values (tp, tn, fp, fn).

### Run IDs

The script uses hardcoded run_id lists from `where_results_are_saved.md`. If new experiments are added, update the lists in `analyze_lpm_probe_results.py`.

### Store Structure

Expected structure:
```
store/
├── lpm_[model]_[dataset]_[agg]_layer[N]_[metric]/
│   └── runs/
│       └── inference_[timestamp]/
│           └── analysis/
│               └── metrics.json
└── probe_[model]_[dataset]_[agg]_layer[N]/
    └── runs/
        └── inference_[timestamp]/
            └── analysis/
                └── metrics.json
```

## Contact

For questions or issues with the analysis code, refer to the project documentation or contact the maintainers.
