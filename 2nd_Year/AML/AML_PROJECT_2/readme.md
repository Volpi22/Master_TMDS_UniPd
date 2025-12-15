# AML Project 2: Sigmoid Kernel Analysis for SVM

This repository contains code for analyzing sigmoid kernel behavior in Support Vector Machines (SVMs), with a focus on handling non-positive semidefinite (non-PSD) kernels through various correction methods.

## Project Structure

```
AML_PROJECT_2/
│
├── Project_2.ipynb          # Main notebook with experiments
├── readme.md                # This file
├── img/                     # Generated plots and visualizations
├── src/
│   ├── plots.py            # Plotting functions
│   └── utils.py            # Kernel functions and utilities
└── .gitignore
```

## Requirements

- Python 3.8+
- NumPy
- scikit-learn
- matplotlib
- joblib
- scipy

### Installation

Install required packages:

```bash
pip install numpy scikit-learn matplotlib joblib scipy
```

## Usage

### Running the Notebook

1. Open `Project_2.ipynb` in Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook Project_2.ipynb
   ```

2. Run cells sequentially from top to bottom.

### Notebook Structure

The notebook contains two main experiments:

#### **Experiment 1: Standard Sigmoid Kernel** (Cells 1-9)
- **Cell 1**: Import libraries and create directories
- **Cell 2**: Load and preprocess breast cancer dataset
- **Cell 3**: Scan parameter grid for kernel properties
- **Cell 4**: Plot kernel property heatmaps
- **Cell 5**: Run parallel parameter sweep with kernel corrections
- **Cell 6**: Analyze parameter combinations with significant changes
- **Cell 7**: Analyze kernel eigenvalue properties
- **Cell 8**: Create scatter plots of improvements vs kernel properties
- **Cell 9**: Plot parameter space analysis

#### **Experiment 2: Normalized Sigmoid Kernel** (Cells 10-17)
- **Cells 11-16**: Same analysis as Experiment 1, but with normalized kernel
- **Cell 17**: Compare results between standard and normalized kernels

#### **Experiment 3: Dataset Size Impact Analysis** (Cells 18-24)
- **Cell 18**: Define dataset size fractions to test (20%, 40%, 60%, 80%)
- **Cell 19**: Run experiments for each dataset size with parallel processing
- **Cell 20**: Create heatmaps for each dataset size
- **Cell 21**: Generate difference heatmaps comparing 80% vs 20% dataset
- **Cell 22**: Comparative analysis across all dataset sizes
- **Cell 23**: Visualize CPD kernel percentage vs dataset size
- **Cell 24**: Baseline F1 vs ΔF1 plot for 20% dataset

### Execution Time

- Full parameter grid scan: ~5-30 minutes (depending on CPU cores)
- Parallel processing utilizes all available CPU cores by default

### Output

Generated files in `img/` directory:
- `heatmap_*_properties.png`: Kernel property heatmaps
- `delta_f1_histogram*.png`: Distribution of F1 score improvements
- `correction_vs_*_eigenvalues*.png`: Improvement vs eigenvalue analysis
- `parameter_space_*.png`: Parameter space visualizations
- `baseline_f1_vs_delta*.png`: Baseline performance analysis
- `heatmaps_exp3_*.png`: Dataset size-specific heatmaps (Experiment 3)
- `heatmaps_diff_80pct_vs_20pct.png`: Difference heatmaps comparing dataset sizes
- `cpd_percentage_vs_dataset_size.png`: CPD kernel percentage across dataset sizes
- `bExperiment 1 & 2: Kernel Variants

1. **Standard Sigmoid Kernel**: `K(x,y) = tanh(γ·<x,y> + c)`
2. **Normalized Sigmoid Kernel**: `K(x,y) = tanh(γ·cos_sim(x,y) + c)`

## Key Functions

### `src/utils.py`
- `sigmoid_kernel()`: Standard sigmoid kernel computation
- `sigmoid_kernel_normalized()`: Normalized sigmoid kernel
- `clip_spectrum()`: Eigenvalue clipping correction
- `shift_spectrum()`: Diagonal shift correction
- `clip_norm_spectrum()`: Clip and normalize correction
- `scan_param_grid()`: Analyze kernel properties across parameters
- `process_parameter_combination()`: Cross-validate single parameter set

### `src/plots.py`
- `plot_heatmaps()`: Visualize kernel properties
- `delta_f1_hist()`: Plot improvement distributions
- `correction_plots_eig()`: Eigenvalue analysis plots
- `correction_plots_imp()`: Improvement scatter plots
- `correction_plots_coefs()`: Parameter space visualization
- `correction_plots_baseline()`: Baseline comparison plots

## Dataset

The experiments use the **Breast Cancer Wisconsin Dataset** from scikit-learn:
- 569 samples
- 30 features
- Binary classification (malignant/benign)
- Train/test split: 80/20 with stratification

## Notes

- The notebook uses parallel processing (`n_jobs=-1`) to speed up computations
- All random operations use `random_state=42` for reproducibility
- Images are saved automatically to the `img/` directory
- Progress is printed to console during execution