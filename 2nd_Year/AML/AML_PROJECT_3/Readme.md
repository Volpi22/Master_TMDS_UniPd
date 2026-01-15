# RFF Component Analysis

This repository contains the code used to study Random Fourier Feature (RFF)–based Kernel Density Estimation (KDE), with experiments on both synthetic datasets of varying difficulty and the real MAGIC Gamma Telescope dataset.

The project is implemented in Python 3 using Jupyter notebooks and a small `src/` package.

---

## Contents

- `Project_3.ipynb` – main notebook orchestrating all experiments and plots  
- `src/`  
	- `Data.py` – synthetic data generators and MAGIC Gamma loading/preprocessing  
	- `Classifiers.py` – models / KDE and RFF-KDE components  
	- `Executions.py` – experiment pipelines (single run, scaling runs, etc.)  
	- `Plots.py` – plotting utilities (pair plots, correlation plots, RFF comparisons, scaling curves)  
	- `Stat_tests.py` – statistical equivalence tests between RFF-KDE and exact KDE  
	- `Utils.py` – general helper utilities  
- `requirements.txt` – Python dependencies (to be added)

---

## Installation

1. **Clone the repository (you can ignore this part if you already have this folder)**

		```bash
		git clone <github_link>
		cd <your-repo-folder>
		```

2. **Create and activate a virtual environment (recommended)**

		```bash
		python -m venv .venv
		.venv\Scripts\activate
		```

3. **Install dependencies**

		```bash
		pip install -r requirements.txt
		```

---

## Running the Experiments

### 1. Open the notebook

You can use either VS Code or Jupyter:

- **VS Code**: open the folder, then open `Project_3.ipynb`.  
- **Classic Jupyter**:

		```bash
		jupyter notebook Project_3.ipynb
		```

Make sure the kernel uses the virtual environment where you installed the dependencies.

### 2. Notebook structure

The notebook is organized into the following main sections:

1. **Synthetic datasets**
	 - Generates synthetic data at three difficulty levels: `easy`, `medium`, `hard`.  
	 - Visualizes the distributions using pair plots.  
	 - Runs RFF-KDE vs exact KDE for a grid of RFF dimensions:
		 - Synthetic: `D_VALUES = logspace(1, 1500, 10)`  
	 - Aggregates results into performance tables and comparison plots.

2. **Real data: MAGIC Gamma**
	 - Loads and preprocesses the MAGIC Gamma dataset.  
	 - Visualizes feature distributions and feature–target correlations.  
	 - Runs RFF-KDE with:
		 - `D_VALUES_REAL = logspace(1, N/2, 10)` (where `N` is the number of samples)  
	 - Compares RFF-KDE with exact KDE baselines.  
	 - Saves performance comparison tables (e.g., to `results/magic_performance_comparison.csv`).

3. **Complexity and scaling analysis**
	 - Evaluates runtime and performance for increasing dataset sizes:
		 - `sizes = [1000, 2000, 3000, 4000, 5000]`  
	 - Compares scaling of exact KDE vs RFF-KDE.  
	 - Produces summary tables and plots.

4. **Statistical tests**
	 - Performs statistical equivalence tests between RFF-KDE and exact KDE.  
	 - Uses a stratified evaluation over:
		 - Difficulty levels: `["easy", "medium", "hard"]`  
		 - RFF dimensions in `D_VALUES`  
		 - A fixed set of random seeds.

---

## Reproducibility and Parallelization

- **Random seeds**

	A fixed set of seeds is used to ensure reproducibility and to estimate variability across runs, e.g.:

	```text
	[42, 123, 456, 789, 1024, 2048, 4096, 8192,
	 271828, 314159, 161803, 999, 5555, 7777,
	 12345, 54321, 13579, 24680, 87654, 11111]
	```

- **Parallel execution**

	Independent configurations are executed in parallel using `joblib.Parallel` and Python’s `multiprocessing`:

	- Number of workers: `N_JOBS = max(1, cpu_count() - 1)`  
	- Tasks include different combinations of:
		- difficulty level  
		- seed  
		- RFF dimension `D`  
		- dataset size (for scaling)

This keeps experiment runtime manageable while fully utilizing available CPU cores.

---

## Outputs

The notebook and utilities generate:

- **Plots**
	- Pair plots for synthetic and real data  
	- RFF-KDE vs exact KDE performance curves over `D`  
	- Scaling plots over dataset size  

- **Tables / CSVs**
	- Aggregated performance comparison tables for synthetic and real data  
	- Statistical test summaries  
	- Scaling summary tables  
---

