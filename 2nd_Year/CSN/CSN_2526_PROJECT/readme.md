# CSN 2526 Project — Repository Guide (Structure & Usage)

This README describes **how the project is organized** and **how to run it**.

---

## Repository Layout

Typical structure:

- **`CSN_2526_Project.ipynb`**
  - Main entrypoint used to run experiments, generate plots, and produce the results shown in the report.
  - Imports project code from `src/`.

- **`src/`**
  - Project Python modules used by the notebook.
  - Expected modules (based on notebook imports):
    - `src/Simulations.py` — simulation routines (e.g., SIR + rewiring runs, Monte Carlo drivers).
    - `src/Metrics.py` — metric computation utilities used during analysis.
    - `src/Plots.py` — plotting/visualization helpers.
    - `src/Epidemic.py` —epidemic functions.

---

## Entry Points

### 1) Notebook (primary)
Open and run:

- `CSN_2526_Project.ipynb`

The notebook is organized into sections:
- Imports
- Simulation parameter setup
- Toy simulation (visual check)
- Metric comparisons across window sizes
- Monte Carlo runs (rewiring vs static)
- Additional Monte Carlo run for Rt comparison

---

## Requirements

### Python
Use Python **3.10+** (3.11 also fine).

### Packages
The notebook directly imports:
- `networkx`

It also uses the project modules in `src/`.

Depending on your implementation inside `src/Plots.py`, you may also need typical plotting/scientific packages (e.g., `matplotlib`, `numpy`, `pandas`). Install whatever your `src/` code imports.

Recommended setup (pick one approach):

#### Option A — pip + venv
```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
pip install -U pip
pip install networkx
# plus any other packages required by src/*
```

#### Option B — conda
```bash
conda create -n csn2526 python=3.11
conda activate csn2526
pip install networkx
# plus any other packages required by src/*
```

---

## How to Run

### Run in VS Code
1. Open the folder: `g:\Il mio Drive\University\CSN\CSN_2526_PROJECT`
2. Select the Python interpreter associated with your environment.
3. Open `CSN_2526_Project.ipynb`
4. Run cells top-to-bottom.

### Run in Jupyter
```bash
jupyter notebook
# then open CSN_2526_Project.ipynb
```

---

## Where to Change Experiment Settings

In the notebook, parameters are defined in the “DEFINING PARAMETERS FOR THE SIMULATIONS” cell, e.g.:
- `N`, `m`
- `steps`
- `beta`, `gamma`
- `w`, `p_rand`
- `gamma_val`

Adjust those values and re-run the relevant sections.

---

## What to Read Next

- For methodology/theory: refer to the report.
- For implementation details: inspect `src/Epidemic.py`,`src/Simulations.py`, `src/Metrics.py`, and `src/Plots.py`.
