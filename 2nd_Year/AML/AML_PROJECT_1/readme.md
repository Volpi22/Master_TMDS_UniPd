# Heart Disease Classification - Code Documentation

## Project Structure

AML_PROJECT_1/
├── src/
│   ├── data_ops.py                  # Data loading, preprocessing, and transformations
│   ├── models.py                    # Model implementations and comparison framework
│   └── plots.py                     # Visualization functions
├── data/
│   └── model_comparison_summary.csv # Model performance summary (generated)
├── img/
│   ├── continuous_distributions_with_qq.png  # Feature distributions with Q-Q plots (generated)
│   ├── correlation_heatmap.png               # Correlation heatmap (generated)
│   ├── auc_boxplots.png                      # AUC comparisons (generated)
│   └── hyperparameter_distributions.png      # Hyperparameter distributions (generated)
├── AML_PROJECT_1.ipynb             # Main analysis notebook
├── requirements.txt                # Python dependencies
└── readme.md                       # This file

---

## Setup Instructions

### Prerequisites
- Python 3.11 or higher  
- Jupyter Notebook

### Installation Steps
1. **Install dependencies:**
   pip install -r requirements.txt

---

## How to Run

### Run the Complete Analysis
Open and execute the Jupyter notebook:

    jupyter notebook AML_PROJECT_1.ipynb

**Execute all cells sequentially.**  
The notebook will:
1. Load and preprocess the UCI Heart Disease dataset  
2. Visualize feature distributions  
3. Train and evaluate 4 models with hyperparameter tuning  
4. Generate comparison visualizations  
5. Save results to `data/` and `img/` folders  

---

## Module Descriptions

### src/data_ops.py

**Functions:**
- load_heart_disease_data(): Fetches UCI Heart Disease dataset  
- preprocess_heart_disease_data(X, y): Cleans data and converts target to binary  
- data_transformation(train_data, test_data): Applies BoxCox/Yeo-Johnson transformations  
- data_standardization(train_data, test_data): Applies z-score normalization  
- find_outliers(dataset): Identifies outliers using IQR method  

**Key Variables:**
- categorical_cols: Dictionary of categorical features  
- numerical_cols: Dictionary of numerical features  
- special_cols: Dictionary of special features (oldpeak)  
- binary_cols: Dictionary of binary features (sex, exang, fbs)  

---

### src/models.py

**Classes:**
- CustomNaiveBayesMod: Custom Naive Bayes classifier for mixed feature types  
  - Handles categorical (Laplace smoothing), binary (Bernoulli), oldpeak (KDE with Gaussian kernel, bandwidth), and numerical features (multivariate Gaussian)  

**Functions:**
- compare_model_statistics(dataset, n_rounds=100):  
  - Performs repeated train-test splits  
  - Tunes hyperparameters via 10-fold CV  
  - Evaluates 4 models (Logistic Regression, Decision Trees, SVM, Custom Naive Bayes)  
  - Returns results dictionary and summary DataFrame  

**Returns:**
- results: Dict with AUC scores and hyperparameters for each model  
- summary_df: DataFrame with mean AUC ± std and hyperparameter statistics  

---

### src/plots.py

**Functions:**
- plot_continuous_distributions_with_qq(dataset): Creates histograms with KDE overlays and Q-Q plots for numerical features by target class  
- plot_correlation_heatmap(dataset): Generates heatmap of feature correlations  
- plot_auc_boxplots(results): Generates boxplots comparing AUC distributions across models  
- plot_hyperparameter_distributions(results): Plots histograms of selected hyperparameter values across iterations  

**Output:**  
All plots are saved to the `img/` folder as PNG files (300 DPI).  

---

## Configuration

### Random Seed
For reproducibility, the random seed is set in the notebook:
    np.random.seed(42)

---

## Generated Outputs

After running the notebook, the following files will be created:

### Data Files
- data/model_comparison_summary.csv: Table with model names, mean AUC scores (± std), and mean hyperparameters (± std)

### Visualizations
- continuous_distributions_with_qq.png: Feature distributions with Q-Q plots split by disease presence  
- correlation_heatmap.png: Correlation matrix of continuous features  
- auc_boxplots.png: Boxplots comparing model performance across 100 iterations  
- hyperparameter_distributions.png: Histograms of tuned hyperparameter values  

---

## Requirements

See requirements.txt for the complete list.

**Key packages:**
- numpy, pandas: Data manipulation  
- scikit-learn: ML algorithms and evaluation  
- matplotlib, seaborn: Visualization  
- feature-engine: Data transformations  
- ucimlrepo: Dataset access
