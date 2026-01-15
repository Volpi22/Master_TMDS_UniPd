from .Data import *
from .Classifiers import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from .Utils import measure_execution

def process_single_run(difficulty, seed, D_VALUES):
    """
    Execute a full benchmark run for a specific difficulty level and random seed.
    
    This function generates a synthetic dataset, applies standard scaling, 
    and evaluates three classifier types:
    1. Exact KDE (Baseline High Accuracy)
    2. Gaussian Naive Bayes (Baseline High Speed)
    3. RFF-KDE (Experimental, iterating over various component counts D)

    

    Parameters
    ----------
    difficulty : str
        The complexity of the synthetic dataset ('easy', 'medium', 'hard').
    seed : int
        Random state for reproducibility.
    D_VALUES : list of int
        A list of component counts (D) to test for the RFF approximation.

    Returns
    -------
    list of dict
        A list of dictionaries, where each dictionary contains metrics 
        (AUC, Time, Memory) for a specific model configuration.
    """
    run_results = []
    
    # Generate dataset
    X, y = generate_points(difficulty, n_samples=5000, random_state=seed)
    
    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------
    # RFF relies on distance metrics (RBF kernel), so scaling is crucial.
    # We fit on Train and transform Test to avoid data leakage.
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # ---------------------------------------------------------
    # Baseline 1: Exact KDE
    # ---------------------------------------------------------
    kde_model = KDENaiveBayes(bandwidth='silverman', random_state=seed)
    
    # Measure Fit
    _, train_time_kde, train_mem_kde = measure_execution(kde_model.fit, X_train, y_train)
    # Measure Inference
    y_pred_kde, infer_time_kde, infer_mem_kde = measure_execution(kde_model.predict_proba, X_test)
    
    y_pred_kde = y_pred_kde[:, 1]
    auc_kde = roc_auc_score(y_test, y_pred_kde)
    
    run_results.append({
        'Difficulty': difficulty,
        'Seed': seed,
        'Method': 'Exact_KDE',
        'D': 0,
        'AUC': auc_kde,
        'Train_Time_sec': train_time_kde,
        'Infer_Time_sec': infer_time_kde,
        'Total_Time_sec': train_time_kde + infer_time_kde,
        'Train_Memory_MB': train_mem_kde,
        'Infer_Memory_MB': infer_mem_kde
    })
    
    # ---------------------------------------------------------
    # Baseline 2: Gaussian Naive Bayes
    # ---------------------------------------------------------
    gnb_model = GaussianNB()
    
    _, train_time_gnb, train_mem_gnb = measure_execution(gnb_model.fit, X_train, y_train)
    y_pred_gnb, infer_time_gnb, infer_mem_gnb = measure_execution(gnb_model.predict_proba, X_test)
    
    y_pred_gnb = y_pred_gnb[:, 1]
    auc_gnb = roc_auc_score(y_test, y_pred_gnb)
    
    run_results.append({
        'Difficulty': difficulty,
        'Seed': seed,
        'Method': 'Gaussian_NB',
        'D': 0,
        'AUC': auc_gnb,
        'Train_Time_sec': train_time_gnb,
        'Infer_Time_sec': infer_time_gnb,
        'Total_Time_sec': train_time_gnb + infer_time_gnb,
        'Train_Memory_MB': train_mem_gnb,
        'Infer_Memory_MB': infer_mem_gnb
    })
    
    # ---------------------------------------------------------
    # Experimental: RFF-KDE (Iterating D)
    # ---------------------------------------------------------
    for D in D_VALUES:
        rff_model = RFFNaiveBayes(n_components=D, random_state=seed)
        
        # Measure training (RFF fitting + Mean Embedding computation)
        _, train_time_rff, train_mem_rff = measure_execution(rff_model.fit, X_train, y_train)
        
        # Measure inference (Dot product in Hilbert space)
        y_pred_rff, infer_time_rff, infer_mem_rff = measure_execution(rff_model.predict_proba, X_test)
        
        y_pred_rff = y_pred_rff[:, 1]
        auc_rff = roc_auc_score(y_test, y_pred_rff)
        
        run_results.append({
            'Difficulty': difficulty,
            'Seed': seed,
            'Method': 'RFF_KDE',
            'D': D,
            'AUC': auc_rff,
            'Train_Time_sec': train_time_rff,
            'Infer_Time_sec': infer_time_rff,
            'Total_Time_sec': train_time_rff + infer_time_rff,
            'Train_Memory_MB': train_mem_rff,
            'Infer_Memory_MB': infer_mem_rff
        })
    
    return run_results

# ==========================================
# PARALLEL EXECUTION FUNCTION
# ==========================================

def process_single_run_real(X, y, seed, D, compute_baselines=True):
    """
    Process a single configuration for the Real Dataset (MAGIC Gamma).
    
    Designed for use with `multiprocessing.Pool`. Unlike `process_single_run`,
    this takes a specific `D` value rather than a list, allowing fine-grained
    parallelization over the hyperparameter grid.

    

    Parameters
    ----------
    X, y : array-like
        The complete dataset features and targets.
    seed : int
        Random seed for splitting and initialization.
    D : int
        Number of RFF components.
    compute_baselines : bool
        If True, compute Exact_KDE and Gaussian_NB. 
        (Optimization: Set to False for subsequent D values on the same seed).
    """
    run_results = []

    # Prepare data
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------------------------------------------------
    # Baselines (Conditional Execution)
    # ---------------------------------------------------------
    if compute_baselines:
        # Exact KDE
        kde_model = KDENaiveBayes(bandwidth='silverman', random_state=seed)
        
        _, train_time_kde, train_mem_kde = measure_execution(kde_model.fit, X_train, y_train)
        y_pred_kde, infer_time_kde, infer_mem_kde = measure_execution(kde_model.predict_proba, X_test)
        
        y_pred_kde = y_pred_kde[:, 1]
        auc_kde = roc_auc_score(y_test, y_pred_kde)
        
        run_results.append({
            'Seed': seed,
            'Method': 'Exact_KDE',
            'D': 0,
            'AUC': auc_kde,
            'Train_Time_sec': train_time_kde,
            'Infer_Time_sec': infer_time_kde,
            'Total_Time_sec': train_time_kde + infer_time_kde,
            'Train_Memory_MB': train_mem_kde,
            'Infer_Memory_MB': infer_mem_kde
        })
        
        # Gaussian Naive Bayes
        gnb_model = GaussianNB()
        
        _, train_time_gnb, train_mem_gnb = measure_execution(gnb_model.fit, X_train, y_train)
        y_pred_gnb, infer_time_gnb, infer_mem_gnb = measure_execution(gnb_model.predict_proba, X_test)
        
        y_pred_gnb = y_pred_gnb[:, 1]
        auc_gnb = roc_auc_score(y_test, y_pred_gnb)
        
        run_results.append({
            'Seed': seed,
            'Method': 'Gaussian_NB',
            'D': 0,
            'AUC': auc_gnb,
            'Train_Time_sec': train_time_gnb,
            'Infer_Time_sec': infer_time_gnb,
            'Total_Time_sec': train_time_gnb + infer_time_gnb,
            'Train_Memory_MB': train_mem_gnb,
            'Infer_Memory_MB': infer_mem_gnb
        })

    # ---------------------------------------------------------
    # RFF-KDE for the specific D value
    # ---------------------------------------------------------
    rff_model = RFFNaiveBayes(n_components=int(D), random_state=seed)
    
    _, train_time_rff, train_mem_rff = measure_execution(rff_model.fit, X_train, y_train)
    y_pred_rff, infer_time_rff, infer_mem_rff = measure_execution(rff_model.predict_proba, X_test)
    
    y_pred_rff = y_pred_rff[:, 1]
    auc_rff = roc_auc_score(y_test, y_pred_rff)

    run_results.append({
        'Seed': seed,
        'Method': 'RFF_KDE',
        'D': int(D),
        'AUC': auc_rff,
        'Train_Time_sec': train_time_rff,
        'Infer_Time_sec': infer_time_rff,
        'Total_Time_sec': train_time_rff + infer_time_rff,
        'Train_Memory_MB': train_mem_rff,
        'Infer_Memory_MB': infer_mem_rff
    })

    return run_results

def process_size_run(size, seed, difficulty='medium'):
    """
    Process a single run to test scalability with dataset size (N).
    
    Instead of iterating D, we fix D relative to N to check consistency.
    Heuristic: D = log(N).
    """
    run_results = []
    
    X, y = generate_points(difficulty, n_samples=size, random_state=seed)
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Calculate D = log(n) for RFF theoretical convergence check
    D = int(np.log(size))
    
    # The actual implementation calls follow the same pattern:
    
    # 1. Exact KDE
    kde_model = KDENaiveBayes(bandwidth='silverman', random_state=seed)
    _, t_kde, m_kde = measure_execution(kde_model.fit, X_train, y_train)
    y_pred, t_inf_kde, m_inf_kde = measure_execution(kde_model.predict_proba, X_test)
    auc_kde = roc_auc_score(y_test, y_pred[:, 1])
    
    run_results.append({
        'Size': size, 'Seed': seed, 'Method': 'Exact_KDE', 'D': 0,
        'AUC': auc_kde, 'Train_Time_sec': t_kde, 'Infer_Time_sec': t_inf_kde,
        'Total_Time_sec': t_kde + t_inf_kde, 'Train_Memory_MB': m_kde, 'Infer_Memory_MB': m_inf_kde
    })
    
    # 2. Gaussian NB
    gnb_model = GaussianNB()
    _, t_gnb, m_gnb = measure_execution(gnb_model.fit, X_train, y_train)
    y_pred, t_inf_gnb, m_inf_gnb = measure_execution(gnb_model.predict_proba, X_test)
    auc_gnb = roc_auc_score(y_test, y_pred[:, 1])
    
    run_results.append({
        'Size': size, 'Seed': seed, 'Method': 'Gaussian_NB', 'D': 0,
        'AUC': auc_gnb, 'Train_Time_sec': t_gnb, 'Infer_Time_sec': t_inf_gnb,
        'Total_Time_sec': t_gnb + t_inf_gnb, 'Train_Memory_MB': m_gnb, 'Infer_Memory_MB': m_inf_gnb
    })
    
    # 3. RFF-KDE
    rff_model = RFFNaiveBayes(n_components=D, random_state=seed)
    _, t_rff, m_rff = measure_execution(rff_model.fit, X_train, y_train)
    y_pred, t_inf_rff, m_inf_rff = measure_execution(rff_model.predict_proba, X_test)
    auc_rff = roc_auc_score(y_test, y_pred[:, 1])
    
    run_results.append({
        'Size': size, 'Seed': seed, 'Method': 'RFF_KDE', 'D': D,
        'AUC': auc_rff, 'Train_Time_sec': t_rff, 'Infer_Time_sec': t_inf_rff,
        'Total_Time_sec': t_rff + t_inf_rff, 'Train_Memory_MB': m_rff, 'Infer_Memory_MB': m_inf_rff
    })
    
    return run_results


def create_synthetic_performance_table(df_easy, df_medium, df_hard,
                                       D_VALUES=[1, 2, 5, 11, 25, 58, 131, 295, 665, 1500],
                                       output_path='results/synthetic_performance_comparison.csv'):
    """
    Aggregate results from multiple synthetic runs into a summary table.
    
    Calculates Mean ± Standard Deviation for AUC across all seeds.
    
    

    Parameters
    ----------
    df_easy, df_medium, df_hard : pd.DataFrame
        DataFrames containing raw results for each difficulty level.
    D_VALUES : list
        Specific D values to extract for the table rows.
    output_path : str
        File path to save the raw statistical data.

    Returns
    -------
    pd.DataFrame
        A formatted DataFrame for display (strings with "Mean ± Std").
    """
    
    datasets = {
        'Easy': df_easy,
        'Medium': df_medium,
        'Hard': df_hard
    }
    
    # Define row structure
    methods_config = [
        {'name': 'Gaussian NB', 'type': 'Gaussian_NB', 'D': 0},
        {'name': 'Exact KDE', 'type': 'Exact_KDE', 'D': 0}
    ]
    for D in D_VALUES:
        methods_config.append({'name': f'RFF-KDE (D={D})', 'type': 'RFF_KDE', 'D': D})
        
    raw_rows = []
    formatted_rows = []
    
    for method in methods_config:
        raw_row = {'Method': method['name']}
        formatted_row = {'Method': method['name']}
        
        for difficulty, df in datasets.items():
            # Filter specific method/D configuration
            if method['type'] == 'RFF_KDE':
                subset = df[(df['Method'] == 'RFF_KDE') & (df['D'] == method['D'])]
            else:
                subset = df[df['Method'] == method['type']]
            
            # Compute Statistics
            if len(subset) > 0 and not subset['AUC'].isnull().all():
                auc_values = subset['AUC'].dropna().values
                mean_auc = np.mean(auc_values)
                std_auc = np.std(auc_values, ddof=1) if len(auc_values) > 1 else 0.0
                
                raw_row[f'{difficulty}_Mean'] = mean_auc
                raw_row[f'{difficulty}_Std'] = std_auc
                formatted_row[difficulty] = f'{mean_auc:.4f} ± {std_auc:.4f}'
            else:
                raw_row[f'{difficulty}_Mean'] = np.nan
                raw_row[f'{difficulty}_Std'] = np.nan
                formatted_row[difficulty] = 'N/A'
                
        raw_rows.append(raw_row)
        formatted_rows.append(formatted_row)
        
    # Save and Return
    df_raw = pd.DataFrame(raw_rows)
    df_formatted = pd.DataFrame(formatted_rows)
    
    # Organize columns
    cols_formatted = ['Method', 'Easy', 'Medium', 'Hard']
    df_formatted = df_formatted[cols_formatted]
    
    df_raw.to_csv(output_path, index=False)
    print(f"Synthetic performance table saved to: {output_path}")
    
    return df_formatted


def create_magic_performance_table(df_magic,
                                   D_VALUES=[1, 2, 7, 21, 58, 162, 448, 1242, 3436, 9509],
                                   output_path='results/magic_performance_comparison.csv'):
    """
    Create a performance comparison table for the MAGIC Gamma Telescope dataset.
    
    Similar to the synthetic table, but handles a single dataset context.
    
    Parameters
    ----------
    df_magic : pd.DataFrame
        Results dataframe for MAGIC dataset runs.
    """
    
    methods_config = [
        {'name': 'Gaussian NB', 'type': 'Gaussian_NB', 'D': 0},
        {'name': 'Exact KDE', 'type': 'Exact_KDE', 'D': 0}
    ]
    for D in D_VALUES:
        methods_config.append({'name': f'RFF-KDE (D={D})', 'type': 'RFF_KDE', 'D': D})
        
    raw_rows = []
    formatted_rows = []
    
    for method in methods_config:
        raw_row = {'Method': method['name']}
        formatted_row = {'Method': method['name']}
        
        if method['type'] == 'RFF_KDE':
            subset = df_magic[(df_magic['Method'] == 'RFF_KDE') & (df_magic['D'] == method['D'])]
        else:
            subset = df_magic[df_magic['Method'] == method['type']]
        
        if len(subset) > 0 and not subset['AUC'].isnull().all():
            auc_values = subset['AUC'].dropna().values
            mean_auc = np.mean(auc_values)
            std_auc = np.std(auc_values, ddof=1) if len(auc_values) > 1 else 0.0
            
            raw_row['MAGIC_Mean'] = mean_auc
            raw_row['MAGIC_Std'] = std_auc
            formatted_row['MAGIC'] = f'{mean_auc:.4f} ± {std_auc:.4f}'
        else:
            raw_row['MAGIC_Mean'] = np.nan
            raw_row['MAGIC_Std'] = np.nan
            formatted_row['MAGIC'] = 'N/A'
            
        raw_rows.append(raw_row)
        formatted_rows.append(formatted_row)
        
    df_raw = pd.DataFrame(raw_rows)
    df_formatted = pd.DataFrame(formatted_rows)
    
    df_raw.to_csv(output_path, index=False)
    print(f"MAGIC performance table saved to: {output_path}")
    
    return df_formatted

if __name__ == "__main__":
    pass