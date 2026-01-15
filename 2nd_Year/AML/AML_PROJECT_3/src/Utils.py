from .Data import *
import pandas as pd
import seaborn as sns
from scipy.stats import iqr
import numpy as np
import time 
import tracemalloc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def hardness_example(hardness):
    """
    Generate and visualize a sample dataset for a given difficulty level.
    
    This function creates a pairplot to visually inspect the class separation
    and feature distributions.
    
    

    Parameters
    ----------
    hardness : str
        The difficulty level ('easy', 'medium', 'hard').
    """
    X, y = generate_points(hardness)
    feat_names = [f"Feat_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_names)
    df['Class'] = y
    
    # Use a smaller sample for plotting speed if dataset is huge
    if len(df) > 1000:
        df = df.sample(1000, random_state=42)
        
    g = sns.pairplot(df, hue='Class', diag_kind='kde', corner=True)
    g.fig.suptitle(f'Data Distribution: {hardness.capitalize()} Difficulty', y=1.02)
    plt.show()

def visualize_distribution(X, y):
    """
    Visualize the pairwise relationships in a dataset.
    
    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    """
    # Ensure y is properly encoded for visualization legend
    le_vis = LabelEncoder()
    y_encoded = le_vis.fit_transform(y)

    # Convert to DataFrame for Seaborn
    # If X is already a DataFrame, use its column names; else generate generic ones.
    if hasattr(X, 'columns'):
        df = X.copy()
    else:
        df = pd.DataFrame(X, columns=[f"Feat_{i}" for i in range(X.shape[1])])
        
    df['Class'] = y_encoded
    
    # Subsample for performance if needed
    if len(df) > 1000:
        print("Subsampling to 1000 points for visualization...")
        df = df.sample(1000, random_state=42)
        
    sns.pairplot(df, hue='Class', corner=True)
    plt.show()

def measure_execution(func, *args, **kwargs):
    """
    Execute a function while measuring its runtime and peak memory usage.
    
    This wrapper is essential for benchmarking the efficiency of the RFF method
    versus the Exact KDE method.

    

    Parameters
    ----------
    func : callable
        The function to execute (e.g., model.fit or model.predict).
    *args, **kwargs
        Arguments to pass to the function.

    Returns
    -------
    result : Any
        The return value of the executed function.
    time_taken : float
        Execution time in seconds.
    peak_memory_mb : float
        Peak memory usage during execution in Megabytes (MB).
    """
    # Start memory tracking
    tracemalloc.start()
    
    # Start timer
    start_time = time.time()
    
    # Execute function
    result = func(*args, **kwargs)
    
    # Stop timer
    end_time = time.time()
    
    # Get memory stats
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Convert bytes to MB
    peak_memory_mb = peak / (1024 * 1024)
    time_taken = end_time - start_time
    
    return result, time_taken, peak_memory_mb

def silverman_bandwidth(X):
    """
    Calculate the optimal bandwidth for KDE using Silverman's Rule of Thumb.
    
    This rule minimizes the Mean Integrated Squared Error (MISE) assuming
    the underlying distribution is Gaussian. It is robust to outliers because
    it considers both Standard Deviation and Interquartile Range (IQR).
    
    Formula: h = 0.9 * min(std, IQR/1.34) * n^(-1/5)

    Parameters
    ----------
    X : array-like
        Input data array (1D).

    Returns
    -------
    h : float
        The estimated bandwidth.
    """
    n = len(X)
    
    # Standard deviation (ddof=1 for sample std)
    sigma = np.std(X, ddof=1)
    
    # Interquartile range (75th percentile - 25th percentile)
    iqr_val = iqr(X)
    
    # Select the smaller of sigma or normalized IQR to be robust against outliers
    # If IQR is 0 (e.g., heavily discrete data), fallback to sigma
    if iqr_val > 0:
        spread = min(sigma, iqr_val / 1.34)
    else:
        spread = sigma
        
    # Handle edge case: zero variance
    if spread == 0:
        return 1.0
        
    h = 0.9 * spread * (n ** (-1/5))
    return h

if __name__ == "__main__":
    pass