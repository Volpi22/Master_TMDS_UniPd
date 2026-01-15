import numpy as np
import pandas as pd
from scipy.stats import (norm, uniform, lognorm, dweibull, wald, laplace)
from sklearn.calibration import LabelEncoder
from ucimlrepo import fetch_ucirepo

def magic_gamma_data():
    """
    Load the Heart Disease dataset from the UCI Machine Learning Repository.
    
    This dataset serves as a real-world benchmark for binary classification tasks.
    ID 159 corresponds to the Cleveland Heart Disease database.

    Returns
    -------
    X : pandas.DataFrame
        The feature matrix containing patient attributes (age, chol, etc.).
    y : pandas.DataFrame
        The target variable (num). Note: The original target is 0-4 (ordinal),
        where 0 is no disease and 1-4 are degrees of disease.
    """
    # Fetch dataset from UCI repository (ID 159 = Heart Disease)
    magic_gamma = fetch_ucirepo(id=159)
    X = magic_gamma.data.features
    y = magic_gamma.data.targets 

    return X, y

def prepare_magic_gamma(X, y):
    """
    Preprocess the Heart Disease dataset for binary classification.

    Converts the multiclass target (0-4) into a binary target (0 vs 1+),
    handles pandas/numpy data types, and encodes labels.

    Parameters
    ----------
    X : array-like or DataFrame
        Feature matrix.
    y : array-like or Series
        Target vector.

    Returns
    -------
    X : array-like
        The passed feature matrix (passed through).
    y : np.ndarray
        Binary target vector where 0 = No Disease, 1 = Disease.
    """
    if hasattr(y, 'values'):
        y = y.values.ravel()
        
    # The original dataset has values 0, 1, 2, 3, 4.
    # Standard practice is to treat 0 as negative and >0 as positive.
    # LabelEncoder will normalize this, but manual binarization might be safer
    # if strictly binary 0/1 is required before encoding.
    # Here we assume the downstream task handles the encoding or distinct classes.
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Force binary if multiple classes exist (common in this specific dataset usage)
    if len(np.unique(y)) > 2:
         y = (y > 0).astype(int)
         
    return X, y

def get_mixed_values(u_vals, distributions):
    """
    Apply a weighted mixture of Inverse CDFs (Percentile Point Functions) to uniform data.
    
    This is the core of the Copula transformation:
    Uniform Data (U) -> Inverse CDF -> Non-Normal Data.
    
    

    Parameters
    ----------
    u_vals : np.ndarray
        Input array of uniform values in range (0, 1).
    distributions : list of tuples
        A list of distributions to mix, format: (name, scipy_func, kwargs, weight).

    Returns
    -------
    np.ndarray
        The transformed data having the shape of the mixed distribution.
    """
    mixed = np.zeros_like(u_vals)
    total_weight = 0.0
    
    for name, dist_func, kwargs, weight in distributions:
        if weight <= 1e-6: continue 

        # Handle custom distributions that aren't in SciPy
        if name == 'custom_bimodal':
            # Create bimodality by mapping [0, 0.5] to left mode and [0.5, 1] to right mode
            sep = kwargs.get('separation', 3.0)
            vals_left = norm.ppf(u_vals * 2) - sep
            vals_right = norm.ppf((u_vals - 0.5) * 2) + sep
            vals = np.where(u_vals < 0.5, vals_left, vals_right)
            
        elif name == 'custom_peaked':
            # A very sharp peak (low variance normal)
            vals = norm.ppf(u_vals) * 0.5 
            
        else:
            # Standard SciPy Inverse CDF transform
            vals = dist_func.ppf(u_vals, **kwargs)
        
        # Standardize component to ensure mixing weights control shape, not scale magnitude
        if np.std(vals) > 0:
            vals = (vals - np.mean(vals)) / np.std(vals)
            
        mixed += vals * weight
        total_weight += weight
        
    return mixed / total_weight if total_weight > 0 else mixed

def get_random_weights(n, dominance_factor=1.0):
    """
    Generate random weights for distribution mixing using a Dirichlet distribution.
    
    Parameters
    ----------
    n : int
        Number of components.
    dominance_factor : float
        Controls sparsity. Lower values (<1) make the weights sparser 
        (one component dominates), higher values (>1) make them more uniform.
    """
    weights = np.random.dirichlet(np.ones(n) * dominance_factor)
    return weights

def generate_points(difficulty: str, n_samples=1000, random_state=None):
    """
    Generate synthetic classification data with controllable difficulty and non-Gaussian features.
    
    This function uses a Gaussian Copula approach:
    1. Generate correlated multivariate normal data.
    2. Transform marginals to Uniform via CDF.
    3. Transform Uniforms to complex shapes (Bimodal, Skewed, etc.) via Inverse CDF.

    Parameters
    ----------
    difficulty : {'easy', 'medium', 'hard'}
        Controls separation of classes, correlation strength, and noise levels.
    n_samples : int, default=1000
        Total number of samples to generate.
    random_state : int, default=None
        Seed for reproducibility.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples, 10).
    y : np.ndarray
        Target vector of shape (n_samples,).
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_features = 10
    n_classes = 2
    samples_per_class = n_samples // n_classes

    # --- 0. Configuration: Difficulty Settings ---
    if difficulty == "easy":
        rho_min, rho_max = 0.0, 0.2     # Low feature correlation
        separation_factor = 4.0         # Large distance between class means
        non_gaussian_count = 0          # All features are Gaussian (easy for standard Bayes)
        n_informative = 10              # All features contain signal
        
    elif difficulty == "medium":
        rho_min, rho_max = 0.3, 0.5     # Moderate correlation
        separation_factor = 2.0         # Moderate distance
        non_gaussian_count = 8          # Most features are non-Gaussian
        n_informative = 5               # 5 features are pure noise
        
    elif difficulty == "hard":
        rho_min, rho_max = 0.6, 0.8     # High correlation (violates Naive Bayes assumption)
        separation_factor = 0.0         # Means are identical! (Separation relies purely on distribution shape)
        non_gaussian_count = 10         # All features are complex/non-Gaussian
        n_informative = 5               # Half the features are noise
        
    else:
        raise ValueError("Difficulty must be 'easy', 'medium', or 'hard'")

    # Identify which features carry signal vs noise
    feature_indices = np.arange(n_features)
    np.random.shuffle(feature_indices)
    informative_indices = set(feature_indices[:n_informative])
    
    # --- 1. Latent Structure Generation (Covariance) ---
    # We create a covariance matrix to enforce correlations between features.
    rho = np.random.uniform(rho_min, rho_max)
    cov_matrix = np.full((n_features, n_features), rho)
    np.fill_diagonal(cov_matrix, 1.0)
    cov_matrix += np.eye(n_features) * 1e-6 # Numerical stability

    # --- 2. Base Data Generation (Gaussian) ---
    mean_0 = np.zeros(n_features)
    mean_1 = np.zeros(n_features) 
    
    # Apply separation factor only to informative features.
    # For noise features, means remain 0 vs 0.
    for i in range(n_features):
        if i in informative_indices:
            mean_1[i] = separation_factor
        else:
            mean_1[i] = 0.0

    X0 = np.random.multivariate_normal(mean_0, cov_matrix, samples_per_class)
    X1 = np.random.multivariate_normal(mean_1, cov_matrix, samples_per_class)

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(samples_per_class), np.ones(samples_per_class)])

    # --- 3. Define Distribution Pools ---
    # Class 0 tends towards these shapes
    pool_class_0 = [
        ('custom_bimodal', None,   {'separation': 3.5}),
        ('uniform',        uniform,{'loc':-3, 'scale':6}),
        ('dweibull',       dweibull,{'c': 1.0}),
    ]
    
    # Class 1 tends towards these shapes
    pool_class_1 = [
        ('laplace',        laplace, {'loc': 0, 'scale': 0.8}),
        ('lognorm',        lognorm, {'s': 0.6}),
        ('custom_peaked',  None,    {}),
    ]

    # --- 4. Non-Gaussian Transformation (The Copula Step) ---
    # Select features to transform based on `non_gaussian_count`
    transform_candidates = np.arange(n_features)
    np.random.shuffle(transform_candidates)
    indices_to_transform = transform_candidates[:non_gaussian_count]

    for idx in indices_to_transform:
        # A. Probability Integral Transform: Gaussian -> Uniform
        # norm.cdf maps the data to the range [0, 1] preserving rank order
        u_vals = norm.cdf(X[:, idx] - X[:, idx].mean())
        u_vals = np.clip(u_vals, 1e-6, 1 - 1e-6)
        
        mask_0 = (y == 0)
        mask_1 = (y == 1)
        
        # Generate random mixing weights for complexity
        weights_A = get_random_weights(len(pool_class_0), dominance_factor=0.8)
        weights_B = get_random_weights(len(pool_class_1), dominance_factor=0.8)
        
        dist_set_A = [(n, f, k, w) for (n, f, k), w in zip(pool_class_0, weights_A)]
        dist_set_B = [(n, f, k, w) for (n, f, k), w in zip(pool_class_1, weights_B)]

        # B. Inverse Transform: Uniform -> Target Distribution
        if idx in informative_indices:
            # INFORMATIVE FEATURE:
            # Class 0 gets distribution Mix A (e.g., Bimodal)
            # Class 1 gets distribution Mix B (e.g., Skewed)
            # result: The classes are separated by *shape*, not just mean.
            X[mask_0, idx] = get_mixed_values(u_vals[mask_0], dist_set_A)
            X[mask_1, idx] = get_mixed_values(u_vals[mask_1], dist_set_B)
        else:
            # NOISY FEATURE:
            # Both classes get the SAME distribution Mix (indistinguishable).
            # We apply Mix A to the entire column.
            X[:, idx] = get_mixed_values(u_vals, dist_set_A)

    # Add final sensor noise
    X += np.random.normal(0, 0.1, size=X.shape)
    
    return X, y

if __name__ == "__main__":
    pass