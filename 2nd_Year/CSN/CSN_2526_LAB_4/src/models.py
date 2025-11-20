import os
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

from src.utils import LANG_DICT

# ============================================================================
# HETEROSKEDASTIC VARIANCE ESTIMATION
# ============================================================================

def estimate_heteroskedastic_variance(y, y_pred, frac=0.3):
    """
    Estimate observation-specific variance using LOWESS smoothing.
    
    Uses Locally Weighted Scatterplot Smoothing (LOWESS) to estimate variance
    as a smooth function of fitted values, accounting for heteroskedasticity.
    
    Args:
        y: observed values
        y_pred: predicted values from fitted model
        frac: fraction of data used for LOWESS smoothing (default 0.3 = 30%)
              Controls the bandwidth of the smoother - smaller values give
              more flexible fits, larger values give smoother estimates
    
    Returns:
        Array of estimated variances σ²ᵢ for each observation
    """
    residuals = y - y_pred
    
    # Sort by predicted values for smoother interpolation
    sort_idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[sort_idx]
    residuals_sorted = residuals[sort_idx]
    
    # Smooth the squared residuals using LOWESS
    # This estimates E[ε²|ŷ] = Var(ε|ŷ)
    smoothed_var = lowess(
        residuals_sorted**2, 
        y_pred_sorted, 
        frac=frac, 
        return_sorted=False
    )
    
    # Ensure positive variance for numerical stability
    smoothed_var = np.maximum(smoothed_var, 1e-10)
    
    # Restore original order
    variance_estimate = np.empty_like(smoothed_var)
    variance_estimate[sort_idx] = smoothed_var
    
    return variance_estimate

def compute_log_likelihood_heteroskedastic(y, y_pred, variances):
    """
    Compute log-likelihood assuming heteroskedastic normal errors.
    
    Uses the formula:
    log L(θ) = -N/2·log(2π) - 1/2·Σ[log(σ²ᵢ) + (yᵢ - ŷᵢ)²/σ²ᵢ]
    
    where σ²ᵢ are observation-specific variances.
    
    Args:
        y: observed values
        y_pred: predicted values
        variances: estimated variance σ²ᵢ for each observation
    
    Returns:
        log-likelihood value
    """
    n = len(y)
    ll = -0.5 * n * np.log(2 * np.pi) - 0.5 * np.sum(
        np.log(variances) + ((y - y_pred) ** 2) / variances
    )
    return ll 

def compute_metrics_heteroskedastic(y, y_pred, n_params, frac=0.3):
    """
    Compute AIC, BIC, RSS and RSE accounting for heteroskedasticity.
    
    The effective number of parameters includes:
    - Model structural parameters (n_params)
    - LOWESS smoother parameters (≈ N * frac)
    
    Args:
        y: observed values
        y_pred: predicted values from model
        n_params: number of structural parameters in the model (k)
        frac: fraction for LOWESS smoothing (default 0.3)
    
    Returns:
        Dictionary with:
        - 'AIC': Akaike Information Criterion
        - 'BIC': Bayesian Information Criterion
        - 'RSS': Residual Sum of Squares
        - 'RSE': Residual Standard Error
        - 'log_likelihood': heteroskedasticity-corrected log-likelihood
        - 'variances': estimated observation-specific variances
    """
    n = len(y)
    
    # Estimate heteroskedastic variance using LOWESS
    variances = estimate_heteroskedastic_variance(y, y_pred, frac=frac)
    
    # Effective degrees of freedom for LOWESS smoother
    # The smoother uses approximately N * frac parameters
    additional_params = n * frac
    
    # Compute heteroskedasticity-corrected log-likelihood
    ll = compute_log_likelihood_heteroskedastic(y, y_pred, variances)
    
    # Total parameters = model params + LOWESS params
    total_params = n_params + additional_params
    
    # Compute information criteria
    aic = 2 * total_params - 2 * ll
    bic = total_params * np.log(n) - 2 * ll
    
    # RSS for reference (not used in AIC/BIC with heteroskedasticity)
    rss = np.sum((y - y_pred) ** 2)

    den = max(n - total_params, 1)
    rse = np.sqrt(rss / den)
    
    return {
        'AIC': aic,
        'BIC': bic,
        'RSS': rss,
        'RSE': rse,
        'log_likelihood': ll,
        'variances': variances
    }

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def define_models():
    """
    Define all 11 models for fitting ⟨k²⟩ vs sentence length n.
    
    Models include:
    - Null model (Model 0): theoretical baseline with no free parameters
    - Base models (Models 1-5): fundamental functional forms
    - Extended models (Models 1+-5+): base models with additional offset parameter d
    
    Returns:
        Dictionary mapping model names to functions
    """
    
    # Null model - theoretical baseline
    def null_model(x):
        """Model 0: f(n) = (1 - 1/n)(5 - 6/n)"""
        return (1 - 1/x) * (5 - 6/x)

    # Base models
    def model1(x, b):
        """Model 1: f(n) = (n/2)^b"""
        return np.power(x / 2, b)

    def model2(x, a, b):
        """Model 2: f(n) = a·n^b (power law)"""
        return a * np.power(x, b)

    def model3(x, a, c):
        """Model 3: f(n) = a·e^(c·n) (exponential)"""
        return a * np.exp(c * x)

    def model4(x, a):
        """Model 4: f(n) = a·log(n) (logarithmic)"""
        return a * np.log(x)
    
    def model5(x, a, b, c):
        """Model 5: f(n) = a·n^b·e^(c·n) (power-exponential)"""
        return a * np.power(x, b) * np.exp(c * x)

    # Extended models with offset parameter d
    def model1_plus(x, b, d):
        """Model 1+: f(n) = (n/2)^b + d"""
        return np.power(x / 2, b) + d

    def model2_plus(x, a, b, d):
        """Model 2+: f(n) = a·n^b + d"""
        return a * np.power(x, b) + d

    def model3_plus(x, a, c, d):
        """Model 3+: f(n) = a·e^(c·n) + d"""
        return a * np.exp(c * x) + d

    def model4_plus(x, a, d):
        """Model 4+: f(n) = a·log(n) + d"""
        return a * np.log(x) + d
    
    def model5_plus(x, a, b, c, d):
        """Model 5+: f(n) = a·n^b·e^(c·n) + d"""
        return a * np.power(x, b) * np.exp(c * x) + d
    
    return {
        'null': null_model,
        'model1': model1,
        'model2': model2,
        'model3': model3,
        'model4': model4,
        'model5': model5,
        'model1p': model1_plus,
        'model2p': model2_plus,
        'model3p': model3_plus,
        'model4p': model4_plus,
        'model5p': model5_plus
    }

# ============================================================================
# INITIAL PARAMETER ESTIMATION
# ============================================================================

def get_initial_parameters(x, y):
    """
    Compute initial parameter guesses using linear regression in transformed spaces.
    
    For each model, we transform the problem to make it linear, then use
    least squares regression to obtain initial parameter estimates. These
    initial guesses help the non-linear optimizer converge reliably.
    
    Transformations:
    - Model 1: log(y) = b·log(n/2)
    - Model 2: log(y) = log(a) + b·log(n)
    - Model 3: log(y) = log(a) + c·n
    - Model 4: y = a·log(n)
    - Model 5: log(y) = log(a) + b·log(n) + c·n (multiple regression)
    
    Args:
        x: sentence lengths (n)
        y: observed ⟨k²⟩ values
    
    Returns:
        Dictionary with initial parameters for each base model
    """
    log_x = np.log(x)
    log_x_half = np.log(x / 2)
    log_y = np.log(y)

    # Model 1: fit log(y) vs log(n/2)
    p1 = Polynomial.fit(log_x_half, log_y, 1)
    b_init_m1 = p1.convert().coef[1]
    
    # Model 2: fit log(y) vs log(n)
    p2 = Polynomial.fit(log_x, log_y, 1)
    b_init_m2 = p2.convert().coef[1]
    a_init_m2 = np.exp(p2.convert().coef[0])

    # Model 3: fit log(y) vs n
    p3 = Polynomial.fit(x, log_y, 1)
    c_init_m3 = p3.convert().coef[1]
    a_init_m3 = np.exp(p3.convert().coef[0])
    
    # Model 4: fit y vs log(n)
    p4 = Polynomial.fit(log_x, y, 1)
    a_init_m4 = p4.convert().coef[1]
    
    # Model 5: multiple linear regression with log(n) and n as predictors
    X = np.column_stack([log_x, x])
    X = sm.add_constant(X)
    model = sm.OLS(log_y, X)
    results = model.fit()
    
    a_init_m5 = np.exp(results.params[0])
    b_init_m5 = results.params[1]
    c_init_m5 = results.params[2]
    
    return {
        'm1': [b_init_m1],
        'm2': [a_init_m2, b_init_m2],
        'm3': [a_init_m3, c_init_m3],
        'm4': [a_init_m4],
        'm5': [a_init_m5, b_init_m5, c_init_m5]
    }

# ============================================================================
# MODEL FITTING
# ============================================================================

def fit_all_models(x, y, models, initial_params):
    """
    Fit all 11 models using non-linear least squares.
    
    Uses scipy.optimize.curve_fit (Levenberg-Marquardt algorithm) to minimize
    residual sum of squares for each model. Extended models (+d) use parameters
    from base models as initial guesses, with d initialized to 0.
    
    Args:
        x: sentence lengths (n)
        y: observed ⟨k²⟩ values
        models: dictionary of model functions from define_models()
        initial_params: dictionary of initial parameter guesses
    
    Returns:
        Dictionary with fitted parameters, predictions, and number of parameters
        for each model
    """
    results = {}
    x_fit = np.linspace(min(x), max(x), 100)
    
    # Null model (no fitting required)
    results['null'] = {
        'pred': models['null'](x),
        'params': None,
        'n_params': 0,
        'x_fit': x_fit,
        'y_fit': models['null'](x_fit)
    }
    
    # Base models - fit using initial parameter guesses
    popt_m1, _ = curve_fit(models['model1'], x, y, p0=initial_params['m1'], maxfev=5000)
    results['model1'] = {
        'pred': models['model1'](x, *popt_m1),
        'params': popt_m1,
        'n_params': 1,
        'x_fit': x_fit,
        'y_fit': models['model1'](x_fit, *popt_m1)
    }
    
    popt_m2, _ = curve_fit(models['model2'], x, y, p0=initial_params['m2'], maxfev=5000)
    results['model2'] = {
        'pred': models['model2'](x, *popt_m2),
        'params': popt_m2,
        'n_params': 2,
        'x_fit': x_fit,
        'y_fit': models['model2'](x_fit, *popt_m2)
    }
    
    popt_m3, _ = curve_fit(models['model3'], x, y, p0=initial_params['m3'], maxfev=5000)
    results['model3'] = {
        'pred': models['model3'](x, *popt_m3),
        'params': popt_m3,
        'n_params': 2,
        'x_fit': x_fit,
        'y_fit': models['model3'](x_fit, *popt_m3)
    }
    
    popt_m4, _ = curve_fit(models['model4'], x, y, p0=initial_params['m4'], maxfev=5000)
    results['model4'] = {
        'pred': models['model4'](x, *popt_m4),
        'params': popt_m4,
        'n_params': 1,
        'x_fit': x_fit,
        'y_fit': models['model4'](x_fit, *popt_m4)
    }
    
    popt_m5, _ = curve_fit(models['model5'], x, y, p0=initial_params['m5'], maxfev=10000)
    results['model5'] = {
        'pred': models['model5'](x, *popt_m5),
        'params': popt_m5,
        'n_params': 3,
        'x_fit': x_fit,
        'y_fit': models['model5'](x_fit, *popt_m5)
    }
    
    # Extended models - use base model parameters + d=0 as initial guesses
    popt_m1p, _ = curve_fit(models['model1p'], x, y, p0=[popt_m1[0], 0], maxfev=5000)
    results['model1p'] = {
        'pred': models['model1p'](x, *popt_m1p),
        'params': popt_m1p,
        'n_params': 2,
        'x_fit': x_fit,
        'y_fit': models['model1p'](x_fit, *popt_m1p)
    }
    
    popt_m2p, _ = curve_fit(models['model2p'], x, y, p0=[popt_m2[0], popt_m2[1], 0], maxfev=5000)
    results['model2p'] = {
        'pred': models['model2p'](x, *popt_m2p),
        'params': popt_m2p,
        'n_params': 3,
        'x_fit': x_fit,
        'y_fit': models['model2p'](x_fit, *popt_m2p)
    }
    
    popt_m3p, _ = curve_fit(models['model3p'], x, y, p0=[popt_m3[0], popt_m3[1], 0], maxfev=5000)
    results['model3p'] = {
        'pred': models['model3p'](x, *popt_m3p),
        'params': popt_m3p,
        'n_params': 3,
        'x_fit': x_fit,
        'y_fit': models['model3p'](x_fit, *popt_m3p)
    }
    
    popt_m4p, _ = curve_fit(models['model4p'], x, y, p0=[popt_m4[0], 0], maxfev=5000)
    results['model4p'] = {
        'pred': models['model4p'](x, *popt_m4p),
        'params': popt_m4p,
        'n_params': 2,
        'x_fit': x_fit,
        'y_fit': models['model4p'](x_fit, *popt_m4p)
    }
    
    popt_m5p, _ = curve_fit(models['model5p'], x, y, p0=[popt_m5[0], popt_m5[1], popt_m5[2], 0], maxfev=10000)
    results['model5p'] = {
        'pred': models['model5p'](x, *popt_m5p),
        'params': popt_m5p,
        'n_params': 4,
        'x_fit': x_fit,
        'y_fit': models['model5p'](x_fit, *popt_m5p)
    }
    
    return results

# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_all_metrics(y, fitted_results, frac=0.3):
    """
    Compute R², RSS, RSE, AIC, and BIC for all fitted models.
    
    Uses heteroskedasticity-corrected AIC and BIC via LOWESS variance estimation.
    
    Args:
        y: observed ⟨k²⟩ values
        fitted_results: dictionary with model predictions from fit_all_models()
        frac: fraction for LOWESS smoothing (default 0.3)
    
    Returns:
        Dictionary with metrics (R2, RSS, RSE, AIC, BIC, log_likelihood) for each model
    """
    metrics = {}
    n = len(y)
    
    for model_name, result in fitted_results.items():
        y_pred = result['pred']
        n_params = result['n_params']
        
        # R² score
        r2 = r2_score(y, y_pred)
        
        # Heteroskedasticity-corrected AIC and BIC
        hetero_metrics = compute_metrics_heteroskedastic(y, y_pred, n_params, frac=frac)
        
        metrics[model_name] = {
            'R2': r2,
            'RSS': hetero_metrics['RSS'],
            'RSE': hetero_metrics['RSE'],
            'AIC': hetero_metrics['AIC'],
            'BIC': hetero_metrics['BIC'],
            'log_likelihood': hetero_metrics['log_likelihood']
        }
    
    return metrics

# ============================================================================
# DATA EXPORT FUNCTIONS
# ============================================================================

def create_parameters_dataframe(all_results):
    """
    Create DataFrame with fitted model parameters for all languages.
    
    Args:
        all_results: list of dictionaries with fitting results per language
    
    Returns:
        DataFrame with columns for language and all model parameters
    """
    params_data = []
    for result in all_results:
        params_data.append({
            "language": result["language"],
            "model1_b": result["model1_b"],
            "model2_a": result["model2_a"],
            "model2_b": result["model2_b"],
            "model3_a": result["model3_a"],
            "model3_c": result["model3_c"],
            "model4_a": result["model4_a"],
            "model5_a": result["model5_a"],
            "model5_b": result["model5_b"],
            "model5_c": result["model5_c"],
            "model1p_b": result["model1p_b"],
            "model1p_d": result["model1p_d"],
            "model2p_a": result["model2p_a"],
            "model2p_b": result["model2p_b"],
            "model2p_d": result["model2p_d"],
            "model3p_a": result["model3p_a"],
            "model3p_c": result["model3p_c"],
            "model3p_d": result["model3p_d"],
            "model4p_a": result["model4p_a"],
            "model4p_d": result["model4p_d"],
            "model5p_a": result["model5p_a"],
            "model5p_b": result["model5p_b"],
            "model5p_c": result["model5p_c"],
            "model5p_d": result["model5p_d"]
        })
    
    return pd.DataFrame(params_data)

def create_metrics_dataframe(all_results):
    """
    Create DataFrame with metrics (R2, RSS, RSE, AIC, BIC) for all models and languages.
    
    Args:
        all_results: list of dictionaries with fitting results per language
    
    Returns:
        DataFrame in long format with columns: language, model, n_params, R2, RSS, RSE, AIC, BIC
    """
    metrics_data = []
    for result in all_results:
        lang = result["language"]
        
        # List of (model_name, n_params, metrics...)
        models_info = [
            ('null',   0, result['null_R2'],   result['null_RSS'],   result['null_RSE'],   result['null_AIC'],   result['null_BIC']),
            ('model1', 1, result['model1_R2'], result['model1_RSS'], result['model1_RSE'], result['model1_AIC'], result['model1_BIC']),
            ('model2', 2, result['model2_R2'], result['model2_RSS'], result['model2_RSE'], result['model2_AIC'], result['model2_BIC']),
            ('model3', 2, result['model3_R2'], result['model3_RSS'], result['model3_RSE'], result['model3_AIC'], result['model3_BIC']),
            ('model4', 1, result['model4_R2'], result['model4_RSS'], result['model4_RSE'], result['model4_AIC'], result['model4_BIC']),
            ('model5', 3, result['model5_R2'], result['model5_RSS'], result['model5_RSE'], result['model5_AIC'], result['model5_BIC']),
            ('model1p',2, result['model1p_R2'],result['model1p_RSS'],result['model1p_RSE'],result['model1p_AIC'],result['model1p_BIC']),
            ('model2p',3, result['model2p_R2'],result['model2p_RSS'],result['model2p_RSE'],result['model2p_AIC'],result['model2p_BIC']),
            ('model3p',3, result['model3p_R2'],result['model3p_RSS'],result['model3p_RSE'],result['model3p_AIC'],result['model3p_BIC']),
            ('model4p',2, result['model4p_R2'],result['model4p_RSS'],result['model4p_RSE'],result['model4p_AIC'],result['model4p_BIC']),
            ('model5p',4, result['model5p_R2'],result['model5p_RSS'],result['model5p_RSE'],result['model5p_AIC'],result['model5p_BIC']),
        ]
        
        for model_name, n_params, r2, rss, rse, aic, bic in models_info:
            metrics_data.append({
                'language': lang,
                'model': model_name,
                'n_params': n_params,
                'R2': r2,
                'RSS': rss,
                'RSE': rse,
                'AIC': aic,
                'BIC': bic
            })
    
    return pd.DataFrame(metrics_data)

def create_delta_metric_dataframe(metrics_df, metric_name='AIC'):
    """
    Create DataFrame with normalized metric values (Δ_metric = metric - metric_best).
    
    Computes the difference between each model's metric and the best (minimum)
    metric for that language. Models with Δ < 2 have substantial support.
    
    Args:
        metrics_df: DataFrame from create_metrics_dataframe()
        metric_name: 'AIC' or 'BIC'
    
    Returns:
        DataFrame with languages as rows, models as columns, and Δ values as entries
    """
    delta_data = []
    
    for lang in metrics_df['language'].unique():
        lang_data = metrics_df[metrics_df['language'] == lang]
        metric_best = lang_data[metric_name].min()
        
        for _, row in lang_data.iterrows():
            delta = row[metric_name] - metric_best
            delta_data.append({
                'language': lang,
                'model': row['model'],
                metric_name: row[metric_name],
                f'delta_{metric_name}': delta
            })
    
    delta_df = pd.DataFrame(delta_data)
    delta_pivot = delta_df.pivot(index='language', columns='model', values=f'delta_{metric_name}')
    
    # Ensure consistent column order
    model_order = ['null', 'model1', 'model2', 'model3', 'model4', 'model5',
                   'model1p', 'model2p', 'model3p', 'model4p', 'model5p']
    delta_pivot = delta_pivot[model_order]
    
    return delta_pivot

# ============================================================================
# MAIN FITTING FUNCTION
# ============================================================================

def fit_dependency_models_k2(frac=0.3):
    """
    Main function to fit all dependency models for ⟨k²⟩ vs sentence length n.
    
    For each of the 21 languages:
    1. Load dependency tree metrics data
    2. Fit 11 models (null + 5 base + 5 extended)
    3. Compute metrics with heteroskedasticity correction
    4. Generate visualization plots
    5. Save results to CSV files
    
    Args:
        frac: fraction of data used for LOWESS variance estimation (default 0.3 = 30%)
              This controls the smoothness of the variance function
    
    Returns:
        Tuple of (params_df, metrics_df, delta_aic_df, delta_bic_df)
    
    Output files:
        - data/k2_model_parameters.csv: fitted parameters for all models
        - data/k2_model_metrics.csv: R², RSS, RSE, AIC, BIC for all models
        - data/k2_delta_AIC.csv: normalized AIC values (Δ_AIC)
        - data/k2_delta_BIC.csv: normalized BIC values (Δ_BIC)
        - img/Model_Fitting/{lang}_k2_vs_n_model_comparison.png: model fits
        - img/Model_Fitting/Residuals/{lang}_residuals_analysis.png: residual diagnostics
    """

    # Local import to break circular dependency
    from src.plots import plot_model_comparison, plot_residuals_analysis

    # Create output directories
    os.makedirs("img/Model_Fitting", exist_ok=True)
    os.makedirs("img/Model_Fitting/Residuals", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Get model definitions
    models = define_models()
    
    # Store results for all languages
    all_results = []
    
    for lang in LANG_DICT.keys():
        filepath = f"data/dependency_metrics/{LANG_DICT[lang]}_dependency_tree_metrics.csv"
        lang_name = LANG_DICT[lang]
        
        print(f"\nFitting models for {lang_name}...")
        
        # Load and prepare data (group by n and take mean)
        df = pd.read_csv(filepath)
        x = df["n"].values
        y = df["⟨k2⟩"].values
        
        # Get initial parameter guesses from linear regression in transformed space
        initial_params = get_initial_parameters(x, y)
        
        # Fit all models using non-linear least squares
        fitted_results = fit_all_models(x, y, models, initial_params)
        
        # Compute metrics with heteroskedasticity correction
        metrics = compute_all_metrics(y, fitted_results, frac=frac)
        
        # Store results in a dictionary
        result_dict = {"language": lang_name}
        
        # Add null model metrics
        result_dict.update({
            "null_R2": metrics['null']['R2'],
            "null_RSS": metrics['null']['RSS'],
            "null_RSE": metrics['null']['RSE'],
            "null_AIC": metrics['null']['AIC'],
            "null_BIC": metrics['null']['BIC']
        })
        
        # Add parameters and metrics for all base and extended models
        for model_key in ['model1', 'model2', 'model3', 'model4', 'model5',
                          'model1p', 'model2p', 'model3p', 'model4p', 'model5p']:
            params = fitted_results[model_key]['params']
            
            # Store parameters based on model structure
            if model_key == 'model1':
                result_dict["model1_b"] = params[0]
            elif model_key == 'model2':
                result_dict["model2_a"] = params[0]
                result_dict["model2_b"] = params[1]
            elif model_key == 'model3':
                result_dict["model3_a"] = params[0]
                result_dict["model3_c"] = params[1]
            elif model_key == 'model4':
                result_dict["model4_a"] = params[0]
            elif model_key == 'model5':
                result_dict["model5_a"] = params[0]
                result_dict["model5_b"] = params[1]
                result_dict["model5_c"] = params[2]
            elif model_key == 'model1p':
                result_dict["model1p_b"] = params[0]
                result_dict["model1p_d"] = params[1]
            elif model_key == 'model2p':
                result_dict["model2p_a"] = params[0]
                result_dict["model2p_b"] = params[1]
                result_dict["model2p_d"] = params[2]
            elif model_key == 'model3p':
                result_dict["model3p_a"] = params[0]
                result_dict["model3p_c"] = params[1]
                result_dict["model3p_d"] = params[2]
            elif model_key == 'model4p':
                result_dict["model4p_a"] = params[0]
                result_dict["model4p_d"] = params[1]
            elif model_key == 'model5p':
                result_dict["model5p_a"] = params[0]
                result_dict["model5p_b"] = params[1]
                result_dict["model5p_c"] = params[2]
                result_dict["model5p_d"] = params[3]
            
            # Store metrics
            result_dict[f"{model_key}_R2"] = metrics[model_key]['R2']
            result_dict[f"{model_key}_RSS"] = metrics[model_key]['RSS']
            result_dict[f"{model_key}_RSE"] = metrics[model_key]['RSE']
            result_dict[f"{model_key}_AIC"] = metrics[model_key]['AIC']
            result_dict[f"{model_key}_BIC"] = metrics[model_key]['BIC']
        
        all_results.append(result_dict)
        
        # Print best model for this language
        best_aic_model = min(metrics.items(), key=lambda x: x[1]['AIC'])[0]
        print(f"  Best model (AIC): {best_aic_model} (AIC = {metrics[best_aic_model]['AIC']:.2f})")
        
        # Generate visualization plots
        plot_model_comparison(
            x, y, fitted_results, metrics, lang_name,
            f"img/Model_Fitting/{lang}_k2_vs_n_model_comparison.png"
        )
        
        plot_residuals_analysis(
            x, y, fitted_results, lang_name,
            f"img/Model_Fitting/Residuals/{lang}_residuals_analysis.png"
        )
    
    # Create and save DataFrames
    params_df = create_parameters_dataframe(all_results)
    params_df.to_csv("data/k2_model_parameters.csv", index=False)
    print("\nSaved: data/k2_model_parameters.csv")
    
    metrics_df = create_metrics_dataframe(all_results)
    metrics_df.to_csv("data/k2_model_metrics.csv", index=False)
    print("Saved: data/k2_model_metrics.csv")
    
    delta_aic_df = create_delta_metric_dataframe(metrics_df, 'AIC')
    delta_aic_df.to_csv("data/k2_delta_AIC.csv", index=True)
    print("Saved: data/k2_delta_AIC.csv")

    delta_bic_df = create_delta_metric_dataframe(metrics_df, 'BIC')
    delta_bic_df.to_csv("data/k2_delta_BIC.csv", index=True)
    print("Saved: data/k2_delta_BIC.csv")

    return params_df, metrics_df, delta_aic_df, delta_bic_df