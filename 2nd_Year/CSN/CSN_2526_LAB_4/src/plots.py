import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.api import het_white
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm

from src.utils import LANG_DICT

def preliminary_plots_k2():
    """
    Generate 4 preliminary plots for ⟨k²⟩ vs n for each language:
        1. Scatter ⟨k²⟩ vs n (linear scale)
        2. log-log scatter ⟨k²⟩ vs n
        3. Averaged ⟨k²⟩ vs n (linear scale)
        4. Averaged ⟨k²⟩ vs n with theoretical bounds (R-style)
    
    These plots help visualize the relationship between sentence length (n)
    and the mean squared degree (⟨k²⟩) in dependency trees.
    
    Plots are saved to img/Preliminary_Plots/{language}_preliminary_plots_k2.png
    """
    # Create output directory for images
    os.makedirs("img/Preliminary_Plots", exist_ok=True)
    
    # Process each language separately
    for lang in LANG_DICT.keys():
        lang_label = LANG_DICT[lang]
        filepath = f"data/dependency_metrics/{lang_label}_dependency_tree_metrics.csv"
        
        # Read the dependency metrics data
        df = pd.read_csv(filepath)
        df = df.sort_values("n")  # Sort by sentence length for better visualization
        
        # Group by sentence length (n) and compute mean values
        mean_df = df.groupby("n", as_index=False).mean(numeric_only=True)

        # Create a 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Preliminary plots for ⟨k²⟩ ({lang_label})", fontsize=14)

        # Top-left: Scatter plot (linear scale)
        sns.scatterplot(x="n", y="⟨k2⟩", data=df, s=12, ax=axes[0, 0])
        axes[0, 0].set_xlabel("Sentence length (n)")
        axes[0, 0].set_ylabel("⟨k²⟩")
        axes[0, 0].set_title("Scatter plot")
        axes[0, 0].grid(True, alpha=0.3)

        # Top-right: Scatter plot (log–log scale)
        # Helps identify power-law relationships: if log(⟨k²⟩) ~ α·log(n), then ⟨k²⟩ ~ n^α
        sns.scatterplot(x=np.log(df["n"]), y=np.log(df["⟨k2⟩"]), s=12, ax=axes[0, 1])
        axes[0, 1].set_xlabel("log(n)")
        axes[0, 1].set_ylabel("log(⟨k²⟩)")
        axes[0, 1].set_title("Log-log scatter")
        axes[0, 1].grid(True, alpha=0.3)

        # Bottom-left: Averaged ⟨k²⟩ vs n (linear scale)
        axes[1, 0].plot(mean_df["n"], mean_df["⟨k2⟩"], marker="o", linestyle="-", color="green", label="Mean ⟨k²⟩")
        axes[1, 0].set_xlabel("Sentence length (n)")
        axes[1, 0].set_ylabel("Mean ⟨k²⟩")
        axes[1, 0].set_title("Averaged ⟨k²⟩")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Bottom-right: Averaged ⟨k²⟩ with theoretical bounds (R-style)
        axes[1, 1].plot(df["n"], df["⟨k2⟩"], marker=".", linestyle="", color="lightgray", alpha=0.5, label="Data points")
        axes[1, 1].plot(mean_df["n"], mean_df["⟨k2⟩"], marker="o", linestyle="-", color="green", linewidth=2, label="Mean ⟨k²⟩")
        n_mean = mean_df["n"].values
        # Red line: (1 - 1/n)*(5 - 6/n)
        red_line_mean = (1 - 1/n_mean) * (5 - 6/n_mean)
        axes[1, 1].plot(n_mean, red_line_mean, color="red", linewidth=1.5, label="(1 - 1/n)(5 - 6/n)")
        # Blue line: lower bound (4 - 6/n)
        blue_lower_mean = 4 - 6/n_mean
        axes[1, 1].plot(n_mean, blue_lower_mean, color="blue", linewidth=1.5, linestyle="--", label="4 - 6/n")
        # Blue line: upper bound (n - 1)
        blue_upper_mean = n_mean - 1
        axes[1, 1].plot(n_mean, blue_upper_mean, color="blue", linewidth=1.5, linestyle="--", label="n - 1")
        
        # Set y-axis limit to focus on data range
        y_max = max(mean_df["⟨k2⟩"].max(), red_line_mean.max()) * 1.2
        axes[1, 1].set_ylim(0, y_max)
        
        axes[1, 1].set_xlabel("Sentence length (n)")
        axes[1, 1].set_ylabel("Mean ⟨k²⟩")
        axes[1, 1].set_title("Averaged ⟨k²⟩ with theoretical bounds")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        # Adjust layout to prevent overlapping labels
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save the figure
        plt.savefig(f"img/Preliminary_Plots/{lang_label}_preliminary_plots_k2.png", dpi=300)
        plt.close() 

def plot_model_comparison(x, y, fitted_results, metrics, lang_name, save_path):
    """Plot model fits comparison (base vs extended models)"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Base models
    ax1.scatter(x, y, label="Empirical ⟨k²⟩", s=12, alpha=0.5, color="lightgray", zorder=1)
    
    base_models = ['null', 'model1', 'model2', 'model3', 'model4', 'model5']
    line_styles = ["-", "--", "-.", ":", "-", "-."]
    alphas = [1, 1, 1, 1, 0.7, 1]
    labels = ["Null", "M1: (n/2)^b", "M2: an^b", "M3: ae^(cn)", "M4: a·log(n)", "M5: an^b*e^(cn)"]
    
    for model, style, alpha, label in zip(base_models, line_styles, alphas, labels):
        x_fit = fitted_results[model]['x_fit']
        y_fit = fitted_results[model]['y_fit']
        r2 = metrics[model]['R2']
        aic = metrics[model]['AIC']
        ax1.plot(x_fit, y_fit, style, linewidth=2, alpha=alpha, 
                label=f"{label} (R²={r2:.3f}, AIC={aic:.1f})")
    
    ax1.set_xlabel("Sentence length (n)")
    ax1.set_ylabel("⟨k²⟩")
    ax1.set_title("Base models")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Extended models
    ax2.scatter(x, y, label="Empirical ⟨k²⟩", s=12, alpha=0.5, color="lightgray", zorder=1)
    
    extended_models = ['null', 'model1p', 'model2p', 'model3p', 'model4p', 'model5p']
    labels_ext = ["Null", "M1+: (n/2)^b+d", "M2+: an^b+d", "M3+: ae^(cn)+d", "M4+: a·log(n)+d", "M5+: an^b*e^(cn)+d"]
    
    for model, style, alpha, label in zip(extended_models, line_styles, alphas, labels_ext):
        x_fit = fitted_results[model]['x_fit']
        y_fit = fitted_results[model]['y_fit']
        r2 = metrics[model]['R2']
        aic = metrics[model]['AIC']
        ax2.plot(x_fit, y_fit, style, linewidth=2, alpha=alpha,
                label=f"{label} (R²={r2:.3f}, AIC={aic:.1f})")
    
    ax2.set_xlabel("Sentence length (n)")
    ax2.set_ylabel("⟨k²⟩")
    ax2.set_title("Extended models (+ d)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f"⟨k²⟩ vs n model comparison ({lang_name})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_residuals_analysis(x, y, fitted_results, lang_name, save_path):
    """
    Plot residuals analysis for all models and save White test results to a CSV.
    
    For each model, this function generates:
    1. A residuals vs. fitted values plot.
    2. A scale-location plot (sqrt(|residuals|) vs. fitted).
    3. It performs the White test for heteroskedasticity and adds the results
       to the plot and a summary CSV file.
    
    Args:
        x (np.ndarray): Independent variable data.
        y (np.ndarray): Dependent variable data.
        fitted_results (dict): Dictionary with fitted values for each model.
        lang_name (str): Name of the language being analyzed.
        save_path (str): Path to save the plot image.
    """
    model_order = ['null', 'model1', 'model2', 'model3', 'model4', 'model5',
                   'model1p', 'model2p', 'model3p', 'model4p', 'model5p']
    model_names = ['Null Model', 'Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5',
                   'Model 1+', 'Model 2+', 'Model 3+', 'Model 4+', 'Model 5+']
    
    num_models = len(model_order)
    fig, axes = plt.subplots(num_models, 2, figsize=(14, 5 * num_models))
    fig.suptitle(f'Residuals Analysis ({lang_name})', fontsize=16, y=1.0)

    white_test_results = {'language': lang_name}

    for i, (model_key, model_name) in enumerate(zip(model_order, model_names)):
        y_pred = fitted_results[model_key]['pred']
        residuals = y - y_pred
        sqrt_abs_residuals = np.sqrt(np.abs(residuals))
        
        # Plot 1: Residuals vs. Fitted
        ax1 = axes[i, 0]
        ax1.scatter(y_pred, residuals, alpha=0.7, s=12)
        ax1.axhline(0, color='red', linestyle='--')
        ax1.set_title(f'{model_name}: Residuals vs. Fitted')
        ax1.set_xlabel('Fitted values')
        ax1.set_ylabel('Residuals')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scale-Location with White Test
        ax2 = axes[i, 1]
        ax2.scatter(y_pred, sqrt_abs_residuals, alpha=0.7, s=12)
        
        # Add LOWESS trend line
        smoothed = lowess(sqrt_abs_residuals, y_pred)
        ax2.plot(smoothed[:, 0], smoothed[:, 1], color='red')

        # Perform White test for heteroskedasticity
        exog = sm.add_constant(np.column_stack((x, x**2)))
        white_test = het_white(residuals, exog)
        white_stat, white_pvalue = white_test[0], white_test[1]
        
        # Store results
        white_test_results[f'{model_key}_stat'] = white_stat
        white_test_results[f'{model_key}_pvalue'] = white_pvalue
        
        ax2.set_title(f'{model_name}: Scale-Location\nWhite Stat: {white_stat:.2f}, p-value: {white_pvalue:.5f}')
        ax2.set_xlabel('Fitted values')
        ax2.set_ylabel('√|Residuals|')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    # Save White test results to CSV
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "white_test_results.csv")
    
    # Create a DataFrame for the current language
    df_results = pd.DataFrame([white_test_results])
    
    # Check if file exists to append or write new
    if os.path.exists(csv_path):
        # Read existing data, drop the language if it's already there, and append
        df_existing = pd.read_csv(csv_path)
        df_existing = df_existing[df_existing['language'] != lang_name]
        df_final = pd.concat([df_existing, df_results], ignore_index=True)
        df_final.to_csv(csv_path, index=False)
    else:
        # If file doesn't exist, write with header
        df_results.to_csv(csv_path, index=False)

def plot_best_fit(params_df, metrics_df, delta_aic_df):
    """
    Generate plots showing empirical data with the best fitting model for each language.
    The best model is selected based on the lowest ΔAIC value.
    
    Args:
        params_df: DataFrame with model parameters
        metrics_df: DataFrame with model metrics
        delta_aic_df: DataFrame with ΔAIC values (languages as rows, models as columns)
    
    Saves plots to img/Best_Fit/{language}_best_fit.png
    """
    # Local import to break circular dependency
    from src.models import define_models

    os.makedirs("img/Best_Fit", exist_ok=True)
    
    # Define model functions
    models = define_models()
    
    for lang in LANG_DICT.values():
        # Load empirical data
        filepath = f"data/dependency_metrics/{lang}_dependency_tree_metrics.csv"
        df_raw = pd.read_csv(filepath)
        
        # Calculate 10th and 90th percentiles for central 80% of data (on raw data)
        n_p10 = df_raw["n"].quantile(0.10)
        n_p90 = df_raw["n"].quantile(0.90)
        
        x = df_raw["n"].values
        y = df_raw["⟨k2⟩"].values

        # Create a smooth x-axis for the fitted line
        x_fit = np.linspace(x.min(), x.max(), 500)
        
        # Find best model (lowest ΔAIC)
        best_model = delta_aic_df.loc[lang].idxmin()
        delta_aic_value = delta_aic_df.loc[lang, best_model]
        
        # Get model parameters
        params_row = params_df[params_df['language'] == lang].iloc[0]
        
        # Get fitted values based on model type
        if best_model == 'null':
            y_fit = models['null'](x_fit) 
            model_label = "Null: (1-1/n)(5-6/n)"
            params_text = "No parameters"
        elif best_model == 'model1':
            b = params_row['model1_b']
            y_fit = models['model1'](x_fit, b)
            model_label = f"Model 1: (n/2)^b"
            params_text = f"b = {b:.4f}"
        elif best_model == 'model2':
            a, b = params_row['model2_a'], params_row['model2_b']
            y_fit = models['model2'](x_fit, a, b)
            model_label = f"Model 2: a·n^b"
            params_text = f"a = {a:.4f}, b = {b:.4f}"
        elif best_model == 'model3':
            a, c = params_row['model3_a'], params_row['model3_c']
            y_fit = models['model3'](x_fit, a, c)
            model_label = f"Model 3: a·e^(c·n)"
            params_text = f"a = {a:.4f}, c = {c:.6f}"
        elif best_model == 'model4':
            a = params_row['model4_a']
            y_fit = models['model4'](x_fit, a)
            model_label = f"Model 4: a·log(n)"
            params_text = f"a = {a:.4f}"
        elif best_model == 'model5':
            a, b, c = params_row['model5_a'], params_row['model5_b'], params_row['model5_c']
            y_fit = models['model5'](x_fit, a, b, c)
            model_label = f"Model 5: a·n^b·e^(c·n)"
            params_text = f"a = {a:.4f}, b = {b:.4f}, c = {c:.6f}"
        elif best_model == 'model1p':
            b, d = params_row['model1p_b'], params_row['model1p_d']
            y_fit = models['model1p'](x_fit, b, d)
            model_label = f"Model 1+: (n/2)^b + d"
            params_text = f"b = {b:.4f}, d = {d:.4f}"
        elif best_model == 'model2p':
            a, b, d = params_row['model2p_a'], params_row['model2p_b'], params_row['model2p_d']
            y_fit = models['model2p'](x_fit, a, b, d)
            model_label = f"Model 2+: a·n^b + d"
            params_text = f"a = {a:.4f}, b = {b:.4f}, d = {d:.4f}"
        elif best_model == 'model3p':
            a, c, d = params_row['model3p_a'], params_row['model3p_c'], params_row['model3p_d']
            y_fit = models['model3p'](x_fit, a, c, d)
            model_label = f"Model 3+: a·e^(c·n) + d"
            params_text = f"a = {a:.4f}, c = {c:.6f}, d = {d:.4f}"
        elif best_model == 'model4p':
            a, d = params_row['model4p_a'], params_row['model4p_d']
            y_fit = models['model4p'](x_fit, a, d)
            model_label = f"Model 4+: a·log(n) + d"
            params_text = f"a = {a:.4f}, d = {d:.4f}"
        elif best_model == 'model5p':
            a, b, c, d = params_row['model5p_a'], params_row['model5p_b'], params_row['model5p_c'], params_row['model5p_d']
            y_fit = models['model5p'](x_fit, a, b, c, d)
            model_label = f"Model 5+: a·n^b·e^(c·n) + d"
            params_text = f"a = {a:.4f}, b = {b:.4f}, c = {c:.6f}, d = {d:.4f}"
        
        # Get metrics for best model
        model_metrics = metrics_df[(metrics_df['language'] == lang) & (metrics_df['model'] == best_model)].iloc[0]
        r2 = model_metrics['R2']
        aic = model_metrics['AIC']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Linear scale
        ax1.scatter(x, y, s=12, alpha=0.5, color='lightgray', label='Empirical data', zorder=1)
        ax1.plot(x_fit, y_fit, color='green', linewidth=2.5, label=f'Best fit: {model_label}', zorder=2)
        
        # Add vertical lines for central 80%
        ax1.axvline(n_p10, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Central 80% range')
        ax1.axvline(n_p90, color='red', linestyle=':', linewidth=2, alpha=0.7)
        
        ax1.set_xlabel('Sentence length (n)', fontsize=12)
        ax1.set_ylabel('⟨k²⟩', fontsize=12)
        ax1.set_title(f'{lang.capitalize()} - Best Model Fit (Linear Scale)', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add text box with model info
        textstr = f'{params_text}\nR² = {r2:.4f}\nAIC = {aic:.2f}\nΔAIC = {delta_aic_value:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Right plot: Log-log scale
        ax2.scatter(x, y, s=12, alpha=0.5, color='lightgray', label='Empirical data', zorder=1)
        ax2.plot(x_fit, y_fit, color='green', linewidth=2.5, label=f'Best fit: {model_label}', zorder=2)
        
        # Add vertical lines for central 80%
        ax2.axvline(n_p10, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Central 80% range')
        ax2.axvline(n_p90, color='red', linestyle=':', linewidth=2, alpha=0.7)
        
        ax2.set_xlabel('log(Sentence length)', fontsize=12)
        ax2.set_ylabel('log(⟨k²⟩)', fontsize=12)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_title(f'{lang.capitalize()} - Best Model Fit (Log-Log Scale)', fontsize=14)
        ax2.legend(fontsize=10, loc ='lower right')
        ax2.grid(True, alpha=0.3, which='both')
        
        # Add text box with model info
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f'img/Best_Fit/{lang}_best_fit.png', dpi=300)
        plt.close(fig)