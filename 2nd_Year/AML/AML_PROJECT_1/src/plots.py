import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import numerical_cols, special_cols
from scipy import stats
import pandas as pd
from collections import Counter

###################### PLOT CONTINUOUS VARIABLES WITH Q-Q PLOTS ######################
def plot_continuous_distributions_with_qq(dataset):
    """
    Plot barplots, KDE plots, and Q-Q plots for all numerical variables comparing target classes.
    Each row contains: barplot, KDE plot, Q-Q plot for class 0, Q-Q plot for class 1.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        Dataset to visualize (must contain 'num' column as target)
    """
    continuous_cols = list(numerical_cols.keys()) + list(special_cols.keys())
    
    # Set color palette for target classes (0=no disease, 1=disease)
    sns.set_palette(['#ff826e', 'red'])
    
    # Create the subplots (one row per feature, 4 columns: barplot, KDE, Q-Q for class 0, Q-Q for class 1)
    fig, ax = plt.subplots(len(continuous_cols), 4, figsize=(25, 5*len(continuous_cols)), 
                          gridspec_kw={'width_ratios': [1, 2, 1.5, 1.5]})
    
    # Handle case where there's only one numerical feature
    if len(continuous_cols) == 1:
        ax = ax.reshape(1, -1)
    
    # Loop through each numerical feature
    for i, col in enumerate(continuous_cols):
        # 1. Barplot showing the mean value of the feature for each target category
        graph = sns.barplot(data=dataset, x="num", y=col, ax=ax[i, 0])
        ax[i, 0].set_xlabel('Heart Disease', fontsize=11)
        ax[i, 0].set_ylabel(col, fontsize=11)
        
        # Add mean values as labels on the barplot
        for cont in graph.containers:
            graph.bar_label(cont, fmt='         %.3g')
        
        # 2. KDE plot showing the distribution of the feature for each target category
        sns.kdeplot(data=dataset[dataset["num"]==0], x=col, fill=True, linewidth=2, 
                   ax=ax[i, 1], label='0')
        sns.kdeplot(data=dataset[dataset["num"]==1], x=col, fill=True, linewidth=2, 
                   ax=ax[i, 1], label='1')
        ax[i, 1].set_yticks([])  # Remove y-axis ticks (density values not needed)
        ax[i, 1].set_xlabel(col, fontsize=11)
        ax[i, 1].legend(title='Heart Disease', loc='upper right')
        
        # 3. Q-Q plot for class 0 (no disease)
        stats.probplot(dataset[dataset["num"]==0][col].dropna(), dist="norm", plot=ax[i, 2])
        ax[i, 2].set_title(f'{col} - Class 0 (No Disease)', fontsize=10, fontweight='bold')
        ax[i, 2].get_lines()[0].set_color('#ff826e')
        ax[i, 2].get_lines()[0].set_markersize(4)
        
        # 4. Q-Q plot for class 1 (disease)
        stats.probplot(dataset[dataset["num"]==1][col].dropna(), dist="norm", plot=ax[i, 3])
        ax[i, 3].set_title(f'{col} - Class 1 (Disease)', fontsize=10, fontweight='bold')
        ax[i, 3].get_lines()[0].set_color('red')
        ax[i, 3].get_lines()[0].set_markersize(4)

        
    
    # Set the title for the entire figure
    #plt.suptitle('Numerical Features: Distribution and Normality Analysis', fontsize=22, fontweight='bold')
    plt.tight_layout()
    plt.savefig('img/continuous_distributions_with_qq.png', dpi=100, bbox_inches='tight')
    plt.show()

######################## CORRELATION PLOT HEATMAP #####################
def plot_correlation_heatmap(dataset):
    """
    Plots a heatmap of the correlation matrix for numerical features in the dataset.

    Parameters:
    -----------
    dataset : pd.DataFrame
        Dataset containing numerical features
    """

    continuous_cols = list(numerical_cols.keys()) + list(special_cols.keys())

    # Compute correlation matrix
    corr = dataset[continuous_cols].corr()
    
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    # Set title
    # plt.title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold')
    
    # Save figure to file
    plt.savefig("img/correlation_heatmap.png", dpi=100, bbox_inches='tight')
    plt.show()

###################### BOXPLOT OF THE AUC VALUES ######################
def plot_auc_boxplots(results):
    """
    Plots boxplots of AUC scores for different models.

    Parameters:
    -----------
    results : dict
        Dictionary containing model statistics with 'auc_scores' list for each model
    """
    # Prepare data for plotting
    model_names = list(results.keys())
    auc_data = [results[model]['auc_scores'] for model in model_names]

    # Create boxplot figure
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(auc_data, labels=model_names, patch_artist=True)
    
    # Customize colors for each model
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Set axis labels and title
    plt.ylabel('AUC Scores', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    # plt.title('AUC Score Distribution by Model', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=15, ha='right')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save figure to file
    plt.savefig("img/auc_boxplots.png", dpi=100)
    plt.show()

############ DISTRIBUTION OF THE HYPERPARAMETERS VALUES ###############

def plot_hyperparameter_distributions(results):
    """
    Plots the distributions of hyperparameter values for different models in separate subplots.
    Uses histograms for continuous hyperparameters and barplots for discrete ones.

    Parameters:
    -----------
    results : dict
        Dictionary containing model statistics with 'auc_scores' and optionally 'best_param' lists
    """
    # Define hyperparameter names and colors for each model
    hyperparameter_info = {
        'Logistic Regression': {'name': 'C (Regularization)', 'color': 'lightblue', 'type': 'continuous'},
        'Decision Trees': {'name': 'Max Depth', 'color': 'lightgreen', 'type': 'discrete'},
        'SVM': {'name': 'C (Regularization)', 'color': 'lightcoral', 'type': 'continuous'}
    }
    
    # Select only models that have hyperparameter data to plot
    models_to_plot = [m for m in results.keys() if 'best_param' in results[m] and results[m]['best_param']]
    
    # Get number of models to plot
    n_models = len(models_to_plot)
    if n_models == 0:
        print("No hyperparameter data to plot.")
        return
    
    # Create subplots (one for each model with hyperparameters)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    if n_models == 1:
        axes = [axes]  # Make iterable for single plot
    
    # Plot histogram or barplot for each model
    for ax, model_name in zip(axes, models_to_plot):
        # Get hyperparameter values and filter out None values
        best_params = results[model_name]['best_param']
        best_params_filtered = [p for p in best_params if p is not None]
        
        # Get hyperparameter name, color, and type from info dictionary
        hyperparam_name = hyperparameter_info[model_name]['name']
        color = hyperparameter_info[model_name]['color']
        param_type = hyperparameter_info[model_name]['type']
        
        # Only plot if there are valid hyperparameter values
        if best_params_filtered:
            if param_type == 'discrete':
                # For discrete parameters (like max_depth), use barplot
                unique_vals, counts = np.unique(best_params_filtered, return_counts=True)
                ax.bar(unique_vals, counts, alpha=0.7, color=color, edgecolor='black', width=0.6)
                
                # Set x-axis to show integer values only
                ax.set_xticks(unique_vals)
                
            else:
                # For continuous parameters, use histogram
                ax.hist(best_params_filtered, bins=20, alpha=0.7, color=color, edgecolor='black')
        
        # Set labels and title
        ax.set_xlabel(hyperparam_name, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        
        # Add grid
        ax.grid(axis='y', alpha=0.3)
    
    # Add overall title
    # plt.suptitle('Distribution of Best Hyperparameter Values', fontsize=14, fontweight='bold', y=1.02)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save figure to file
    plt.savefig("img/hyperparameter_distributions.png", dpi=100, bbox_inches='tight')
    plt.show()


###################### PLOT MOST MISCLASSIFIED OBSERVATIONS ######################
def plot_most_misclassified_observations(results, top_n=10):
    """
    Plot the observations that have been misclassified the most across all test rounds.
    
    For each model, counts how many times each observation was misclassified across
    all rounds and displays the top N most frequently misclassified observations.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results with 'misclassified_indices' for each model
        (output from compare_model_statistics function)
    top_n : int, default=10
        Number of top misclassified observations to display
    
    Returns:
    --------
    dict
        Dictionary with model names as keys and DataFrames containing the top N
        most misclassified observations with their counts and percentages
    """
    
    # Extract model names (excluding models without misclassification tracking)
    model_names = [model for model in results.keys() if 'misclassified_indices' in results[model]]
    n_models = len(model_names)
    
    if n_models == 0:
        print("No misclassification data available in results")
        return {}
    
    # Create figure with subplots (2 rows x 2 columns for 4 models)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Dictionary to store top misclassified observations for each model
    top_misclassified = {}
    
    # Colors for each model
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7b731']
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        
        # Get all misclassified indices across all rounds
        all_misclassified = []
        for round_misclassified in results[model_name]['misclassified_indices']:
            all_misclassified.extend(round_misclassified)
        
        # Count frequency of each observation being misclassified
        misclass_counts = Counter(all_misclassified)
        
        # Get total number of rounds
        n_rounds = len(results[model_name]['misclassified_indices'])
        
        # Get top N most misclassified observations
        most_common = misclass_counts.most_common(top_n)
        
        if len(most_common) == 0:
            ax.text(0.5, 0.5, 'No misclassifications', 
                   ha='center', va='center', fontsize=14)
            ax.set_title(model_name, fontsize=14, fontweight='bold')
            ax.axis('off')
            continue
        
        # Create DataFrame with results
        obs_indices = [obs for obs, _ in most_common]
        counts = [count for _, count in most_common]
        percentages = [(count / n_rounds) * 100 for count in counts]
        
        df_misclass = pd.DataFrame({
            'Observation Index': obs_indices,
            'Misclassification Count': counts,
            'Percentage': percentages
        })
        
        top_misclassified[model_name] = df_misclass
        
        # Create bar plot
        bars = ax.barh(range(len(obs_indices)), counts, color=colors[idx], 
                      alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Customize y-axis with observation indices
        ax.set_yticks(range(len(obs_indices)))
        ax.set_yticklabels([f'Obs {obs}' for obs in obs_indices], fontsize=10)
        
        # Add value labels on bars
        for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{count} ({pct:.1f}%)',
                   ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Set labels and title
        ax.set_xlabel('Number of Misclassifications', fontsize=11, fontweight='bold')
        ax.set_ylabel('Observation', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}\nTop {top_n} Most Misclassified Observations', 
                    fontsize=12, fontweight='bold')
        
        # Add grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Invert y-axis so highest count is at top
        ax.invert_yaxis()
        
        # Set x-axis limit with some padding
        max_count = max(counts) if counts else 1
        ax.set_xlim(0, max_count * 1.15)
    
    # Add overall title
    fig.suptitle(f'Most Frequently Misclassified Observations (out of {n_rounds} rounds)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig("img/most_misclassified_observations.png", dpi=100, bbox_inches='tight')
    plt.show()
    
    return top_misclassified


def get_misclassified_observations_details(dataset, top_misclassified_dict):
    """
    Retrieve the actual data for the most frequently misclassified observations.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        The original dataset with all features
    top_misclassified_dict : dict
        Dictionary output from plot_most_misclassified_observations containing
        DataFrames with observation indices for each model
    
    Returns:
    --------
    dict
        Dictionary with model names as keys and DataFrames containing the full
        observation details for the most misclassified samples
    """
    import pandas as pd
    
    misclassified_details = {}
    
    for model_name, df_misclass in top_misclassified_dict.items():
        # Get observation indices
        obs_indices = df_misclass['Observation Index'].tolist()
        
        # Retrieve observations from dataset
        obs_data = dataset.loc[obs_indices].copy()
        
        # Add misclassification statistics
        obs_data = obs_data.merge(
            df_misclass.set_index('Observation Index'),
            left_index=True,
            right_index=True,
            how='left'
        )
        
        # Reorder columns to show statistics first
        stat_cols = ['Misclassification Count', 'Percentage']
        other_cols = [col for col in obs_data.columns if col not in stat_cols]
        obs_data = obs_data[stat_cols + other_cols]
        
        misclassified_details[model_name] = obs_data
        
        # Save to CSV
        filename = f"data/most_misclassified_{model_name.lower().replace(' ', '_')}.csv"
        obs_data.to_csv(filename, index=True)
        print(f"Saved {model_name} most misclassified observations to {filename}")
    
    return misclassified_details

if __name__ == "__main__":
    pass

if __name__ == "__main__":
    pass