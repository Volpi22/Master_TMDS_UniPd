import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_rff_comparative_analysis(df, D_VALUES, DIFFICULTIES):
    """
    Generate comprehensive comparative plots for Synthetic Datasets.
    
    This function produces two main figures per difficulty level:
    1. Accuracy Analysis: AUC vs D, Approximation Error, and Distribution Boxplots.
    2. Resource Analysis: Training/Inference Time and Memory usage.

    

    Parameters
    ----------
    df : pd.DataFrame
        The aggregated results DataFrame containing columns for Method, D, AUC, Time, etc.
    D_VALUES : list
        The list of RFF component counts tested (x-axis for plots).
    DIFFICULTIES : list
        List of difficulty levels to iterate through ('easy', 'medium', 'hard').
    """
    
    # Create visualizations for each difficulty level
    for DIFFICULTY in DIFFICULTIES:
        print(f"\n{'='*80}")
        print(f"Analysis for {DIFFICULTY.upper()} Difficulty")
        print(f"{'='*80}")
        
        # Filter data for this specific difficulty context
        df_diff = df[df['Difficulty'] == DIFFICULTY]
        df_kde = df_diff[df_diff['Method'] == 'Exact_KDE']
        df_gnb = df_diff[df_diff['Method'] == 'Gaussian_NB']
        df_rff = df_diff[df_diff['Method'] == 'RFF_KDE']
        
        # =====================================================================
        # FIGURE 1: Accuracy Analysis (AUC)
        # 3 Subplots: Absolute AUC, Error relative to Baseline, Boxplot distribution
        # =====================================================================
        fig1 = plt.figure(figsize=(18, 5))
        gs1 = fig1.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
        
        # --- 1.1: AUC vs Number of Components (D) ---
        ax1 = fig1.add_subplot(gs1[0, 0])
        rff_auc_mean = df_rff.groupby('D')['AUC'].mean()
        rff_auc_std = df_rff.groupby('D')['AUC'].std()
        
        # Plot RFF Curve with Error Bars
        ax1.errorbar(
            rff_auc_mean.index, rff_auc_mean.values,
            yerr=rff_auc_std.values,
            marker='o', capsize=5, label='RFF-KDE'
        )
        
        # Plot Exact KDE Baseline (Target Accuracy)
        ax1.axhline(
            df_kde['AUC'].mean(), color='red', linestyle='--',
            label='Exact KDE'
        )
        # Shaded region represents standard deviation across seeds
        ax1.fill_between(
            D_VALUES,
            df_kde['AUC'].mean() - df_kde['AUC'].std(),
            df_kde['AUC'].mean() + df_kde['AUC'].std(),
            color='red', alpha=0.2
        )
        
        # Plot Gaussian NB Baseline (Lower Bound / Speed Baseline)
        ax1.axhline(
            df_gnb['AUC'].mean(), color='green', linestyle=':',
            label='Gaussian NB'
        )
        ax1.fill_between(
            D_VALUES,
            df_gnb['AUC'].mean() - df_gnb['AUC'].std(),
            df_gnb['AUC'].mean() + df_gnb['AUC'].std(),
            color='green', alpha=0.2
        )
        
        ax1.set_xlabel('Number of RFF Components (D)')
        ax1.set_ylabel('AUC')
        ax1.set_title('AUC vs RFF Components')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log') # Log scale because D varies by orders of magnitude
        
        # --- 1.2: Approximation Error (RFF - Exact) ---
        ax2 = fig1.add_subplot(gs1[0, 1])
        exact_auc_mean = df_kde['AUC'].mean()
        # We calculate the deviation from the "Ground Truth" (Exact KDE)
        auc_error = df_rff.groupby('D')['AUC'].mean() - exact_auc_mean
        auc_error_std = df_rff.groupby('D')['AUC'].std()
        
        ax2.errorbar(
            auc_error.index, auc_error.values,
            yerr=auc_error_std.values,
            marker='o', capsize=5, color='green'
        )
        ax2.axhline(
            0, color='red', linestyle='--', label='Exact KDE baseline',
            linewidth=2
        )
        ax2.set_xlabel('Number of RFF Components (D)')
        ax2.set_ylabel('AUC Error (RFF - Exact)')
        ax2.set_title('Approximation Error')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # --- 1.3: AUC Distribution Boxplot ---
        ax3 = fig1.add_subplot(gs1[0, 2])
        df_rff_copy = df_rff.copy()
        df_rff_copy['D_str'] = df_rff_copy['D'].astype(str)
        
        # Boxplot to visualize outliers and quartile spreads across seeds
        sns.boxplot(
            data=df_rff_copy, x='D_str', y='AUC', ax=ax3, color='skyblue'
        )
        ax3.axhline(
            df_kde['AUC'].mean(), color='red', linestyle='--', label='Exact KDE'
        )
        ax3.axhline(
            df_gnb['AUC'].mean(), color='green', linestyle=':', label='Gaussian NB'
        )
        ax3.set_xlabel('RFF Components (D)')
        ax3.set_ylabel('AUC')
        ax3.set_title('AUC Distribution')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        fig1.suptitle(f"AUC Analysis - {DIFFICULTY.capitalize()} Difficulty", 
              fontsize=16, y=1.02) 
        plt.savefig(f'img/auc_analysis_{DIFFICULTY}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # FIGURE 2: Computational Performance Metrics
        # 2x2 Grid: Train Time, Inference Time, Train Memory, Inference Memory
        # =====================================================================
        fig2 = plt.figure(figsize=(14, 10))
        gs2 = fig2.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
        
        # Helper to plot metric comparisons
        def plot_metric(ax, col_name, ylabel, title, color):
            # RFF Data
            mean = df_rff.groupby('D')[col_name].mean()
            std = df_rff.groupby('D')[col_name].std()
            ax.errorbar(mean.index, mean.values, yerr=std.values,
                        marker='s', capsize=5, label='RFF-KDE', color=color)
            
            # KDE Baseline
            kde_m = df_kde[col_name].mean()
            kde_s = df_kde[col_name].std()
            ax.axhline(kde_m, color='red', linestyle='--', label='Exact KDE')
            ax.fill_between(D_VALUES, kde_m - kde_s, kde_m + kde_s, color='red', alpha=0.2)
            
            # GNB Baseline
            gnb_m = df_gnb[col_name].mean()
            gnb_s = df_gnb[col_name].std()
            ax.axhline(gnb_m, color='green', linestyle=':', label='Gaussian NB')
            ax.fill_between(D_VALUES, gnb_m - gnb_s, gnb_m + gnb_s, color='green', alpha=0.2)
            
            ax.set_xlabel('Number of RFF Components (D)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log') # Log y-scale is crucial for time comparison (orders of mag diff)

        # 2.1: Training Time
        plot_metric(fig2.add_subplot(gs2[0, 0]), 'Train_Time_sec', 'Training Time (sec)', 'Training Time vs D', 'blue')
        
        # 2.2: Inference Time
        plot_metric(fig2.add_subplot(gs2[0, 1]), 'Infer_Time_sec', 'Inference Time (sec)', 'Inference Time vs D', 'orange')
        
        # 2.3: Training Memory
        plot_metric(fig2.add_subplot(gs2[1, 0]), 'Train_Memory_MB', 'Training Memory (MB)', 'Training Memory Usage vs D', 'purple')
        
        # 2.4: Inference Memory
        plot_metric(fig2.add_subplot(gs2[1, 1]), 'Infer_Memory_MB', 'Inference Memory (MB)', 'Inference Memory Usage vs D', 'brown')
        
        fig2.suptitle(f"Performance Metrics - {DIFFICULTY.capitalize()} Difficulty", 
              fontsize=16, y=1.02)
        plt.savefig(f'img/performance_metrics_{DIFFICULTY}.png', dpi=150, bbox_inches='tight')
        plt.show()

def plot_feature_target_correlation(X, y):
    """
    Visualize feature relevance to the target class.
    
    1. Bar Chart: Overall correlation strength.
    2. Heatmap: Top correlated features and their relationships.

    
    """
    df = pd.DataFrame(X.values, columns=[f"Feat_{i}" for i in range(X.shape[1])])
    df['Class'] = y
    
    # Calculate Pearson correlation
    correlation = df.corr()['Class'].drop('Class').sort_values(ascending=False)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Bar plot of correlations
    ax = axes[0]
    colors = ['green' if x > 0 else 'red' for x in correlation.values]
    correlation.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('Correlation with Class', fontsize=12)
    ax.set_title('Feature Correlation with Target', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    # 2. Heatmap of top correlations
    ax = axes[1]
    top_features = correlation.abs().nlargest(10).index.tolist()
    top_corr_matrix = df[top_features + ['Class']].corr()
    sns.heatmap(top_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                ax=ax, cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
    ax.set_title('Top 10 Features Correlation Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('img/feature_target_correlation.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nFeature-Target Correlation:")
    print(correlation)

def plot_rff_comparative_analysis_magic(df_magic, D_VALUES_MAGIC):
    """
    Create comparative figures specifically for the MAGIC Gamma Telescope dataset.
    
    Identical logic to `plot_rff_comparative_analysis` but handles potentially
    different D values and checks for data availability.
    """
    
    # Extract data by method
    df_kde = df_magic[df_magic['Method'] == 'Exact_KDE'].copy()
    df_gnb = df_magic[df_magic['Method'] == 'Gaussian_NB'].copy()
    df_rff = df_magic[(df_magic['Method'] == 'RFF_KDE') & (df_magic['D'] > 0)].copy()
    
    has_kde = not df_kde.empty
    has_gnb = not df_gnb.empty
    
    if df_rff.empty:
        print("❌ ERROR: No RFF_KDE data found. Cannot plot.")
        return
    
    D_VALUES_MAGIC = sorted(df_rff['D'].unique())
    
    # =================================================================
    # FIGURE 1: Accuracy Analysis
    # =================================================================
    fig1 = plt.figure(figsize=(18, 5))
    gs1 = fig1.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    
    # 1.1: AUC vs D
    ax1 = fig1.add_subplot(gs1[0, 0])
    rff_auc_mean = df_rff.groupby('D')['AUC'].mean()
    rff_auc_std = df_rff.groupby('D')['AUC'].std()
    ax1.errorbar(rff_auc_mean.index, rff_auc_mean.values, yerr=rff_auc_std.values, 
                 marker='o', capsize=5, label='RFF-KDE')
    
    if has_kde:
        kde_auc_mean = df_kde['AUC'].mean()
        kde_auc_std = df_kde['AUC'].std()
        ax1.axhline(kde_auc_mean, color='red', linestyle='--', label='Exact KDE')
        ax1.fill_between(D_VALUES_MAGIC, kde_auc_mean - kde_auc_std, 
                         kde_auc_mean + kde_auc_std, color='red', alpha=0.2)
    
    if has_gnb:
        gnb_auc_mean = df_gnb['AUC'].mean()
        gnb_auc_std = df_gnb['AUC'].std()
        ax1.axhline(gnb_auc_mean, color='green', linestyle=':', label='Gaussian NB')
        ax1.fill_between(D_VALUES_MAGIC, gnb_auc_mean - gnb_auc_std,
                         gnb_auc_mean + gnb_auc_std, color='green', alpha=0.2)
    
    ax1.set_xlabel('Number of RFF Components (D)')
    ax1.set_ylabel('AUC')
    ax1.set_title('AUC vs RFF Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # 1.2: AUC Error (Deviation from Exact KDE)
    ax2 = fig1.add_subplot(gs1[0, 1])
    if has_kde:
        exact_auc_mean = df_kde['AUC'].mean()
        auc_error = df_rff.groupby('D')['AUC'].mean() - exact_auc_mean
        auc_error_std = df_rff.groupby('D')['AUC'].std()
        ax2.errorbar(auc_error.index, auc_error.values, yerr=auc_error_std.values,
                     marker='o', capsize=5, color='green')
        ax2.axhline(0, color='red', linestyle='--', linewidth=2, label='Exact KDE baseline')
        ax2.set_ylabel('AUC Error (RFF - Exact)')
        ax2.set_title('Approximation Error')
    else:
        ax2.text(0.5, 0.5, 'No baseline data available', 
                 ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Approximation Error (N/A)')
    
    ax2.set_xlabel('Number of RFF Components (D)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # 1.3: AUC Boxplot
    ax3 = fig1.add_subplot(gs1[0, 2])
    df_rff_copy = df_rff.copy()
    df_rff_copy['D_str'] = df_rff_copy['D'].astype(str)
    sns.boxplot(data=df_rff_copy, x='D_str', y='AUC', ax=ax3, color='skyblue')
    if has_kde:
        ax3.axhline(df_kde['AUC'].mean(), color='red', linestyle='--', label='Exact KDE')
    if has_gnb:
        ax3.axhline(df_gnb['AUC'].mean(), color='green', linestyle=':', label='Gaussian NB')
    
    ax3.set_xlabel('RFF Components (D)')
    ax3.set_ylabel('AUC')
    ax3.set_title('AUC Distribution')
    ax3.tick_params(axis='x', rotation=45)
    
    fig1.suptitle("AUC Analysis - MAGIC Dataset", fontsize=16, y=1.02)
    plt.savefig('img/auc_analysis_magic.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # =================================================================
    # FIGURE 2: Computational Performance Metrics
    # =================================================================
    fig2 = plt.figure(figsize=(14, 10))
    gs2 = fig2.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
    
    # Reusing logic for 2x2 grid plotting
    # (Metric plotting logic replicated from first function for independence)
    
    # 2.1 Training Time
    ax4 = fig2.add_subplot(gs2[0, 0])
    train_mean = df_rff.groupby('D')['Train_Time_sec'].mean()
    train_std = df_rff.groupby('D')['Train_Time_sec'].std()
    ax4.errorbar(train_mean.index, train_mean.values, yerr=train_std.values, 
                 marker='s', label='RFF-KDE', color='blue')
    if has_kde: ax4.axhline(df_kde['Train_Time_sec'].mean(), color='red', linestyle='--', label='Exact KDE')
    if has_gnb: ax4.axhline(df_gnb['Train_Time_sec'].mean(), color='green', linestyle=':', label='Gaussian NB')
    ax4.set_title('Training Time vs D'); ax4.set_yscale('log'); ax4.set_xscale('log'); ax4.grid(True, alpha=0.3)
    
    # 2.2 Inference Time
    ax5 = fig2.add_subplot(gs2[0, 1])
    infer_mean = df_rff.groupby('D')['Infer_Time_sec'].mean()
    infer_std = df_rff.groupby('D')['Infer_Time_sec'].std()
    ax5.errorbar(infer_mean.index, infer_mean.values, yerr=infer_std.values, 
                 marker='s', label='RFF-KDE', color='orange')
    if has_kde: ax5.axhline(df_kde['Infer_Time_sec'].mean(), color='red', linestyle='--', label='Exact KDE')
    if has_gnb: ax5.axhline(df_gnb['Infer_Time_sec'].mean(), color='green', linestyle=':', label='Gaussian NB')
    ax5.set_title('Inference Time vs D'); ax5.set_yscale('log'); ax5.set_xscale('log'); ax5.grid(True, alpha=0.3)
    
    # 2.3 Training Memory
    ax6 = fig2.add_subplot(gs2[1, 0])
    tmem_mean = df_rff.groupby('D')['Train_Memory_MB'].mean()
    tmem_std = df_rff.groupby('D')['Train_Memory_MB'].std()
    ax6.errorbar(tmem_mean.index, tmem_mean.values, yerr=tmem_std.values, 
                 marker='s', label='RFF-KDE', color='purple')
    if has_kde: ax6.axhline(df_kde['Train_Memory_MB'].mean(), color='red', linestyle='--', label='Exact KDE')
    if has_gnb: ax6.axhline(df_gnb['Train_Memory_MB'].mean(), color='green', linestyle=':', label='Gaussian NB')
    ax6.set_title('Training Memory vs D'); ax6.set_yscale('log'); ax6.set_xscale('log'); ax6.grid(True, alpha=0.3)
    
    # 2.4 Inference Memory
    ax7 = fig2.add_subplot(gs2[1, 1])
    imem_mean = df_rff.groupby('D')['Infer_Memory_MB'].mean()
    imem_std = df_rff.groupby('D')['Infer_Memory_MB'].std()
    ax7.errorbar(imem_mean.index, imem_mean.values, yerr=imem_std.values, 
                 marker='s', label='RFF-KDE', color='brown')
    if has_kde: ax7.axhline(df_kde['Infer_Memory_MB'].mean(), color='red', linestyle='--', label='Exact KDE')
    if has_gnb: ax7.axhline(df_gnb['Infer_Memory_MB'].mean(), color='green', linestyle=':', label='Gaussian NB')
    ax7.set_title('Inference Memory vs D'); ax7.set_yscale('log'); ax7.set_xscale('log'); ax7.grid(True, alpha=0.3)

    fig2.suptitle(f"Performance Metrics - MAGIC Dataset", fontsize=16, y=1.02)
    plt.savefig('img/performance_metrics_magic.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_rff_comparative_analysis_scaling(results_scaling):
    """
    Create scaling analysis figure comparing Inference/Training time vs Dataset Size.
    
    This visualizes the algorithmic complexity differences:
    - Exact KDE inference is O(N) (linear with dataset size).
    - RFF KDE inference is O(1) (independent of dataset size, only depends on D).
    """
    
    df_scaling = pd.DataFrame(results_scaling)

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

    # 1. Inference Time vs Dataset Size
    ax1 = fig.add_subplot(gs[0, 0])
    
    for method, color, marker, label in [
        ('Exact_KDE', 'red', 'o', 'Exact KDE'), 
        ('RFF_KDE', 'blue', 's', 'RFF-KDE (D=log(n))')
    ]:
        df_method = df_scaling[df_scaling['Method'] == method]
        mean = df_method.groupby('Size')['Infer_Time_sec'].mean()
        std = df_method.groupby('Size')['Infer_Time_sec'].std()
        
        ax1.errorbar(mean.index, mean.values, yerr=std.values,
                    marker=marker, capsize=5, label=label, color=color)

    ax1.set_xlabel('Dataset Size (n)')
    ax1.set_ylabel('Inference Time (sec)')
    ax1.set_title('Inference Time vs Dataset Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Training Time vs Dataset Size
    ax2 = fig.add_subplot(gs[0, 1])
    
    for method, color, marker, label in [
        ('Exact_KDE', 'red', 'o', 'Exact KDE'), 
        ('RFF_KDE', 'blue', 's', 'RFF-KDE (D=log(n))')
    ]:
        df_method = df_scaling[df_scaling['Method'] == method]
        mean = df_method.groupby('Size')['Train_Time_sec'].mean()
        std = df_method.groupby('Size')['Train_Time_sec'].std()
        
        ax2.errorbar(mean.index, mean.values, yerr=std.values,
                    marker=marker, capsize=5, label=label, color=color)

    ax2.set_xlabel('Dataset Size (n)')
    ax2.set_ylabel('Training Time (sec)')
    ax2.set_title('Training Time vs Dataset Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Scaling Analysis: Exact KDE vs RFF-KDE', fontsize=16, y=1.02)
    plt.savefig('img/rff_comparative_analysis_scaling.png', dpi=150, bbox_inches='tight')
    plt.show()

def scaling_summary_table(results_scaling, sizes):
    """
    Print a tabular summary of the scaling experiment, calculating speedups and accuracy retention.
    """
    df_scaling = pd.DataFrame(results_scaling)
    summary_table = []
    
    for size in sizes:
        df_size = df_scaling[df_scaling['Size'] == size]
        df_kde_size = df_size[df_size['Method'] == 'Exact_KDE']
        df_rff_size = df_size[df_size['Method'] == 'RFF_KDE']
        
        D_value = df_rff_size['D'].iloc[0]
        
        summary_table.append({
            'Size': size,
            'D': D_value,
            'KDE_Train_Time': df_kde_size['Train_Time_sec'].mean(),
            'RFF_Train_Time': df_rff_size['Train_Time_sec'].mean(),
            'KDE_Infer_Time': df_kde_size['Infer_Time_sec'].mean(),
            'RFF_Infer_Time': df_rff_size['Infer_Time_sec'].mean(),
            'Train_Speedup': df_kde_size['Train_Time_sec'].mean() / df_rff_size['Train_Time_sec'].mean(),
            'Infer_Speedup': df_kde_size['Infer_Time_sec'].mean() / df_rff_size['Infer_Time_sec'].mean(),
            'KDE_AUC': df_kde_size['AUC'].mean(),
            'RFF_AUC': df_rff_size['AUC'].mean(),
            'AUC_Diff': df_rff_size['AUC'].mean() - df_kde_size['AUC'].mean()
        })

    summary_df = pd.DataFrame(summary_table)
    print(summary_df.round(4).to_string(index=False))

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print(f"As dataset size increases from {min(sizes)} to {max(sizes)}:")
    print(f"- RFF-KDE uses D=log(n) features, ranging from {summary_df['D'].min()} to {summary_df['D'].max()}")
    print(f"- Training speedup ranges from {summary_df['Train_Speedup'].min():.2f}x to {summary_df['Train_Speedup'].max():.2f}x")
    print(f"- Inference speedup ranges from {summary_df['Infer_Speedup'].min():.2f}x to {summary_df['Infer_Speedup'].max():.2f}x")
    print(f"- AUC difference (RFF - KDE) ranges from {summary_df['AUC_Diff'].min():.4f} to {summary_df['AUC_Diff'].max():.4f}")
    print("="*80)

if __name__ == "__main__":
    pass