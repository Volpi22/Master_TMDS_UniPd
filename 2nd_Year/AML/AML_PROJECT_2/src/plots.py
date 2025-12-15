import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr
from matplotlib.colors import LinearSegmentedColormap

def plot_heatmaps(gammas, coef0s, H1, H2, H3, H4=None):
    """
    Plot three side-by-side heatmaps for kernel analysis:
    - Heatmap 1: E_minus (negative mass / total mass)
    - Heatmap 2: f_neg   (fraction of negative eigenvalues)
    - Heatmap 3: Nneg    (number of negative eigenvalues)
      Special coloring for Nneg = 0 (background) and Nneg > 0 (heatmap).
    If H4 (CPD status) is provided, CPD kernels are marked with stars.

    Parameters:
        gammas (array-like): List of gamma values.
        coef0s (array-like): List of coef0 values.
        H1 (ndarray): Matrix of E_minus values.
        H2 (ndarray): Matrix of f_neg values.
        H3 (ndarray): Matrix of Nneg values.
        H4 (ndarray, optional): Matrix of CPD status (True if CPD).
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Select indices for tick labeling
    x_idx = np.linspace(0, len(gammas) - 1, min(len(gammas), 5), dtype=int)
    y_idx = np.linspace(0, len(coef0s) - 1, min(len(coef0s), 5), dtype=int)

    # Heatmap 1: E_minus
    im1 = axes[0].imshow(H1, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_title("Heatmap 1: $E_{-}$ (negative mass / total mass)", fontsize=11, fontweight='bold')
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels([f"{gammas[i]:.2g}" for i in x_idx])
    axes[0].set_yticks(y_idx)
    axes[0].set_yticklabels([f"{coef0s[i]:.2g}" for i in y_idx])
    axes[0].set_xlabel(r"$\gamma$")
    axes[0].set_ylabel(r"$\mathrm{coef0}$")
    fig.colorbar(im1, ax=axes[0])

    # Heatmap 2: f_neg
    im2 = axes[1].imshow(H2, origin="lower", aspect="auto", cmap="magma")
    axes[1].set_title("Heatmap 2: $f_{neg}$ (fraction of eigenvalues < 0)", fontsize=11, fontweight='bold')
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels([f"{gammas[i]:.2g}" for i in x_idx])
    axes[1].set_yticks(y_idx)
    axes[1].set_yticklabels([f"{coef0s[i]:.2g}" for i in y_idx])
    axes[1].set_xlabel(r"$\gamma$")
    axes[1].set_ylabel(r"$\mathrm{coef0}$")
    fig.colorbar(im2, ax=axes[1])

    # Heatmap 3: Nneg
    ax3 = axes[2]
    ax3.set_facecolor("lightgreen")
    Nneg = H3.astype(float).copy()
    Nneg[Nneg == 0] = np.nan
    cmap = mpl.colormaps.get_cmap("Reds").copy()
    cmap.set_bad(color="lightblue")
    im3 = ax3.imshow(Nneg, origin="lower", aspect="auto", cmap=cmap)
    ax3.set_title("Heatmap 3: $N_{neg}$ (number of negative eigenvalues)", fontsize=11, fontweight='bold')
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels([f"{gammas[i]:.2g}" for i in x_idx])
    ax3.set_yticks(y_idx)
    ax3.set_yticklabels([f"{coef0s[i]:.2g}" for i in y_idx])
    ax3.set_xlabel(r"$\gamma$")
    ax3.set_ylabel(r"$\mathrm{coef0}$")
    cbar = fig.colorbar(im3, ax=ax3)
    cbar.set_label("Number of negative eigenvalues (Nneg)")

    # Mark CPD kernels with stars on all heatmaps
    if H4 is not None:
        cpd_indices = np.argwhere(H4)
        for ax in axes:
            if len(cpd_indices) > 0:
                ax.scatter(cpd_indices[:, 1], cpd_indices[:, 0], 
                          marker='*', s=150, c='gold', edgecolors='black', 
                          linewidth=0.5, label='CPD', zorder=5)
                ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("img/heatmaps.png")
    plt.show()

def plot_sigmoid_kernel_distribution(X_train_std, sigmoid_kernel, gamma, coef0):
    """
    Plot the histogram of sigmoid kernel values for given gamma and coef0.

    Parameters:
        X_train_std (ndarray): Standardized training data.
        sigmoid_kernel (callable): Kernel function.
        gamma (float): Kernel gamma parameter.
        coef0 (float): Kernel coef0 parameter.
    """
    K = sigmoid_kernel(X_train_std, gamma=gamma, coef0=coef0)
    plt.figure(figsize=(8,6))
    plt.hist(K.flatten(), bins=100, color='skyblue', edgecolor='black')
    plt.title('Histogram of Sigmoid Kernel Values (gamma=0.1, coef0=-5)')
    plt.xlabel('Kernel Value')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.savefig("img/sigmoid_distribution.png")
    plt.show()

def plot_rbf_kernel_distribution(X_train_std, gamma):
    """
    Plot the histogram of linear kernel values scaled by gamma.

    Parameters:
        X_train_std (ndarray): Standardized training data.
        gamma (float): Kernel gamma parameter.
    """
    K_linear = gamma * np.dot(X_train_std, X_train_std.T)
    plt.figure(figsize=(8,6))
    plt.hist(K_linear.flatten(), bins=200, color='lightcoral')
    plt.title('Histogram of Linear Kernel Values')
    plt.xlabel('Kernel Value')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.savefig("img/kernel_distribution.png")
    plt.show()

def plot_heatmaps_with_points(gammas, coef0s, H1, H2, H3, points1=None, points2=None, points3=None):
    """
    Plot three heatmaps and highlight selected points for three conditions.
    Each set of points is plotted on all three heatmaps.

    Parameters:
        gammas (array-like): List of gamma values.
        coef0s (array-like): List of coef0 values.
        H1, H2, H3 (ndarray): Heatmap matrices.
        points1, points2, points3 (list): Lists of (gamma, coef0) tuples for each condition.
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    x_idx = np.linspace(0, len(gammas) - 1, min(len(gammas), 5), dtype=int)
    y_idx = np.linspace(0, len(coef0s) - 1, min(len(coef0s), 5), dtype=int)

    # Heatmap 1: E_minus
    im1 = axes[0].imshow(H1, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_title("Heatmap 1: $E_{-}$ (negative mass / total mass)", fontsize=11, fontweight='bold')
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels([f"{gammas[i]:.2g}" for i in x_idx])
    axes[0].set_yticks(y_idx)
    axes[0].set_yticklabels([f"{coef0s[i]:.2g}" for i in y_idx])
    axes[0].set_xlabel(r"$\gamma$")
    axes[0].set_ylabel(r"$\mathrm{coef0}$")
    fig.colorbar(im1, ax=axes[0])

    # Heatmap 2: f_neg
    im2 = axes[1].imshow(H2, origin="lower", aspect="auto", cmap="magma")
    axes[1].set_title("Heatmap 2: $f_{neg}$ (fraction of eigenvalues < 0)", fontsize=11, fontweight='bold')
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels([f"{gammas[i]:.2g}" for i in x_idx])
    axes[1].set_yticks(y_idx)
    axes[1].set_yticklabels([f"{coef0s[i]:.2g}" for i in y_idx])
    axes[1].set_xlabel(r"$\gamma$")
    axes[1].set_ylabel(r"$\mathrm{coef0}$")
    fig.colorbar(im2, ax=axes[1])

    # Heatmap 3: Nneg
    ax3 = axes[2]
    ax3.set_facecolor("lightgreen")
    Nneg = H3.astype(float).copy()
    Nneg[Nneg == 0] = np.nan
    cmap = mpl.colormaps.get_cmap("Reds").copy()
    cmap.set_bad(color="lightblue")
    im3 = ax3.imshow(Nneg, origin="lower", aspect="auto", cmap=cmap)
    ax3.set_title("Heatmap 3: $N_{neg}$ (number of negative eigenvalues)", fontsize=11, fontweight='bold')
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels([f"{gammas[i]:.2g}" for i in x_idx])
    ax3.set_yticks(y_idx)
    ax3.set_yticklabels([f"{coef0s[i]:.2g}" for i in y_idx])
    ax3.set_xlabel(r"$\gamma$")
    ax3.set_ylabel(r"$\mathrm{coef0}$")
    cbar = fig.colorbar(im3, ax=ax3)
    cbar.set_label("Number of negative eigenvalues (Nneg)")

    def scatter_points(ax, points, color, label):
        """
        Helper function to plot selected points on a heatmap.

        Parameters:
            ax (matplotlib.axes): Axis to plot on.
            points (list): List of (gamma, coef0) tuples.
            color (str): Color for the points.
            label (str): Legend label.
        """
        if points:
            gammas_list = list(gammas)
            coef0s_list = list(coef0s)
            xs = []
            ys = []
            for g, c in points:
                try:
                    x_idx_val = gammas_list.index(g)
                    y_idx_val = coef0s_list.index(c)
                    xs.append(x_idx_val)
                    ys.append(y_idx_val)
                except ValueError:
                    pass
            if xs and ys:
                ax.scatter(xs, ys, color=color, s=80, edgecolor='black', label=label, marker='o', alpha=0.8)

    # Plot all sets on all axes
    for ax in axes:
        scatter_points(ax, points1, 'red', 'Cond 1')
        scatter_points(ax, points2, 'blue', 'Cond 2')
        scatter_points(ax, points3, 'green', 'N_neg == 0')

    for ax in axes:
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("img/heatmaps_with_points.png")
    plt.show()

def plot_boxplots(te_rbf_acc, te_rbf_prec, te_rbf_rec, te_rbf_f1,
                  te_acc_1, te_acc_2 ,
                  te_prec_1, te_prec_2,
                  te_rec_1, te_rec_2,
                  te_f1_1, te_f1_2):
    """
    Plot four side-by-side boxplots for accuracy, precision, recall, and F1 score.
    Each plot compares RBF, Condition 1, Condition 2, and N_neg == 0.

    Parameters:
        te_rbf_acc, te_rbf_prec, te_rbf_rec, te_rbf_f1 (list): RBF metric values.
        te_acc_1, te_acc_2 (list): Accuracy for each condition.
        te_prec_1, te_prec_2 (list): Precision for each condition.
        te_rec_1, te_rec_2 (list): Recall for each condition.
        te_f1_1, te_f1_2 (list): F1 score for each condition.
    """
    plt.figure(figsize=(20, 5))

    # Accuracy
    plt.subplot(1, 4, 1)
    plt.boxplot([te_rbf_acc, te_acc_1, te_acc_2], tick_labels=['RBF', 'Cond 1', 'Cond 2'])
    plt.title('Test Accuracy')

    # Precision
    plt.subplot(1, 4, 2)
    plt.boxplot([te_rbf_prec, te_prec_1, te_prec_2], tick_labels=['RBF', 'Cond 1', 'Cond 2'])
    plt.title('Test Precision')

    # Recall
    plt.subplot(1, 4, 3)
    plt.boxplot([te_rbf_rec, te_rec_1, te_rec_2], tick_labels=['RBF', 'Cond 1', 'Cond 2'])
    plt.title('Test Recall')

    # F1 Score
    plt.subplot(1, 4, 4)
    plt.boxplot([te_rbf_f1, te_f1_1, te_f1_2], tick_labels=['RBF', 'Cond 1', 'Cond 2'])
    plt.title('Test F1 Score')

    plt.tight_layout()
    plt.savefig("img/SVM_boxplots.png")
    plt.show()

def delta_f1_hist(delta_shift, delta_clip, delta_clipnorm, suffix=''):
    """
    Plot histograms of ΔF1 for each correction method (Shift, Clip, clipnorm).

    Parameters:
        delta_shift (ndarray): ΔF1 for Shift correction.
        delta_clip (ndarray): ΔF1 for Clip correction.
        delta_clipnorm (ndarray): ΔF1 for clipnorm correction.
        suffix (str): Suffix to append to the filename (e.g., '_norm').
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Shift
    axes[0].hist(delta_shift, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='No change')
    axes[0].axvline(x=delta_shift.mean(), color='green', linestyle='-', linewidth=2, 
                    label=f'Mean Δ={delta_shift.mean():.4f}')
    axes[0].set_title('Shift correction', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('delta_F1 = percentage of corrected errors (Shift - Original)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Clip
    axes[1].hist(delta_clip, bins=40, color='coral', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='No change')
    axes[1].axvline(x=delta_clip.mean(), color='green', linestyle='-', linewidth=2, 
                    label=f'Mean Δ={delta_clip.mean():.4f}')
    axes[1].set_title('Clip correction', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('delta_F1 = percentage of corrected errors (Clip - Original)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    # clipnorm
    axes[2].hist(delta_clipnorm, bins=40, color='mediumpurple', alpha=0.7, edgecolor='black')
    axes[2].axvline(x=0, color='r', linestyle='--', linewidth=2, label='No change')
    axes[2].axvline(x=delta_clipnorm.mean(), color='green', linestyle='-', linewidth=2, 
                    label=f'Mean Δ={delta_clipnorm.mean():.4f}')
    axes[2].set_title('Clipnorm correction', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('delta_F1 = percentage of corrected errors (clipnorm - Original)', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'img/delta_f1_corrections_histogram{suffix}.png', dpi=150)
    plt.show()

def correction_plots_eig(methods, n_negative_eigenvalues, negative_mass_fraction, is_cpd_arr=None, suffix=''):
    """
    Plot ΔF1 vs. kernel properties (number of negative eigenvalues and negative mass fraction)
    for each correction method, with shared axes and Spearman correlation.
    CPD kernels are marked with stars, non-CPD with circles.

    Parameters:
        methods (list): List of tuples (method_name, delta, color).
        n_negative_eigenvalues (ndarray): Number of negative eigenvalues per parameter combination.
        negative_mass_fraction (ndarray): Negative mass fraction per parameter combination.
        is_cpd_arr (ndarray, optional): Boolean array indicating CPD status for each combination.
        suffix (str): Suffix to append to the filename (e.g., '_norm').
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Build masks per method once
    non_zero_masks = [np.abs(delta) > 1e-10 for (_, delta, _) in methods]

    # Collect x/y for shared limits (Row 1)
    x_vals_row1 = [n_negative_eigenvalues[mask] for mask in non_zero_masks]
    y_vals_row1 = [methods[i][1][non_zero_masks[i]] for i in range(len(methods))]

    x_row1_all = np.concatenate(x_vals_row1) if x_vals_row1 else np.array([0])
    y_row1_all = np.concatenate(y_vals_row1) if y_vals_row1 else np.array([0])

    x_min_row1, x_max_row1 = x_row1_all.min(), x_row1_all.max()
    y_min_row1, y_max_row1 = y_row1_all.min(), y_row1_all.max()

    # Row 1: Delta F1 vs Number of Negative Eigenvalues (shared axes)
    for i, (method_name, delta, color) in enumerate(methods):
        ax = axes[0, i]
        non_zero_mask = non_zero_masks[i]
        n_neg_filtered = n_negative_eigenvalues[non_zero_mask]
        delta_filtered = delta[non_zero_mask]

        # Plot with different markers for CPD vs non-CPD
        if is_cpd_arr is not None:
            cpd_filtered = is_cpd_arr[non_zero_mask]
            # Non-CPD points (circles)
            non_cpd_mask = ~cpd_filtered
            if np.any(non_cpd_mask):
                ax.scatter(n_neg_filtered[non_cpd_mask], delta_filtered[non_cpd_mask], 
                          alpha=0.6, c=color, edgecolors='black', linewidth=0.5, s=60, 
                          marker='o', label='Non-CPD')
            # CPD points (stars)
            cpd_mask = cpd_filtered
            if np.any(cpd_mask):
                ax.scatter(n_neg_filtered[cpd_mask], delta_filtered[cpd_mask], 
                          alpha=0.8, c=color, edgecolors='black', linewidth=0.5, s=120, 
                          marker='*', label='CPD')
        else:
            ax.scatter(n_neg_filtered, delta_filtered, alpha=0.6, c=color,
                      edgecolors='black', linewidth=0.5, s=60)
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No change')

        if len(delta_filtered) > 1 and len(np.unique(n_neg_filtered)) > 1:
            corr, _ = spearmanr(n_neg_filtered, delta_filtered)
        else:
            corr = np.nan

        ax.set_xlabel('Faction of Negative Eigenvalues', fontsize=11)
        ax.set_ylabel(f'delta_F1 = % corrected errors ({method_name})', fontsize=11)
        ax.set_title(f'{method_name} Correction\nSpearman: {corr:.3f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax.set_xlim(x_min_row1, x_max_row1)
        ax.set_ylim(y_min_row1, y_max_row1)

    # Precompute common axis limits for Row 2 (x: Negative Mass Fraction, y: ΔF1)
    x_vals_row2 = [negative_mass_fraction[mask] for mask in non_zero_masks]
    x_row2_all = np.concatenate(x_vals_row2) if x_vals_row2 else np.array([0.0])

    x_min_row2, x_max_row2 = x_row2_all.min(), x_row2_all.max()
    y_min_row2, y_max_row2 = y_row1_all.min(), y_row1_all.max()

    # Row 2: Delta F1 vs Negative Mass Fraction (shared axes)
    for i, (method_name, delta, color) in enumerate(methods):
        ax = axes[1, i]
        non_zero_mask = non_zero_masks[i]
        neg_frac_filtered = negative_mass_fraction[non_zero_mask]
        delta_filtered = delta[non_zero_mask]

        # Plot with different markers for CPD vs non-CPD
        if is_cpd_arr is not None:
            cpd_filtered = is_cpd_arr[non_zero_mask]
            # Non-CPD points (circles)
            non_cpd_mask = ~cpd_filtered
            if np.any(non_cpd_mask):
                ax.scatter(neg_frac_filtered[non_cpd_mask], delta_filtered[non_cpd_mask], 
                          alpha=0.6, c=color, edgecolors='black', linewidth=0.5, s=60, 
                          marker='o', label='Non-CPD')
            # CPD points (stars)
            cpd_mask = cpd_filtered
            if np.any(cpd_mask):
                ax.scatter(neg_frac_filtered[cpd_mask], delta_filtered[cpd_mask], 
                          alpha=0.8, c=color, edgecolors='black', linewidth=0.5, s=120, 
                          marker='*', label='CPD')
        else:
            ax.scatter(neg_frac_filtered, delta_filtered, alpha=0.6, c=color,
                      edgecolors='black', linewidth=0.5, s=60)
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No change')

        if len(delta_filtered) > 1 and len(np.unique(neg_frac_filtered)) > 1:
            corr, _ = spearmanr(neg_frac_filtered, delta_filtered)
        else:
            corr = np.nan

        ax.set_xlabel('Negative Mass Fraction', fontsize=11)
        ax.set_ylabel(f'delta_F1 = % corrected errors ({method_name})', fontsize=11)
        ax.set_title(f'{method_name} Correction\nSpearman: {corr:.3f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax.set_xlim(x_min_row2, x_max_row2)
        ax.set_ylim(y_min_row2, y_max_row2)
    plt.tight_layout()
    plt.savefig(f'img/delta_f1_vs_kernel_properties{suffix}.png', dpi=150, bbox_inches='tight')
    plt.show()

def correction_plots_imp(methods, frac_negative_eigenvalues, negative_mass_fraction, is_cpd_arr=None, suffix=''):
    """
    Create scatter plots of kernel properties (fraction of negative eigenvalues vs negative mass fraction)
    colored by improvement/degradation. All points have equal size.
    CPD kernels are marked with stars, non-CPD with circles.

    Parameters:
        methods (list): List of tuples (method_name, delta, color).
        frac_negative_eigenvalues (ndarray): Fraction of negative eigenvalues per parameter combination.
        negative_mass_fraction (ndarray): Negative mass fraction per parameter combination.
        is_cpd_arr (ndarray, optional): Boolean array indicating CPD status for each combination.
        suffix (str): Suffix to append to the filename (e.g., '_norm').
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Filter out zeros per method
    non_zero_masks = [np.abs(m[1]) > 1e-10 for m in methods]

    # Collect global x/y for shared axes
    frac_neg_all = np.concatenate([frac_negative_eigenvalues[m] for m in non_zero_masks]) if len(non_zero_masks) else np.array([0])
    neg_frac_all = np.concatenate([negative_mass_fraction[m] for m in non_zero_masks]) if len(non_zero_masks) else np.array([0.0])

    # Shared axis limits with padding
    x_min, x_max = frac_neg_all.min(), frac_neg_all.max()
    y_min, y_max = neg_frac_all.min(), neg_frac_all.max()
    x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 0.05
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.05
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    # Fixed point sizes
    point_size = 80
    cpd_point_size = 150

    for i, (method_name, delta, color) in enumerate(methods):
        ax = axes[i]

        # Filter non-zero deltas
        non_zero_mask = np.abs(delta) > 1e-10
        frac_neg_plot = frac_negative_eigenvalues[non_zero_mask]
        neg_frac_plot = negative_mass_fraction[non_zero_mask]
        delta_plot = delta[non_zero_mask]
        
        # Get CPD status if available
        if is_cpd_arr is not None:
            cpd_plot = is_cpd_arr[non_zero_mask]
        else:
            cpd_plot = np.zeros(len(delta_plot), dtype=bool)

        # Separate improved vs degraded
        improved_mask = delta_plot > 0
        degraded_mask = delta_plot < 0

        colors = ['#FFFFFF', '#00D98C']  # bianco -> verde smeraldo
        custom_cmap_green = LinearSegmentedColormap.from_list('white_emerald', colors, N=256)

        colors = ['#FFFFFF', '#FF0000']  # bianco -> rosso acceso
        custom_cmap_red = LinearSegmentedColormap.from_list('white_red', colors, N=256)



        # Plot improvements (green shades) - Non-CPD (circles)
        improved_non_cpd = improved_mask & ~cpd_plot
        if np.sum(improved_non_cpd) > 0:
            ax.scatter(
                frac_neg_plot[improved_non_cpd],
                neg_frac_plot[improved_non_cpd],
                s=point_size,
                c=delta_plot[improved_non_cpd],
                cmap=custom_cmap_green,
                alpha=0.6,
                edgecolors='darkgreen',
                linewidth=1,
                marker='o',
                vmin=0,
                vmax=delta_plot[improved_mask].max() if np.any(improved_mask) else 1,
                label='Improved (Non-CPD)'
            )
        
        # Plot improvements (green shades) - CPD (stars)
        improved_cpd = improved_mask & cpd_plot
        if np.sum(improved_cpd) > 0:
            ax.scatter(
                frac_neg_plot[improved_cpd],
                neg_frac_plot[improved_cpd],
                s=cpd_point_size,
                c=delta_plot[improved_cpd],
                cmap=custom_cmap_green,
                alpha=0.8,
                edgecolors='darkgreen',
                linewidth=1.5,
                marker='*',
                vmin=0,
                vmax=delta_plot[improved_mask].max() if np.any(improved_mask) else 1,
                label='Improved (CPD)'
            )

        # Plot degradations (red shades) - Non-CPD (circles)
        degraded_non_cpd = degraded_mask & ~cpd_plot
        if np.sum(degraded_non_cpd) > 0:
            ax.scatter(
                frac_neg_plot[degraded_non_cpd],
                neg_frac_plot[degraded_non_cpd],
                s=point_size,
                c=-delta_plot[degraded_non_cpd],
                cmap=custom_cmap_red,
                alpha=0.6,
                edgecolors='darkred',
                linewidth=1,
                marker='o',
                vmin=0,
                vmax=-delta_plot[degraded_mask].min() if np.any(degraded_mask) else 1,
                label='Degraded (Non-CPD)'
            )
        
        # Plot degradations (red shades) - CPD (stars)
        degraded_cpd = degraded_mask & cpd_plot
        if np.sum(degraded_cpd) > 0:
            ax.scatter(
                frac_neg_plot[degraded_cpd],
                neg_frac_plot[degraded_cpd],
                s=cpd_point_size,
                c=-delta_plot[degraded_cpd],
                cmap=custom_cmap_red,
                alpha=0.8,
                edgecolors='darkred',
                linewidth=1.5,
                marker='*',
                vmin=0,
                vmax=-delta_plot[degraded_mask].min() if np.any(degraded_mask) else 1,
                label='Degraded (CPD)'
            )

        ax.set_xlabel('Fraction of Negative Eigenvalues', fontsize=12, fontweight='bold')
        ax.set_ylabel('Negative Mass Fraction', fontsize=12, fontweight='bold')
        ax.set_title(f'{method_name} Correction (n={len(delta_plot)})',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best')

        # Shared axes across the 3 plots
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f'img/delta_f1_scatter_sized{suffix}.png', dpi=150, bbox_inches='tight')
    plt.show()

def correction_plots_coefs(results, methods, is_cpd_arr=None, suffix=''):
    """
    Scatter plots of gamma vs coef0 for each correction method.
    CPD kernels are marked with stars, non-CPD with circles.

    Parameters:
        results (dict): Results dictionary containing parameter combinations.
        methods (list): List of tuples (method_name, delta, color/cmap).
        is_cpd_arr (ndarray, optional): Boolean array indicating CPD status for each combination.
        suffix (str): Suffix to append to the filename (e.g., '_norm').
    """
    gammas_arr = np.array([gc[0] for gc in results['param_combinations']])
    coef0s_arr = np.array([gc[1] for gc in results['param_combinations']])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Shared limits for gamma and coef0 across all three plots with padding
    x_min, x_max = gammas_arr.min(), gammas_arr.max()
    y_min, y_max = coef0s_arr.min(), coef0s_arr.max()
    x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 0.05
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.05
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    # Fixed point sizes
    point_size = 80
    cpd_point_size = 150

    # Custom colormaps (same as correction_plots_imp)
    colors_green = ['#FFFFFF', '#00D98C']  # white -> emerald green
    custom_cmap_green = LinearSegmentedColormap.from_list('white_emerald', colors_green, N=256)
    colors_red = ['#FFFFFF', '#FF0000']  # white -> red
    custom_cmap_red = LinearSegmentedColormap.from_list('white_red', colors_red, N=256)

    for i, (method_name, delta, color_or_cmap) in enumerate(methods):
        ax = axes[i]
        non_zero_mask = np.isfinite(delta)
        x = gammas_arr[non_zero_mask]
        y = coef0s_arr[non_zero_mask]
        d = delta[non_zero_mask]

        improved = d > 0
        degraded = d < 0
        
        # Get CPD status if available
        if is_cpd_arr is not None:
            cpd_mask = is_cpd_arr[non_zero_mask]
        else:
            cpd_mask = np.zeros(len(d), dtype=bool)

        # Improved points - Non-CPD (circles)
        improved_non_cpd = improved & ~cpd_mask
        if np.any(improved_non_cpd):
            vmax_imp = float(np.nanmax(d[improved])) if np.any(improved) else 1.0
            ax.scatter(
                x[improved_non_cpd], y[improved_non_cpd], s=point_size,
                c=d[improved_non_cpd], cmap=custom_cmap_green, vmin=0, vmax=vmax_imp,
                alpha=0.6, edgecolors='darkgreen', linewidth=1, marker='o', label='Improved (Non-CPD)'
            )
        
        # Improved points - CPD (stars)
        improved_cpd = improved & cpd_mask
        if np.any(improved_cpd):
            vmax_imp = float(np.nanmax(d[improved])) if np.any(improved) else 1.0
            ax.scatter(
                x[improved_cpd], y[improved_cpd], s=cpd_point_size,
                c=d[improved_cpd], cmap=custom_cmap_green, vmin=0, vmax=vmax_imp,
                alpha=0.8, edgecolors='darkgreen', linewidth=1.5, marker='*', label='Improved (CPD)'
            )

        # Degraded points - Non-CPD (circles)
        degraded_non_cpd = degraded & ~cpd_mask
        if np.any(degraded_non_cpd):
            vmax_deg = float(-np.nanmin(d[degraded])) if np.any(degraded) else 1.0
            ax.scatter(
                x[degraded_non_cpd], y[degraded_non_cpd], s=point_size,
                c=-d[degraded_non_cpd], cmap=custom_cmap_red, vmin=0, vmax=vmax_deg,
                alpha=0.6, edgecolors='darkred', linewidth=1, marker='o', label='Degraded (Non-CPD)'
            )
        
        # Degraded points - CPD (stars)
        degraded_cpd = degraded & cpd_mask
        if np.any(degraded_cpd):
            vmax_deg = float(-np.nanmin(d[degraded])) if np.any(degraded) else 1.0
            ax.scatter(
                x[degraded_cpd], y[degraded_cpd], s=cpd_point_size,
                c=-d[degraded_cpd], cmap=custom_cmap_red, vmin=0, vmax=vmax_deg,
                alpha=0.8, edgecolors='darkred', linewidth=1.5, marker='*', label='Degraded (CPD)'
            )

        ax.set_title(f'{method_name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('gamma', fontsize=12)
        ax.set_ylabel('coef0', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)

        # Shared axes
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f'img/scatter_gamma_coef0_size_delta{suffix}.png', dpi=150, bbox_inches='tight')
    plt.show()

def correction_plots_baseline(results, delta_shift, delta_clip, delta_clipnorm, is_cpd_arr=None, suffix=''):
    """
    Plot Baseline F1 (original) vs ΔF1 for each correction method, with fixed marker size and robust y-limits.
    CPD kernels are marked with stars, non-CPD with circles.
    Row 1: Full view, Row 2: Zoomed view (-100 to 100).

    Parameters:
        results (dict): Results dictionary containing parameter combinations and F1 scores.
        delta_shift, delta_clip, delta_clipnorm (ndarray): ΔF1 arrays for each method.
        is_cpd_arr (ndarray, optional): Boolean array indicating CPD status for each combination.
        suffix (str): Suffix to append to the filename (e.g., '_norm').
    """
    baseline_f1 = results['original']['f1']

    # Shared limits (with padding) and symmetric y
    x_min_raw, x_max_raw = baseline_f1.min(), baseline_f1.max()
    x_pad = 0.05 * (x_max_raw - x_min_raw) if x_max_raw > x_min_raw else 0.05
    x_min, x_max = x_min_raw - x_pad, x_max_raw + x_pad

    # Robust y-limits: symmetric, based on max |ΔF1|, with small padding
    y_abs_max = max(np.abs(delta_shift).max(),
                    np.abs(delta_clip).max(),
                    np.abs(delta_clipnorm).max())
    y_pad = 0.05 * y_abs_max
    y_min, y_max = -(y_abs_max + y_pad), (y_abs_max/30 + y_pad)

    # Zoomed y-limits with padding
    y_min_zoom, y_max_zoom = -100 - 5, 100 + 5

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    methods_data = [
        ('Shift',  delta_shift,  'steelblue'),
        ('Clip',   delta_clip,   'coral'),
        ('clipnorm', delta_clipnorm, 'mediumpurple')
    ]

    for row in range(2):
        for i, (method_name, delta, color) in enumerate(methods_data):
            ax = axes[row, i]
            mask = np.isfinite(delta) & (np.abs(delta) > 1e-10)
            x = baseline_f1[mask]
            y = delta[mask]
            
            # Get CPD status if available
            if is_cpd_arr is not None:
                cpd_mask = is_cpd_arr[mask]
            else:
                cpd_mask = np.zeros(len(y), dtype=bool)

            # Plot Non-CPD points (circles)
            non_cpd_mask = ~cpd_mask
            if np.any(non_cpd_mask):
                ax.scatter(x[non_cpd_mask], y[non_cpd_mask], s=80, c=color, alpha=0.7, 
                          edgecolors='black', linewidth=0.5, marker='o', label='Non-CPD')
            
            # Plot CPD points (stars)
            if np.any(cpd_mask):
                ax.scatter(x[cpd_mask], y[cpd_mask], s=150, c=color, alpha=0.9, 
                          edgecolors='black', linewidth=1, marker='*', label='CPD')
            
            ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='delta_F1 = 0')

            if row == 0:
                ax.set_title(f'{method_name} (Full View)', fontsize=13, fontweight='bold')
                ax.set_ylim(y_min, y_max)
            else:
                ax.set_title(f'{method_name} (Zoomed: -100 to 100)', fontsize=13, fontweight='bold')
                ax.set_ylim(y_min_zoom, y_max_zoom)

            ax.set_xlabel('Baseline F1 (Original)', fontsize=12)
            ax.set_ylabel('delta_F1 = % corrected errors', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=10)
            ax.set_xlim(x_min, x_max)

    plt.tight_layout()
    plt.savefig(f'img/baseline_vs_delta_3methods{suffix}.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function placeholder for consistency.
    """
    pass

if __name__ == "__main__":
    main()