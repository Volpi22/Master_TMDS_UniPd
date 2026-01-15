import numpy as np
from statsmodels.stats.weightstats import ttost_paired
from statsmodels.stats.multitest import multipletests
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def equivalence_test_rff_kde_simple(df_results, D_values, equivalence_margin=0.02, alpha=0.05):
    """
    Perform a basic Two One-Sided Test (TOST) for paired equivalence.
    
    Standard t-tests check if means are different. TOST checks if the difference 
    between means falls entirely within a specific interval (-margin, +margin).
    
    

    Hypotheses:
    -----------
    H0: |AUC_exact - AUC_rff| >= margin (The models are different)
    H1: |AUC_exact - AUC_rff| < margin  (The models are practically equivalent)
    
    Parameters
    ----------
    equivalence_margin : float
        The practical threshold for "sameness" (default 0.02 AUC).
    """
    
    results_dict = {}
    
    for difficulty in df_results['Difficulty'].unique():
        print(f"\n{'='*70}")
        print(f"DIFFICULTY: {difficulty}")
        print(f"{'='*70}")
        
        df_diff = df_results[df_results['Difficulty'] == difficulty]
        exact_aucs = df_diff[df_diff['Method'] == 'Exact_KDE']['AUC'].values
        
        test_results = []
        
        for D in D_values:
            rff_aucs = df_diff[df_diff['D'] == D]['AUC'].values
            
            # statsmodels TOST returns (p_value, (t_lower, p_lower), (t_upper, p_upper))
            pval_tost, _, _ = ttost_paired(
                exact_aucs, 
                rff_aucs, 
                low=-equivalence_margin, 
                upp=equivalence_margin
            )
            
            # Unpack tuple if necessary (version dependent)
            if isinstance(pval_tost, tuple):
                pval_tost = pval_tost[0]
            
            differences = exact_aucs - rff_aucs
            mean_diff = np.mean(differences)
            
            test_results.append({
                'D': D,
                'mean_diff': mean_diff,
                'p_tost': pval_tost
            })
        
        # Apply Holm-Bonferroni correction to control Family-Wise Error Rate
        # across multiple D comparisons.
        pvalues = [r['p_tost'] for r in test_results]
        reject, pvals_corr, _, _ = multipletests(pvalues, alpha=alpha, method='holm')
        
        for i, r in enumerate(test_results):
            r['corrected_p_tost'] = pvals_corr[i]
            r['equivalent'] = reject[i]
        
        min_D = next((r['D'] for r in test_results if r['equivalent']), None)
        results_dict[difficulty] = {'results': test_results, 'min_D': min_D}
        
        # Display
        print(f"\nMargin: ±{equivalence_margin:.3f}")
        print(f"{'D':<8} {'Mean Δ':<10} {'p(equiv)':<12} {'Adj.p':<12} {'Decision'}")
        print("-"*70)
        for r in test_results:
            decision = "EQUIVALENT" if r['equivalent'] else "NOT equiv"
            print(f"{r['D']:<8} {r['mean_diff']:<10.5f} {r['p_tost']:<12.6f} "
                  f"{r['corrected_p_tost']:<12.6f} {decision}")
        
        if min_D:
            print(f"\n✓ Minimum D: {min_D}")
    
    return results_dict

# ============================================================================
# COMPLETE TOST EQUIVALENCE TEST WITH BOOTSTRAP SUPPORT
# Handles both normal and non-normal data automatically
# ============================================================================

def bootstrap_tost_paired(exact_vals, rff_vals, equivalence_margin=0.02, 
                          n_bootstrap=10000, random_state=42):
    """
    Perform TOST using Bootstrap resampling (Distribution-Free).
    
    Useful when the differences between models are not normally distributed,
    which violates the assumption of the standard T-test.
    
    

    Returns
    -------
    p_tost : float
        The maximum of the lower and upper one-sided p-values.
    ci_90_lower, ci_90_upper : float
        The 90% confidence interval (corresponding to alpha=0.05 one-sided).
    """
    np.random.seed(random_state)

    n = len(exact_vals)
    differences = exact_vals - rff_vals
    mean_diff_obs = np.mean(differences)

    # Resample indices with replacement to build empirical distribution of the mean
    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        boot_diff = differences[indices]
        bootstrap_means[i] = np.mean(boot_diff)

    # Calculate 90% bootstrap confidence interval (1 - 2*alpha)
    ci_90_lower = np.percentile(bootstrap_means, 5)
    ci_90_upper = np.percentile(bootstrap_means, 95)

    # Bootstrap TOST p-values calculation
    # 1. Lower test: H0: mean <= -margin (The exact model is much worse)
    p_lower = np.mean(bootstrap_means <= -equivalence_margin)

    # 2. Upper test: H0: mean >= margin (The exact model is much better)
    p_upper = np.mean(bootstrap_means >= equivalence_margin)

    # TOST p-value is the maximum of the two (Intersection-Union test)
    p_tost = max(p_lower, p_upper)

    return p_tost, p_lower, p_upper, ci_90_lower, ci_90_upper


def equivalence_test_rff_kde(df_results, D_values, equivalence_margin=0.02, 
                             alpha=0.05, method='auto', check_assumptions=True, 
                             save_diagnostics=True, n_bootstrap=10000, random_state=42):
    """
    Comprehensive Equivalence Testing Suite.
    
    Automatically selects between Parametric T-test and Non-Parametric Bootstrap
    based on the normality of residuals (Shapiro-Wilk test).

    Parameters
    ----------
    method : str
        'auto': Automatic selection based on normality check.
        'parametric': Force T-test (assumes normality).
        'bootstrap': Force Bootstrap (robust to non-normality).
    """

    results_dict = {}

    # Input Validation
    required_cols = ['Difficulty', 'Method', 'D', 'AUC']
    for col in required_cols:
        assert col in df_results.columns, f"Missing '{col}' column"

    assert equivalence_margin > 0, "Equivalence margin must be positive"
    assert 0 < alpha < 1, "Alpha must be between 0 and 1"

    print("\n" + "="*80)
    print(" TOST EQUIVALENCE TESTING: RFF vs EXACT KDE")
    print("="*80)
    print(f"Equivalence margin: ±{equivalence_margin:.4f} AUC")
    print(f"Method selection: {method}")
    print("="*80)

    for difficulty in sorted(df_results['Difficulty'].unique()):
        print(f"\n{'='*80}")
        print(f"DIFFICULTY: {difficulty.upper()}")
        print(f"{'='*80}")

        df_diff = df_results[df_results['Difficulty'] == difficulty]
        exact_aucs = df_diff[df_diff['Method'] == 'Exact_KDE']['AUC'].values
        n_seeds = len(exact_aucs)

        if n_seeds < 10:
            print(f"⚠️  WARNING: Only {n_seeds} seeds - low statistical power")

        test_results = []
        normality_violations = []
        methods_used = []

        for D in D_values:
            rff_aucs = df_diff[df_diff['D'] == D]['AUC'].values

            if len(rff_aucs) != n_seeds:
                raise ValueError(f"Mismatched lengths for D={D}")

            # Calculate paired differences
            differences = exact_aucs - rff_aucs
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            se_diff = std_diff / np.sqrt(n_seeds)

            # --- Assumption Check: Normality ---
            if check_assumptions and n_seeds >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(differences)
                is_normal = shapiro_p > 0.05

                if not is_normal:
                    normality_violations.append(D)
                    print(f"  ⚠️  D={D}: Non-normal differences (Shapiro p={shapiro_p:.4f})")
            else:
                shapiro_p = np.nan
                is_normal = True

            # --- Method Selection ---
            if method == 'auto':
                # Use parametric if normal OR n>=30 (Central Limit Theorem applies)
                if is_normal:
                    selected_method = 'parametric'
                else:
                    selected_method = 'bootstrap'
            else:
                selected_method = method

            methods_used.append(selected_method)

            # --- Execution ---
            if selected_method == 'parametric':
                # Standard Parametric TOST
                tost_result = ttost_paired(
                    exact_aucs, rff_aucs,
                    low=-equivalence_margin,
                    upp=equivalence_margin
                )
                
                # Unpack results safely
                pval_tost = tost_result[0]
                if isinstance(pval_tost, tuple): pval_tost = pval_tost[0]

                # Calculate CIs analytically
                t_crit_90 = stats.t.ppf(0.95, df=n_seeds-1)
                ci_90_lower = mean_diff - t_crit_90 * se_diff
                ci_90_upper = mean_diff + t_crit_90 * se_diff
                
                t_crit_95 = stats.t.ppf(0.975, df=n_seeds-1)
                ci_95_lower = mean_diff - t_crit_95 * se_diff
                ci_95_upper = mean_diff + t_crit_95 * se_diff

                # Fill placeholders for bootstrap-specific outputs
                pval_lower = np.nan
                pval_upper = np.nan
                test_method_label = 'Parametric'

            else:  # Bootstrap
                pval_tost, pval_lower, pval_upper, ci_90_lower, ci_90_upper = \
                    bootstrap_tost_paired(
                        exact_aucs, rff_aucs,
                        equivalence_margin=equivalence_margin,
                        n_bootstrap=n_bootstrap,
                        random_state=random_state
                    )

                # Calculate 95% CI from bootstrap for reporting consistency
                np.random.seed(random_state)
                # (Re-running simplified loop for 95% CI)
                bootstrap_means = [np.mean(differences[np.random.choice(n_seeds, size=n_seeds, replace=True)]) for _ in range(n_bootstrap)]
                ci_95_lower = np.percentile(bootstrap_means, 2.5)
                ci_95_upper = np.percentile(bootstrap_means, 97.5)

                test_method_label = 'Bootstrap'

            # Effect size (Cohen's d)
            cohens_d = mean_diff / std_diff if std_diff > 0 else 0

            test_results.append({
                'D': D,
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'se_diff': se_diff,
                'p_tost': pval_tost,
                'p_lower': pval_lower,
                'p_upper': pval_upper,
                'ci_90_lower': ci_90_lower,
                'ci_90_upper': ci_90_upper,
                'ci_95_lower': ci_95_lower,
                'ci_95_upper': ci_95_upper,
                'cohens_d': cohens_d,
                'shapiro_p': shapiro_p,
                'normality_ok': is_normal,
                'n_seeds': n_seeds,
                'test_method': test_method_label
            })

        # --- Multiple Testing Correction ---
        # Crucial to avoid Type I errors when testing many D values sequentially
        pvalues = [r['p_tost'] for r in test_results]
        reject, pvals_corr, alphacSidak, alphacBonf = multipletests(
            pvalues, alpha=alpha, method='holm'
        )

        for i, r in enumerate(test_results):
            r['corrected_p_tost'] = pvals_corr[i]
            r['equivalent'] = reject[i]

        # Determine the lowest D that satisfies equivalence
        min_D = next((r['D'] for r in test_results if r['equivalent']), None)

        results_dict[difficulty] = {
            'results': test_results,
            'min_D': min_D,
            'normality_violations': normality_violations,
            'methods_used': dict(zip(D_values, methods_used)),
            'n_comparisons': len(D_values),
            'n_seeds': n_seeds
        }

        # Print Table
        print(f"{'D':<8} {'Mean Δ':<10} {'SE':<9} {'p(equiv)':<11} {'Adj.p':<11} "
              f"{'90% CI':<24} {'Cohen d':<9} {'Method':<11} {'Decision'}")
        print("-"*110)

        for r in test_results:
            ci_str = f"[{r['ci_90_lower']:>7.5f}, {r['ci_90_upper']:>7.5f}]"
            decision = "EQUIV" if r['equivalent'] else "NOT equiv"
            normality_flag = " norm " if r['normality_ok'] else " not norm"

            print(f"{r['D']:<8} {r['mean_diff']:<10.6f} {r['se_diff']:<9.6f} "
                  f"{r['p_tost']:<11.6f} {r['corrected_p_tost']:<11.6f} "
                  f"{ci_str:<24} {r['cohens_d']:<9.3f} {r['test_method']:<11} "
                  f"{decision}{normality_flag}")

        # Summary Block
        if min_D:
            print(f"\n{'='*80}")
            print(f"✓ RECOMMENDATION: Minimum D = {min_D}")
            print(f"{'='*80}")
        else:
            print(f"\n{'='*80}")
            print(f"NO EQUIVALENCE at margin ±{equivalence_margin:.4f}")
            print(f"{'='*80}")

        if save_diagnostics and check_assumptions:
            _save_diagnostic_plots(df_diff, D_values, exact_aucs, difficulty,
                                   equivalence_margin, test_results)

    return results_dict


def _save_diagnostic_plots(df_diff, D_values, exact_aucs, difficulty, margin, test_results):
    """
    Generate diagnostic plots (Q-Q and Histograms) to visually verify assumptions.
    
    
    """
    n_plots = min(4, len(D_values))
    fig, axes = plt.subplots(2, n_plots, figsize=(5*n_plots, 10))

    if n_plots == 1:
        axes = axes.reshape(2, 1)

    for idx, D in enumerate(D_values[:n_plots]):
        rff_aucs = df_diff[df_diff['D'] == D]['AUC'].values
        differences = exact_aucs - rff_aucs
        result = next(r for r in test_results if r['D'] == D)

        # 1. Q-Q plot (Normality Check)
        stats.probplot(differences, dist="norm", plot=axes[0, idx])
        axes[0, idx].set_title(
            f'Q-Q Plot: D={D}\n'
            f'Shapiro p={result["shapiro_p"]:.4f}\n'
            f'Method: {result["test_method"]}',
            fontsize=10
        )
        axes[0, idx].grid(True, alpha=0.3)

        # 2. Histogram (Equivalence Region Visualization)
        axes[1, idx].hist(differences, bins=15, edgecolor='black', 
                         alpha=0.7, color='steelblue')
        axes[1, idx].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        axes[1, idx].axvline(-margin, color='orange', linestyle='--', 
                            linewidth=2, label=f'±{margin}')
        axes[1, idx].axvline(margin, color='orange', linestyle='--', linewidth=2)
        axes[1, idx].axvline(result['mean_diff'], color='green', linestyle='-',
                            linewidth=2, label=f'Mean={result["mean_diff"]:.4f}')

        axes[1, idx].set_xlabel('AUC Difference (Exact - RFF)', fontsize=10)
        axes[1, idx].set_title(f'Distribution: D={D}', fontsize=10)
        axes[1, idx].legend(fontsize=8)
        axes[1, idx].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filename = f'img/tost_diagnostics_{difficulty}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n Diagnostic plots saved: {filename}")


def save_results_to_csv(results_dict, filename='tost_results_complete.csv'):
    """Flatten results dictionary and save to CSV for external analysis."""
    all_results = []
    for difficulty, res_dict in results_dict.items():
        for r in res_dict['results']:
            # Create a flat dictionary for DataFrame
            flat_res = {
                'Difficulty': difficulty,
                'D': r['D'],
                'mean_diff_AUC': r['mean_diff'],
                'std_diff': r['std_diff'],
                'se_diff': r['se_diff'],
                'p_tost': r['p_tost'],
                'corrected_p_tost': r['corrected_p_tost'],
                'test_method': r['test_method'],
                'equivalent': r['equivalent'],
            }
            # Add remaining keys dynamically
            flat_res.update({k: v for k, v in r.items() if k not in flat_res})
            all_results.append(flat_res)

    df = pd.DataFrame(all_results)
    df.to_csv(filename, index=False)
    print(f"\n✓ Detailed results saved to '{filename}'")
    return df


def print_summary_table(results_dict):
    """Print a high-level executive summary of equivalence findings."""
    print("\n" + "="*80)
    print(" SUMMARY: MINIMUM D FOR EQUIVALENCE BY DIFFICULTY")
    print("="*80)

    summary_data = []
    for difficulty, res in results_dict.items():
        min_D = res['min_D']
        if min_D:
            result = next(r for r in res['results'] if r['D'] == min_D)
            summary_data.append({
                'Difficulty': difficulty,
                'Min_D': min_D,
                'Mean_Diff': f"{result['mean_diff']:.6f}",
                '90%_CI': f"[{result['ci_90_lower']:.5f}, {result['ci_90_upper']:.5f}]",
                'Adj_p': f"{result['corrected_p_tost']:.6f}",
                'Method': result['test_method']
            })
        else:
            summary_data.append({
                'Difficulty': difficulty,
                'Min_D': 'None',
                'Mean_Diff': 'N/A',
                '90%_CI': 'N/A',
                'Adj_p': 'N/A',
                'Method': 'N/A'
            })

    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))

    # Calculate final recommendation
    valid_min_Ds = [res['min_D'] for res in results_dict.values() if res['min_D']]
    if valid_min_Ds:
        conservative_D = max(valid_min_Ds)
        print(f"\n{'='*80}")
        print(f" FINAL RECOMMENDATION: Use D = {conservative_D}")
        print(" This ensures equivalence across all tested difficulty levels.")
    else:
        print(f"\n No universal D found for the given margin.")

    print("="*80)

if __name__ == "__main__":
    pass