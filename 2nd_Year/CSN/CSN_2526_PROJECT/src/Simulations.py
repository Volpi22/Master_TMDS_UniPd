import copy
import random
import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed
from IPython.display import display, Markdown

from src.Epidemic import simulate_SIR
from src.Metrics import MetricCalculator, StatisticalAnalysis
from src.Plots import plot_epidemic_threshold, plot_degree_correlation

def run_single_simulation_H5(G_initial, beta, gamma, w, steps, p_rand, seed):
    """
    Executes a single run of the SIR simulation with adaptive and random rewiring.
    
    This function is optimized for speed using set operations for tracking node states,
    rather than iterating through the entire graph at every step. It calculates 
    Theoretical (HMF) and Effective (Microscopic) reproduction numbers at each step.

    Parameters:
        G_initial (nx.Graph): The starting network topology.
        beta (float): Infection probability.
        gamma (float): Recovery probability.
        w (float): Adaptive rewiring probability (S disconnects from I).
        steps (int): Maximum simulation steps.
        p_rand (float): Global random rewiring probability (noise).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple(np.array, np.array): Arrays containing the Theoretical R(t) and 
                                   Effective R(t) time series respectively.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Deep copy to ensure thread safety during parallel execution
    G = copy.deepcopy(G_initial)
    nodes = list(G.nodes())
    N = len(nodes)
    
    # Initialize State
    nx.set_node_attributes(G, "S", "state")
    
    # Patient Zero Selection (1% of population)
    num_init = max(1, int(0.01 * N))
    initial_infected = random.sample(nodes, num_init)
    for node in initial_infected:
        G.nodes[node]["state"] = "I"
        
    # Maintain sets for O(1) lookup speed
    S_nodes = set(nodes) - set(initial_infected)
    I_nodes = set(initial_infected)
    R_nodes = set()
    
    r_theory_list = []
    r_effective_list = []
    transmissibility = beta / gamma if gamma > 0 else 0

    # --- SIMULATION LOOP ---
    for step in range(steps):
        count_S = len(S_nodes)
        count_I = len(I_nodes)
        
        # 1. EXTINCTION CHECK & METRICS PADDING
        if count_I == 0:
            # If epidemic ends early, pad the R(t) arrays with zeros
            rem = steps - step
            r_theory_list.extend([0] * rem)
            r_effective_list.extend([0] * rem)
            break 

        # A. Theoretical R Calculation (Heterogeneous Mean Field - HMF)
        # Considers the moment ratio <k^2>/<k> of the degree distribution
        degrees = [d for n, d in G.degree()]
        k_mean = np.mean(degrees)
        k2_mean = np.mean([d**2 for d in degrees])
        if k_mean > 0:
            moment_ratio = k2_mean / k_mean - 1
            r_th = transmissibility * moment_ratio * (count_S / N)
        else:
            r_th = 0
        r_theory_list.append(r_th)

        # B. Effective R Calculation (Microscopic)
        # Direct measurement: average number of susceptible neighbors per infected node
        total_S_neighbors = 0
        for inf_node in I_nodes:
            # Fast intersection with S_nodes set
            neighbors = list(G.neighbors(inf_node))
            s_neigh = sum(1 for n in neighbors if n in S_nodes)
            total_S_neighbors += s_neigh
        
        avg_targets = total_S_neighbors / count_I
        r_eff = transmissibility * avg_targets
        r_effective_list.append(r_eff)

        # --- 2. DYNAMICS: Adaptive Rewiring (S cuts link with I) ---
        # Logic: Susceptible nodes identify Infected neighbors and may break the link
        if w > 0:
            discordant_edges = []
            for inf in I_nodes:
                for neigh in G.neighbors(inf):
                    if neigh in S_nodes:
                        discordant_edges.append((neigh, inf)) # (S, I)
            
            # Stochastic selection of edges to sever
            edges_to_cut = [e for e in discordant_edges if random.random() < w]
            
            for (s_node, i_node) in edges_to_cut:
                if G.has_edge(s_node, i_node):
                    G.remove_edge(s_node, i_node)
                    
                    # Rewire S to a random non-Infected node
                    # Attempt up to 5 times to find a valid target
                    for _ in range(5):
                        candidate = random.choice(nodes)
                        if candidate != s_node and candidate not in I_nodes and not G.has_edge(s_node, candidate):
                            G.add_edge(s_node, candidate)
                            break

        # --- 3. DYNAMICS: Infection & Recovery ---
        new_infected = set()
        new_recovered = set()
        
        # Infection Process (I -> S)
        for u in I_nodes:
            for v in G.neighbors(u):
                if v in S_nodes:
                    if random.random() < beta:
                        new_infected.add(v)
        
        # Recovery Process (I -> R)
        for u in I_nodes:
            if random.random() < gamma:
                new_recovered.add(u)
                
        # Apply State Updates
        for n in new_infected:
            G.nodes[n]["state"] = "I"
            S_nodes.remove(n)
            I_nodes.add(n)
            
        for n in new_recovered:
            G.nodes[n]["state"] = "R"
            I_nodes.remove(n)
            R_nodes.add(n)

        # --- 4. DYNAMICS: Global Random Rewiring ---
        # Background topological noise unrelated to disease state
        if p_rand > 0:
            all_edges = list(G.edges())
            num_edges = len(all_edges)
            if num_edges >= 2:
                num_rewires = int(num_edges * p_rand)
                if num_rewires > 0:
                    indices = np.random.choice(num_edges, size=min(num_rewires * 2, num_edges), replace=False)
                    
                    # Process edge swaps in pairs (X-Swap / Double Edge Swap)
                    for i in range(0, len(indices) - 1, 2):
                        idx1, idx2 = indices[i], indices[i+1]
                        if idx1 < len(all_edges) and idx2 < len(all_edges):
                            u, v = all_edges[idx1]
                            x, y = all_edges[idx2]

                            if G.has_edge(u, v) and G.has_edge(x, y):
                                if len({u, v, x, y}) == 4:
                                    if not G.has_edge(u, y) and not G.has_edge(x, v):
                                        G.remove_edge(u, v)
                                        G.remove_edge(x, y)
                                        G.add_edge(u, y)
                                        G.add_edge(x, v)

    return np.array(r_theory_list), np.array(r_effective_list)

def run_monte_carlo_H5(G, beta=0.13, gamma=0.08, w=0.3, steps=150, p_rand=0.2, n_sims=20):
    """
    Orchestrates multiple parallel simulations to generate robust statistical averages for R(t).
    
    Returns:
        tuple: Means and Standard Deviations for Theoretical and Effective R(t).
    """
    print(f"Running {n_sims} parallel simulations for Rt analysis...")
    
    seeds = np.random.randint(0, 100000, size=n_sims)
    
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(run_single_simulation_H5)(G, beta, gamma, w, steps, p_rand, seed) 
        for seed in seeds
    )
    
    # Data Aggregation
    max_len = max(len(r[0]) for r in results) if results else 0
    
    matrix_th = np.zeros((n_sims, max_len))
    matrix_eff = np.zeros((n_sims, max_len))
    
    for i, (r_th, r_eff) in enumerate(results):
        L = len(r_th)
        matrix_th[i, :L] = r_th
        matrix_eff[i, :L] = r_eff
    
    return (np.mean(matrix_th, axis=0), np.std(matrix_th, axis=0),
            np.mean(matrix_eff, axis=0), np.std(matrix_eff, axis=0))


def single_run_wrapper(G, beta, gamma, w, p_rand, max_steps, seed_node=None, windows=None):
    """
    Worker function for Targeted Analysis.
    
    Executes a standard SIR simulation (using the external simulate_SIR function)
    and computes specific metrics.
    
    Returns:
        tuple: (Global Final Epidemic Size, Dictionary of Dynamic Metrics)
    """
    N = len(G)
    
    # Setup Patient Zero
    if seed_node is None:
        patient_zero = random.choice(list(G.nodes()))
    else:
        patient_zero = seed_node

    # Prepare Graph
    G_sim = copy.deepcopy(G)
    nx.set_node_attributes(G_sim, "S", "state")
    G_sim.nodes[patient_zero]["state"] = "I"

    # Run Simulation (Using the standard engine from Epidemic.py)
    history = simulate_SIR(G_sim, beta, gamma, w, p_rand, max_steps)
    
    # 1. Global Metric: Final Size
    final_G = history[-1]
    infected_count = len([n for n, d in final_G.nodes(data=True) if d['state'] in ['R', 'I']])
    final_size = infected_count / N
    
    # 2. Local Metrics (Only if requested for targeted analysis)
    metrics = {}
    if seed_node is not None and windows is not None:
        for win in windows:
            m_data = MetricCalculator.get_time_series_metrics(history, seed_node, win)
            m_data['Final_Size'] = final_size 
            metrics[win] = m_data
            
    return final_size, metrics

def stratified_degree_sampling(G, n_samples=15, n_strata=5):
    """
    Selects a representative sample of nodes uniformly across the degree distribution.
    
    It divides the degree range into 'n_strata' bins and samples an equal number
    of nodes from each bin to ensure high, medium, and low degree nodes are analyzed.
    """
    degrees = np.array([G.degree(n) for n in G.nodes()])
    nodes = np.array(list(G.nodes()))
    
    min_d, max_d = degrees.min(), degrees.max()
    bins = np.linspace(min_d, max_d, n_strata + 1)  # n_strata+1 for n_strata bins
    
    # Calculate nodes per stratum dynamically to handle remainders
    samples_per_stratum = n_samples // n_strata
    remainder = n_samples % n_strata
    
    sampled = []
    for i in range(n_strata):
        mask = (degrees >= bins[i]) & (degrees < bins[i+1])
        candidates = nodes[mask]
        
        if len(candidates) > 0:
            # Distribute remainder count to the first few strata
            n_from_stratum = samples_per_stratum + (1 if i < remainder else 0)
            n_from_stratum = min(n_from_stratum, len(candidates))  # Don't exceed available
            sampled.extend(np.random.choice(candidates, n_from_stratum, replace=False))
    
    return sorted(sampled[:n_samples])  # Ensure exact count



def run_targeted_analysis(G, gamma, w, p_rand, max_steps=2000,
                          num_ratios=20, num_sims=15, n_jobs=-1,
                          analysis_ratio=None, windows=[5, 10, 15]):
    """
    Main pipeline for Targeted Node Analysis and Statistical Significance testing.
    
    Pipeline Steps:
    1. Global Threshold Sweep: Identify the epidemic phase transition.
    2. Targeted Selection: Pick nodes across degree strata.
    3. Monte Carlo: Run repeated simulations starting from specific seeds.
    4. Correlation Analysis: Spearman rank tests per node.
    5. Meta-Analysis: Aggregated statistical tests (Permutation, Wilcoxon) across the network.
    """
    
    

    # --- PHASE 1: GLOBAL THRESHOLD SWEEP ---
    print("=== PHASE 1: Generating Epidemic Threshold Plot ===")
    ratios = np.linspace(0, 4, num_ratios)
    
    avg_rew = []; std_rew = []
    avg_stat = []; std_stat = []
    
    for i, ratio in enumerate(ratios):
        beta = ratio * gamma
        
        # Parallel execution for Rewiring case
        results_rew = Parallel(n_jobs=n_jobs)(
            delayed(single_run_wrapper)(G, beta, gamma, w, p_rand, max_steps) 
            for _ in range(num_sims)
        )
        sizes_rew = [r[0] for r in results_rew]
        
        # Parallel execution for Static Control case (w=0, p_rand=0)
        results_stat = Parallel(n_jobs=n_jobs)(
            delayed(single_run_wrapper)(G, beta, gamma, 0, 0, max_steps) 
            for _ in range(num_sims)
        )
        sizes_stat = [r[0] for r in results_stat]
        
        avg_rew.append(np.mean(sizes_rew)); std_rew.append(np.std(sizes_rew))
        avg_stat.append(np.mean(sizes_stat)); std_stat.append(np.std(sizes_stat))
        
        if i % 5 == 0: print(f"Processing ratio {ratio:.2f}...")

    # Set analysis point to the middle of the range if not specified
    if analysis_ratio is None:
        analysis_ratio = ratios[len(ratios)//2]
    
    plot_epidemic_threshold(ratios, np.array(avg_rew), np.array(std_rew), 
                            np.array(avg_stat), np.array(std_stat), analysis_ratio, w)

    # --- PHASE 2: TARGETED NODE SELECTION ---
    print(f"\n=== PHASE 2: Targeted Analysis at Ratio {analysis_ratio:.2f} ===")
    
    target_nodes = stratified_degree_sampling(G, n_samples=50)

    starting_degrees = {n: G.degree(n) for n in target_nodes}
    
    beta_target = analysis_ratio * gamma
    dist_data = {n: [] for n in target_nodes}
    metric_data = {n: [] for n in target_nodes}

    # Create job list for parallel processing (Node X Simulation)
    jobs = [(n, i) for n in target_nodes for i in range(num_sims)]
    
    raw_results = Parallel(n_jobs=n_jobs)(
        delayed(single_run_wrapper)(G, beta_target, gamma, w, p_rand, max_steps, seed_node=n, windows=windows)
        for n, _ in jobs
    )
    
    # Unpack results into data structures
    for (n, _), res in zip(jobs, raw_results):
        size, metrics = res
        dist_data[n].append(size)
        metric_data[n].append(metrics)

    # --- PHASE 3: STATISTICAL ANALYSIS ---
    print("\n=== PHASE 3: Statistical Analysis ===")

    # 3.3 Correlation Analysis (SPEARMAN + WITHIN-NODE)
    # -------------------------------------------------------------------------
    from scipy.stats import spearmanr, ttest_1samp

    for plot_win in windows:
        print(f"\n-- Spearman Correlation Analysis for Window {plot_win} --")
        
        correlation_results = []
        
        for n in target_nodes:
            # Data Alignment: Ensure metrics and outcomes are paired correctly per run
            valid_pairs_mom = []
            valid_pairs_avg = []
            valid_pairs_peak = []
            
            # Iterate by index to guarantee alignment
            for idx in range(len(dist_data[n])):
                epi_size = dist_data[n][idx]
                
                if idx >= len(metric_data[n]):
                    break
                    
                run_result = metric_data[n][idx]
                
                # Check for valid data in the nested dictionary structure
                if isinstance(run_result.get(plot_win), dict):
                    mi = run_result[plot_win].get('Mean_Mi')
                    avg = run_result[plot_win].get('Mean_Avg_Window')
                    peak = run_result[plot_win].get('Final_Deg_At_Peak')
                    
                    # Only append non-NaN pairs
                    if mi is not None and not np.isnan(mi):
                        valid_pairs_mom.append((mi, epi_size))
                    if avg is not None and not np.isnan(avg):
                        valid_pairs_avg.append((avg, epi_size))
                    if peak is not None and not np.isnan(peak):
                        valid_pairs_peak.append((peak, epi_size))
            
            # Minimum sample size check
            if len(valid_pairs_mom) < 3:
                print(f"Skipping node {n}: insufficient data")
                continue
            
            # Unzip pairs for correlation calculation
            mom_vals, epi_mom = zip(*valid_pairs_mom)
            avg_vals, epi_avg = zip(*valid_pairs_avg) if valid_pairs_avg else ([], [])
            peak_vals, epi_peak = zip(*valid_pairs_peak) if valid_pairs_peak else ([], [])
            
            # Compute Spearman correlations
            corr_mom, p_mom = spearmanr(mom_vals, epi_mom)
            corr_avg, p_avg = spearmanr(avg_vals, epi_avg) if len(avg_vals) >= 3 else (np.nan, np.nan)
            corr_peak, p_peak = spearmanr(peak_vals, epi_peak) if len(peak_vals) >= 3 else (np.nan, np.nan)

            def format_corr(rho, p):
                if np.isnan(rho):
                    return "NaN"
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                return f"{rho:.3f}{sig}"
            
            correlation_results.append({
                'Node': n,
                'Start_Degree': starting_degrees[n],
                'ρ_Momentum': corr_mom,  # Storing numeric value for meta-analysis
                'p_Momentum': p_mom,
                'ρ_AvgDeg': corr_avg,
                'p_AvgDeg': p_avg,
                'ρ_DegPeak': corr_peak,
                'p_DegPeak': p_peak,
                'N_Sims': len(valid_pairs_mom)
            })
        
        # Display formatted table for the current window
        display_df = pd.DataFrame([{
            'Node': r['Node'],
            'Start_Degree': r['Start_Degree'],
            'ρ_Momentum': format_corr(r['ρ_Momentum'], r['p_Momentum']),
            'p_Momentum': f"{r['p_Momentum']:.4f}" if not np.isnan(r['p_Momentum']) else "NaN",
            'ρ_AvgDeg': format_corr(r['ρ_AvgDeg'], r['p_AvgDeg']),
            'p_AvgDeg': f"{r['p_AvgDeg']:.4f}" if not np.isnan(r['p_AvgDeg']) else "NaN",
            'ρ_DegPeak': format_corr(r['ρ_DegPeak'], r['p_DegPeak']),
            'p_DegPeak': f"{r['p_DegPeak']:.4f}" if not np.isnan(r['p_DegPeak']) else "NaN",
            'N_Sims': r['N_Sims']
        } for r in correlation_results])
        
        display(Markdown(f"#### Spearman Correlations (Within-Node) - Window {plot_win}"))
        display(display_df.set_index('Node'))
        
        # ★★★ META-ANALYSIS ★★★
        # Aggregating correlation coefficients across all target nodes to find systematic trends
        print(f"\n=== META-ANALYSIS ACROSS {len(correlation_results)} NODES ===")

        # Extract valid correlations (filter out NaNs)
        rho_mom = [r['ρ_Momentum'] for r in correlation_results if not np.isnan(r['ρ_Momentum'])]
        rho_avg = [r['ρ_AvgDeg'] for r in correlation_results if not np.isnan(r['ρ_AvgDeg'])]
        rho_peak = [r['ρ_DegPeak'] for r in correlation_results if not np.isnan(r['ρ_DegPeak'])]

        # 1. Descriptive Statistics
        meta_stats = pd.DataFrame({
            'Metric': ['Momentum', 'Avg Degree', 'Deg at Peak'],
            'Mean ρ': [np.mean(rho_mom), np.mean(rho_avg), np.mean(rho_peak)],
            'Std ρ': [np.std(rho_mom), np.std(rho_avg), np.std(rho_peak)],
            'Median ρ': [np.median(rho_mom), np.median(rho_avg), np.median(rho_peak)],
            'N_nodes': [len(rho_mom), len(rho_avg), len(rho_peak)]
        })
        display(meta_stats)

        # 2. Statistical Tests for Each Metric
        from scipy.stats import shapiro, wilcoxon, permutation_test

        def test_correlation_significance(rho_values, metric_name):
            """
            Comprehensive statistical testing with normality checks and robust alternatives.
            Tests if the mean correlation is significantly different from zero.
            """
            if len(rho_values) < 3:
                print(f"\n{metric_name}: Insufficient data (n={len(rho_values)})")
                return
            
            print(f"\n{'='*60}")
            print(f"{metric_name} - Statistical Tests")
            print(f"{'='*60}")
            
            # A. Normality Test (Shapiro-Wilk)
            # Determines if we can use parametric tests (T-test)
            if len(rho_values) >= 3:
                stat_shapiro, p_shapiro = shapiro(rho_values)
                print(f"\nShapiro-Wilk normality test: W={stat_shapiro:.4f}, p={p_shapiro:.4f}")
                if p_shapiro < 0.05:
                    print("  ⚠ Distribution is NOT normal → Use non-parametric tests")
                    use_nonparam = True
                else:
                    print("  ✓ Normality assumption satisfied → T-test is valid")
                    use_nonparam = False
            
            # B. T-Test (parametric)
            t_stat, p_ttest = ttest_1samp(rho_values, 0)
            print(f"\nT-test (H0: mean_ρ=0): t={t_stat:.3f}, p={p_ttest:.4f}")
            if not use_nonparam:
                if p_ttest < 0.05:
                    direction = "POSITIVE" if np.mean(rho_values) > 0 else "NEGATIVE"
                    print(f"  ✓ Significant {direction} correlation (p<0.05)")
                else:
                    print(f"  ✗ No significant correlation")
            else:
                print(f"  ⚠ Interpret with caution (normality violated)")
            
            # C. Wilcoxon Signed-Rank Test (non-parametric alternative)
            try:
                stat_wilcox, p_wilcox = wilcoxon(rho_values, alternative='two-sided')
                print(f"\nWilcoxon signed-rank test (H0: median_ρ=0): p={p_wilcox:.4f}")
                if p_wilcox < 0.05:
                    direction = "POSITIVE" if np.median(rho_values) > 0 else "NEGATIVE"
                    print(f"  ✓ Significant {direction} correlation (p<0.05)")
                else:
                    print(f"  ✗ No significant correlation")
            except Exception as e:
                print(f"\nWilcoxon test failed: {e}")
            
            # D. Permutation Test (Most robust, assumes no specific distribution)
            def statistic(x, axis):
                return np.mean(x, axis=axis)
            
            res = permutation_test((rho_values,), statistic, 
                                permutation_type='samples',
                                alternative='two-sided',
                                n_resamples=10000,
                                random_state=42)
            print(f"\nPermutation test (H0: mean_ρ=0): p={res.pvalue:.4f}")
            if res.pvalue < 0.05:
                direction = "POSITIVE" if np.mean(rho_values) > 0 else "NEGATIVE"
                print(f"  ✓ Significant {direction} correlation (p<0.05)")
            else:
                print(f"  ✗ No significant correlation")
            
            # E. Summary of Results
            sig_tests = sum([p_ttest < 0.05, p_wilcox < 0.05 if 'p_wilcox' in locals() else False, 
                             res.pvalue < 0.05])
            total_tests = 3 if 'p_wilcox' in locals() else 2
            print(f"\n>>> CONCLUSION: {sig_tests}/{total_tests} tests are significant")
            if sig_tests >= 2:
                print(f">>> Strong evidence of systematic correlation")
            elif sig_tests == 1:
                print(f">>> Weak/mixed evidence - interpret cautiously")
            else:
                print(f">>> No evidence of systematic correlation")

        # Apply tests to all metrics
        test_correlation_significance(rho_mom, "MOMENTUM")
        test_correlation_significance(rho_avg, "AVG DEGREE")
        test_correlation_significance(rho_peak, "DEGREE AT PEAK")

        # 3. Moderation Analysis
        # Checks if the strength of the correlation depends on the node's initial degree
        print(f"\n{'='*60}")
        print("MODERATION ANALYSIS")
        print(f"{'='*60}")

        degrees_for_corr = [r['Start_Degree'] for r in correlation_results if not np.isnan(r['ρ_Momentum'])]
        if len(degrees_for_corr) >= 3:
            rho_deg_effect, p_deg_effect = spearmanr(degrees_for_corr, rho_mom)
            print(f"\nCorrelation between initial_degree and correlation_strength (Momentum):")
            print(f"  ρ={rho_deg_effect:.3f}, p={p_deg_effect:.4f}")
            if p_deg_effect < 0.05:
                trend = "increases" if rho_deg_effect > 0 else "decreases"
                print(f"  → Predictive power {trend} with initial degree!")
            else:
                print(f"  → No moderation effect (correlation strength independent of degree)")

        # -------------------------------------------------------------------------

    # --- PHASE 4: TABLES ---
    final_tables = {}
    for win in windows:
        rows = []
        for n in target_nodes:
            runs = metric_data[n]
            vals_epi = dist_data[n]
            
            # Safe extraction of values across runs
            vals_mi = [r[win]['Mean_Mi'] for r in runs if isinstance(r.get(win), dict) and not np.isnan(r[win]['Mean_Mi'])]
            vals_avg = [r[win]['Mean_Avg_Window'] for r in runs if isinstance(r.get(win), dict) and not np.isnan(r[win]['Mean_Avg_Window'])]
            vals_static = [r[win]['Final_Deg_At_Peak'] for r in runs if isinstance(r.get(win), dict) and not np.isnan(r[win]['Final_Deg_At_Peak'])]
            vals_peak = [r[win]['Peak_Time'] for r in runs if isinstance(r.get(win), dict) and not np.isnan(r[win]['Peak_Time'])]
            
            # Check how often the epidemic peaked before our window filled up
            peak_before_window_count = sum(
                1
                for r in runs
                if isinstance(r.get(win), dict)
                and not np.isnan(r[win]['Peak_Time'])
                and r[win]['Peak_Time'] < win
            )
            
            rows.append({
                'Seed Node': n,
                'Start Deg': starting_degrees[n],
                'Final Size': f"{np.mean(vals_epi):.2f} ± {np.std(vals_epi)/np.sqrt(len(vals_epi)):.2f}",
                'Peak Time': f"{np.mean(vals_peak):.2f} ± {np.std(vals_peak)/np.sqrt(len(vals_peak)):.2f}" if vals_peak else "NaN",
                'Peak < Window Count': peak_before_window_count,
                'Mean Deg at Peak': f"{np.mean(vals_static):.2f} ± {np.std(vals_static)/np.sqrt(len(vals_static)):.2f}" if vals_static else "NaN",
                'Mean Momentum': f"{np.mean(vals_mi):.2f} ± {np.std(vals_mi)/np.sqrt(len(vals_mi)):.2f}" if vals_mi else "NaN",
                'Mean Avg Deg': f"{np.mean(vals_avg):.2f} ± {np.std(vals_avg)/np.sqrt(len(vals_avg)):.2f}" if vals_avg else "NaN"
            })
            
        df = pd.DataFrame(rows).set_index('Seed Node')
        final_tables[win] = df
        display(Markdown(f"#### Table for Window {win}"))
        display(df)

    return final_tables