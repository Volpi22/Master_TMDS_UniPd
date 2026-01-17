import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import networkx as nx
from src.Metrics import calculate_rolling_temporal_metrics

def plot_evolution(history, title="Epidemic Evolution"):
    """
    Visualizes the aggregate epidemic dynamics by plotting the number of nodes 
    in each compartment (Susceptible, Infected, Recovered) over time.

    Parameters:
        history (List[nx.Graph]): A time-ordered list of graph snapshots representing 
                                  the simulation states.
        title (str): The title of the generated plot.
    """
    S = []; I = []; R = []
    for G in history:
        states = [G.nodes[n]['state'] for n in G.nodes()]
        S.append(states.count('S'))
        I.append(states.count('I'))
        R.append(states.count('R'))
        
    steps = range(len(history))
    plt.figure(figsize=(10, 6))
    plt.plot(steps, S, label='Susceptible', color='blue', alpha=0.7)
    plt.plot(steps, I, label='Infected', color='red', alpha=0.7)
    plt.plot(steps, R, label='Recovered', color='green', alpha=0.7)
    plt.xlabel('Time Steps'); plt.ylabel('Number of Nodes')
    plt.title(title); plt.legend(); plt.grid(True, alpha=0.3)
    plt.show()

def plot_Rt_comparison_H5(avg_th, std_th, avg_eff, std_eff, w):
    """
    Compares the theoretical (HMF) and effective (microscopic) reproduction numbers (R_t) 
    over the course of the epidemic.

    Parameters:
        avg_th (np.ndarray): The mean theoretical R_t curve (Heterogeneous Mean Field approximation).
        std_th (np.ndarray): The standard deviation of the theoretical R_t.
        avg_eff (np.ndarray): The mean effective R_t curve observed in the simulation.
        std_eff (np.ndarray): The standard deviation of the effective R_t.
        w (float): The rewiring probability parameter used in the simulation, for context in the title.
    """
    plt.figure(figsize=(10, 6))
    t = np.arange(len(avg_th))

    # Plot Theoretical curve with confidence interval
    plt.plot(t, avg_th, label='Theoretical (HMF)', color='blue', linestyle='--')
    plt.fill_between(t, avg_th - std_th, avg_th + std_th, color='blue', alpha=0.15)

    # Plot Effective curve with confidence interval
    plt.plot(t, avg_eff, label='Effective (Microscopic)', color='red')
    plt.fill_between(t, avg_eff - std_eff, avg_eff + std_eff, color='red', alpha=0.15)

    plt.axhline(1, color='black', linestyle=':', label='Threshold R=1')
    plt.xlabel('Time Steps'); plt.ylabel('Reproduction Number R(t)')
    plt.title(f'Temporal Evolution of R(t) in Adaptive Network (w={w})')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout(); plt.show()

def plot_epidemic_threshold(ratios, y_rew, err_rew, y_stat, err_stat, target_ratio, w):
    """
    Plots the final epidemic size against the transmission ratio to identify the epidemic threshold.
    Compares a static network against the adaptive rewiring scenario.

    Parameters:
        ratios (List[float]): The x-axis values representing the ratio beta/gamma.
        y_rew (np.ndarray): The final epidemic sizes for the rewiring scenario.
        err_rew (np.ndarray): Error bars (std dev) for the rewiring scenario.
        y_stat (np.ndarray): The final epidemic sizes for the static control scenario.
        err_stat (np.ndarray): Error bars (std dev) for the static scenario.
        target_ratio (float): A specific ratio value to highlight on the plot (e.g., critical threshold).
        w (float): The rewiring probability used.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(ratios, y_stat, 'k--', label='Static Network')
    plt.fill_between(ratios, y_stat - err_stat, y_stat + err_stat, color='k', alpha=0.1)
    plt.plot(ratios, y_rew, 'r-', label=f'Rewiring (w={w})')
    plt.fill_between(ratios, y_rew - err_rew, y_rew + err_rew, color='r', alpha=0.15)
    plt.axvline(target_ratio, color='blue', linestyle=':', label='Analysis Point')
    plt.title("Phase 1: Epidemic Threshold Analysis")
    plt.xlabel(r"Transmission Ratio ($\beta / \gamma$)"); plt.ylabel("Final Epidemic Size")
    plt.legend(); plt.grid(alpha=0.5); plt.show()

def plot_degree_correlation(metric_dict, outcomes, window_val):
    """
    Generates scatter plots correlating various node metrics (e.g., initial degree, momentum) 
    with the final epidemic outcome.

    Produces a row of subplots, one for each metric provided in `metric_dict`.

    Parameters:
        metric_dict (Dict[str, List[float]]): A dictionary mapping metric names to lists of metric values.
        outcomes (List[float]): A list of outcome values (e.g., mean epidemic size) corresponding to the metrics.
        window_val (int): The window size used for calculation, displayed in the figure title.
    """
    n_metrics = len(metric_dict)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), sharey=True)
    if n_metrics == 1:
        axes = [axes]

    y_vals = np.asarray(outcomes, dtype=float)

    for ax, (metric_name, x_vals) in zip(axes, metric_dict.items()):
        x_vals = np.asarray(x_vals, dtype=float)

        # 1. Statistical Analysis (Pearson Correlation + Linear Fit)
        if len(x_vals) > 1 and np.std(x_vals) > 0 and np.std(y_vals) > 0:
            corr_matrix = np.corrcoef(x_vals, y_vals)
            corr = corr_matrix[0, 1] if not np.isnan(corr_matrix).any() else 0.0
            m, b = np.polyfit(x_vals, y_vals, 1)
        else:
            corr, m, b = 0.0, 0.0, 0.0

        # 2. Scatter Plot Generation
        ax.scatter(
            x_vals, y_vals,
            s=45,
            color='#2b8cbe',
            edgecolor='black',
            linewidth=0.7,
            alpha=0.85,
            zorder=2
        )

        # 3. Regression Line Plotting
        if len(x_vals) > 1 and np.std(x_vals) > 0:
            x_range = np.linspace(np.min(x_vals), np.max(x_vals), 200)
            ax.plot(x_range, m * x_range + b, 'r--', lw=2, label=f'r = {corr:.2f}')
        else:
            ax.plot([], [], 'r--', lw=2, label=f'r = {corr:.2f}')

        # 4. Axis Formatting
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xlabel("Metric Value")
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

    axes[0].set_ylabel("Mean Final Epidemic Size", fontsize=12)
    plt.suptitle(f"Metric Correlations for Time Window: {window_val}", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()

def plot_full_comparison_row(history, target_nodes, max_window_size=None, sensitivity_step=5):
    """
    Constructs a comprehensive visualization grid for analyzing specific nodes.
    
    The layout consists of:
    - Left Column: Sensitivity analysis (Metric values vs Window Size).
    - Right Columns: Time-series plots of node metrics for selected window sizes.

    Parameters:
        history (List[nx.Graph]): The simulation history.
        target_nodes (List[int]): The IDs of the nodes to analyze.
        max_window_size (int, optional): The maximum window size to scan. Defaults to half the history length.
        sensitivity_step (int): The step size for the window sensitivity scan.
    """
    sim_length = len(history)
    if max_window_size is None: max_window_size = sim_length // 2

    # Define the range of window sizes to analyze
    all_windows = list(range(5, max_window_size + 1, sensitivity_step))
    indices = np.linspace(0, len(all_windows) - 1, 5, dtype=int)
    display_windows = sorted(list(set([all_windows[i] for i in indices])))
    
    print(f"Analyzing Nodes: {target_nodes}")
    print(f"Displaying Time Series for Windows: {display_windows}")

    # Compute metrics for all windows and cache necessary data
    time_series_cache = {}
    sensitivity_data = {n: {'windows': [], 'Classic_Degree': [], 'Metric_Mi': [], 'Avg_Degree_Window': []} for n in target_nodes}
        
    for w in all_windows:
        df = calculate_rolling_temporal_metrics(history, window_size=w, alpha=w/2)
        if df.empty: continue

        if w in display_windows: time_series_cache[w] = df
        
        means = df.reset_index().groupby('Node')[['Classic_Degree', 'Metric_Mi', 'Avg_Degree_Window']].mean()
        for node in target_nodes:
            if node in means.index:
                sensitivity_data[node]['windows'].append(w)
                sensitivity_data[node]['Classic_Degree'].append(means.loc[node, 'Classic_Degree'])
                sensitivity_data[node]['Metric_Mi'].append(means.loc[node, 'Metric_Mi'])
                sensitivity_data[node]['Avg_Degree_Window'].append(means.loc[node, 'Avg_Degree_Window'])

    # Configure plot grid dimensions
    n_rows = len(target_nodes)
    n_cols = 1 + len(display_windows)
    
    # Create figure with extra vertical space for the global legend
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows + 1))
    if n_rows == 1: axes = axes.reshape(1, -1)

    styles = {
        'Classic_Degree':    {'color': 'blue', 'marker': 'o', 'label': 'Classic Degree'},
        'Metric_Mi':         {'color': 'red',  'marker': 's', 'label': 'Momentum (Mi)'},
        'Avg_Degree_Window': {'color': 'green', 'marker': '^', 'label': 'Window Avg'},
    }

    for row_idx, node in enumerate(target_nodes):
        # A. Plot Sensitivity (Leftmost Column)
        ax_sens = axes[row_idx, 0]
        node_sens = sensitivity_data[node]
        for metric, style in styles.items():
            if node_sens['windows']:
                ax_sens.plot(node_sens['windows'], node_sens[metric], **style, alpha=0.8, markersize=4)
        ax_sens.set_ylabel(f"NODE {node}\nAvg Metric"); ax_sens.grid(True, ls='--', alpha=0.4)
        if row_idx == 0: ax_sens.set_title("SENSITIVITY\n(Mean Variation vs Window)")

        # B. Plot Time Series (Right Columns)
        for col_idx, w in enumerate(display_windows):
            ax_ts = axes[row_idx, col_idx + 1]
            if w in time_series_cache and node in time_series_cache[w].index:
                df_w = time_series_cache[w].loc[node]
                ax_ts.plot(df_w.index, df_w['Classic_Degree'], 'blue', alpha=0.3, label='Classic')
                ax_ts.plot(df_w.index, df_w['Metric_Mi'], 'red', lw=2, label='Momentum')
                ax_ts.plot(df_w.index, df_w['Avg_Degree_Window'], 'green', ls='--', label='Avg')
                
                # Add vertical lines for state changes (Infection/Recovery)
                states = df_w['State']
                times = df_w.index.get_level_values('Time')
                prev_st = states.iloc[0]
                for t, curr_st in zip(times, states):
                    if curr_st != prev_st:
                        col = 'red' if (prev_st=='S' and curr_st=='I') else ('green' if (prev_st=='I' and curr_st=='R') else None)
                        if col: ax_ts.axvline(x=t, color=col, ls=':', alpha=0.5)
                        prev_st = curr_st
            
            ax_ts.grid(True, ls=':', alpha=0.4)
            if row_idx == 0: ax_ts.set_title(f"TIME SERIES\nW={w}")
            if row_idx == n_rows - 1: ax_ts.set_xlabel("Steps")

    # --- Global Legend Configuration ---
    
    # Create custom legend handles for clarity
    legend_elements = [
        # Metric lines
        Line2D([0], [0], color='blue', lw=2, label='Classic Degree'),
        Line2D([0], [0], color='red', lw=2, label='Momentum (Mi)'),
        Line2D([0], [0], color='green', lw=2, linestyle='--', label='Window Avg'),
        # Event markers
        Line2D([0], [0], color='red', linestyle=':', label='Infection Event'),
        Line2D([0], [0], color='green', linestyle=':', label='Recovery Event')
    ]

    # Position the legend at the top center of the figure
    fig.legend(handles=legend_elements, loc='upper center', ncol=5, frameon=True)

    # Adjust layout to prevent overlap with the top legend
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()