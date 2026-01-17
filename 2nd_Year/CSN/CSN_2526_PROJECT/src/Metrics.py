import pandas as pd
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy.stats import ks_2samp, pearsonr

class MetricCalculator:
    """
    A utility class for calculating scalar and time-series metrics 
    related to node dynamics in a network.
    """

    @staticmethod
    def calculate_momentum(degrees_history: List[float], window: int, alpha: Optional[float] = None) -> float:
        """
        Calculates the 'Momentum' metric based on the rate of change in node degree.

        Momentum is defined as the current degree plus a scaled gradient of the degree 
        change over a specific window. It captures both the current state and the 
        trajectory of the node's connectivity.

        Formula: max(0, current_degree + alpha * gradient)

        Parameters:
            degrees_history (List[float]): A list of degree values over time.
            window (int): The lookback period for calculating the gradient.
            alpha (float, optional): Scaling factor for the gradient. 
                                     Defaults to window / 2.0 if not provided.

        Returns:
            float: The calculated momentum value (non-negative).
        """
        # Adjust window if history is shorter than the requested window
        if len(degrees_history) <= window:
            window = len(degrees_history) - 1
            if window < 1:
                return degrees_history[-1] if degrees_history else 0.0
            
        if alpha is None:
            alpha = window / 2.0
            
        curr = degrees_history[-1]
        past = degrees_history[-1 - window]
        
        # Calculate gradient (slope) and apply momentum formula
        gradient = (curr - past) / window
        momentum = max(0, curr + alpha * gradient)
        
        return momentum

    @staticmethod
    def get_time_series_metrics(history: List[nx.Graph], target_node: Any, window: int) -> Dict[str, float]:
        """
        Extracts temporal metrics (Momentum, Average Degree) for a specific node,
        analyzing the timeline up to the peak of the infection.

        Parameters:
            history (List[nx.Graph]): List of graph snapshots.
            target_node (Any): The identifier of the node to analyze.
            window (int): The sliding window size for metric calculations.

        Returns:
            Dict[str, float]: A dictionary containing mean momentum, mean average degree,
                              final degree at peak, and the time step of the peak.
        """
        # 1. Identify the Epidemic Peak
        # Count infected nodes in each snapshot
        active_counts = [
            len([n for n, d in g.nodes(data=True) if d.get("state") == "I"]) 
            for g in history
        ]
        peak_idx = np.argmax(active_counts)
        
        # 2. Truncate History to Peak
        # We only analyze dynamics leading up to the point of maximum spread
        trunc_history = history[:peak_idx + 1]
        degrees = [g.degree(target_node) for g in trunc_history]
        
        # 3. Adjust Window Size
        # Ensure the window does not exceed the available data length
        effective_window = min(window, len(degrees) - 1)
        
        if effective_window < 1:
            # Insufficient data points; return default/NaN values
            return {
                'Mean_Mi': degrees[0] if degrees else np.nan,
                'Mean_Avg_Window': degrees[0] if degrees else np.nan,
                'Final_Deg_At_Peak': degrees[-1] if degrees else np.nan,
                'Peak_Time': peak_idx
            }
        
        # 4. Compute Rolling Metrics
        mi_vals = []
        avg_vals = []
        
        # Iterate starting from the first valid window position
        for t in range(effective_window, len(degrees)):
            segment = degrees[:t+1]
            
            # Calculate Momentum for this segment
            mi = MetricCalculator.calculate_momentum(segment, effective_window)

            # Calculate Simple Moving Average (SMA)
            # Handle cases where the segment is shorter than the effective window (early steps)
            start_slice = max(0, t - effective_window)
            avg = np.mean(degrees[start_slice : t+1])
            
            mi_vals.append(mi)
            avg_vals.append(avg)
            
        return {
            'Mean_Mi': np.mean(mi_vals) if mi_vals else np.nan,
            'Mean_Avg_Window': np.mean(avg_vals) if avg_vals else np.nan,
            'Final_Deg_At_Peak': degrees[-1] if degrees else np.nan,
            'Peak_Time': peak_idx
        }


class StatisticalAnalysis:
    """
    A utility class for performing statistical comparisons and correlation analysis
    on network simulation results.
    """

    @staticmethod
    def compute_ks_matrix(node_ids: List[Any], distribution_dict: Dict[Any, List[float]]) -> np.ndarray:
        """
        Computes the Kolmogorov-Smirnov (KS) test p-value matrix for pairwise comparisons.

        This assesses whether the distributions of values (e.g., degree histories) for
        every pair of nodes are drawn from the same underlying distribution.

        Parameters:
            node_ids (List[Any]): List of node identifiers to compare.
            distribution_dict (Dict[Any, List[float]]): Map of node ID to its list of observations.

        Returns:
            np.ndarray: A square symmetric matrix where element [i, j] is the p-value 
                        comparing node i and node j.
        """
        n = len(node_ids)
        p_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    p_matrix[i, j] = 1.0
                else:
                    data_i = distribution_dict[node_ids[i]]
                    data_j = distribution_dict[node_ids[j]]
                    
                    # ks_2samp returns (statistic, pvalue)
                    _, p_val = ks_2samp(data_i, data_j)
                    p_matrix[i, j] = p_val
        return p_matrix

    @staticmethod
    def compute_degree_correlation(node_degrees: List[float], mean_outcomes: List[float]) -> Tuple[float, float, float, float]:
        """
        Computes the Pearson correlation and a linear fit between node degrees and outcomes.

        Parameters:
            node_degrees (List[float]): Independent variable (e.g., Initial Degree).
            mean_outcomes (List[float]): Dependent variable (e.g., Epidemic Size).

        Returns:
            Tuple containing:
            - Correlation coefficient (float)
            - P-value (float)
            - Slope (m) of the linear fit (float)
            - Intercept (b) of the linear fit (float)
        """
        if len(node_degrees) < 2:
            return 0.0, 0.0, 0.0, 0.0
            
        corr, p_val = pearsonr(node_degrees, mean_outcomes)
        
        # Perform Linear Regression (y = mx + b)
        m, b = np.polyfit(node_degrees, mean_outcomes, 1)
        
        return corr, p_val, m, b


def calculate_rolling_temporal_metrics(history: List[nx.Graph], window_size: int, alpha: float = 1.0) -> pd.DataFrame:
    """
    Calculates temporal metrics for all nodes using a sliding window over graph snapshots.
    
    This function vectorizes the calculation of Degree, Momentum, and Moving Averages 
    for the entire network across time.

    Parameters:
        history (List[nx.Graph]): A time-ordered list of NetworkX graph objects.
        window_size (int): The number of past steps to consider for the rolling window.
        alpha (float): Scaling factor for the momentum calculation.

    Returns:
        pd.DataFrame: A MultiIndex DataFrame (indexes: 'Node', 'Time') containing metrics.
                      Returns an empty DataFrame if history length <= window_size.
    """
    
    # 1. Pre-process Graph History into DataFrames
    # Converting graph properties to DataFrames allows for vectorized operations (much faster than looping)
    temporal_degrees = [dict(G.degree()) for G in history]
    df_degrees = pd.DataFrame(temporal_degrees).fillna(0)
    
    temporal_states = [{n: G.nodes[n].get('state', 'S') for n in G.nodes()} for G in history]
    df_states = pd.DataFrame(temporal_states)
    
    n_steps = len(df_degrees)
    results_list = []

    # Ensure we have enough data points to form at least one full window
    if n_steps <= window_size:
        return pd.DataFrame()

    # 2. Sliding Window Loop
    # We iterate from 'window_size' to n_steps to ensure we always have a full lookback period
    for t in range(window_size, n_steps):
        start_idx = t - window_size
        
        # Select the slice of history for the current window
        window_slice = df_degrees.iloc[start_idx : t + 1]
        
        # Get specific snapshots for calculation
        current_deg = df_degrees.iloc[t]
        past_deg = df_degrees.iloc[start_idx]
        current_states = df_states.iloc[t]
        
        # 3. Calculate Statistical Metrics
        avg_vals = window_slice.mean()
        
        # 4. Calculate Momentum Metric (Mi)
        # Formula: Current_Degree + Alpha * (Rate_of_Change)
        # Rate_of_Change is the slope between the start of the window and the current time
        rate_of_change = (current_deg - past_deg) / window_size
        raw_momentum = current_deg + (alpha * rate_of_change)
        
        # Clip momentum to ensure non-negative values (degrees cannot be effectively negative)
        momentum = raw_momentum.clip(lower=0)

        # 5. Assemble Data for Current Step
        step_df = pd.DataFrame({
            'Time': t,
            'Classic_Degree': current_deg,
            'Metric_Mi': momentum,
            'Avg_Degree_Window': avg_vals,
            'State': current_states
        })
        
        # Explicitly set the Node column from the index (which contains Node IDs)
        step_df['Node'] = step_df.index
        results_list.append(step_df)
        
    # 6. Final Concatenation
    if results_list:
        final_df = pd.concat(results_list, ignore_index=True)
        # Set MultiIndex for efficient querying by Node and Time
        final_df = final_df.set_index(['Node', 'Time'])
        return final_df
    else:
        return pd.DataFrame()