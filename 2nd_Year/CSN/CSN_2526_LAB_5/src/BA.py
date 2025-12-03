import numpy as np
from typing import List
from collections import defaultdict

def _simulate_ba_preferential(n0: int, m0: int, tmax: int, arrival_times: List[int]):
    """
    Simulates the Barabási–Albert (BA) model with preferential attachment.

    Parameters:
        n0 (int): Initial number of nodes (complete graph).
        m0 (int): Number of edges each new node brings.
        tmax (int): Number of time steps (new nodes added).
        arrival_times (List[int]): Time steps at which to record degree evolution.

    Returns:
        vertex_time_series (dict): Degree evolution for selected nodes at arrival times.
        degrees (np.ndarray): Final degree sequence of all nodes.
    """
    N_final = n0 + tmax
    degrees = np.zeros(N_final, dtype=int)

    # Build initial complete graph and stubs list for preferential attachment
    stubs = []
    for u in range(n0):
        for v in range(u + 1, n0):
            stubs.extend([u, v])
            degrees[u] += 1
            degrees[v] += 1

    vertex_time_series = {i: {} for i in arrival_times}

    # Add new nodes one at a time
    for t in range(1, tmax + 1):
        new_vertex = n0 + t - 1
        degrees_new = 0
        L = len(stubs)

        # Select m0 unique targets preferentially (by degree)
        targets = set()
        while len(targets) < m0:
            v = stubs[np.random.randint(0, L)]
            if v != new_vertex:
                targets.add(v)

        # Connect new node to targets and update stubs/degrees
        for v in targets:
            degrees[v] += 1
            degrees_new += 1
            stubs.extend([v, new_vertex])

        degrees[new_vertex] = degrees_new

        # Record degree evolution for selected nodes
        if t in arrival_times:
            vertex_time_series[t][new_vertex] = []

        for i in arrival_times:
            for vid in vertex_time_series[i]:
                vertex_time_series[i][vid].append(degrees[vid])

    return vertex_time_series, degrees

def _simulate_ba_random(n0: int, m0: int, tmax: int, arrival_times: List[int]):
    """
    Simulates the BA model with random attachment (no preferential mechanism).

    Parameters:
        n0 (int): Initial number of nodes (complete graph).
        m0 (int): Number of edges each new node brings.
        tmax (int): Number of time steps (new nodes added).
        arrival_times (List[int]): Time steps at which to record degree evolution.

    Returns:
        vertex_time_series (dict): Degree evolution for selected nodes at arrival_times.
        degrees (np.ndarray): Final degree sequence of all nodes.
    """
    import random

    N_final = n0 + tmax
    degrees = np.zeros(N_final, dtype=int)
    degrees[:n0] = n0 - 1
    vertex_time_series = {i: {} for i in arrival_times}

    # Add new nodes one at a time
    for t in range(1, tmax + 1):
        new_vertex = n0 + t - 1
        # Select m0 unique targets randomly
        targets = np.array(random.sample(range(n0 + t - 1), min(m0, n0 + t - 1)))

        degrees[targets] += 1
        degrees[new_vertex] = len(targets)

        # Record degree evolution for selected nodes
        if t in arrival_times:
            vertex_time_series[t][new_vertex] = []

        for i in arrival_times:
            for vid in vertex_time_series[i]:
                vertex_time_series[i][vid].append(degrees[vid])

    return vertex_time_series, degrees

def no_growth_pa(n0: int, m0: int, tmax: int, seed: int):
    """
    Simulates the 'no growth' preferential attachment model on a fixed-size graph.
    Ensures a simple graph (no duplicate edges, no self-loops).

    Parameters:
        n0 (int): Number of nodes (fixed, no growth).
        m0 (int): Number of edges to add per time step.
        tmax (int): Number of time steps.
        seed (int): Random seed for reproducibility.

    Returns:
        degree_evolution (defaultdict): Degree evolution for all nodes over time.
        final_degree_sequence (List[int]): Final degree sequence of all nodes.
    """
    rng = np.random.default_rng(seed)

    degrees = np.zeros(n0, dtype=int)
    adj = [set() for _ in range(n0)]
    k_all = np.zeros((n0, tmax+1), dtype=int)
    k_all[:, 0] = degrees

    # For each time step, add m0 edges according to preferential attachment
    for t in range(1, tmax+1):

        # Step 1: pick node i uniformly at random
        i = rng.integers(n0)

        for _ in range(m0):

            # Compute preferential weights (degree-based)
            total_deg = degrees.sum()
            if total_deg == 0:
                # If all degrees are zero, pick a non-self node uniformly
                j = rng.integers(n0 - 1)
                j = j + 1 if j >= i else j
            else:
                probs = degrees.astype(float)
                probs[i] = 0  # Avoid self-loop
                s = probs.sum()
                if s > 0:
                    probs /= s
                    # Rejection sampling to avoid multi-edges and self-loops
                    for _att in range(20):
                        j = rng.choice(n0, p=probs)
                        if j != i and j not in adj[i]:
                            break
                    else:
                        continue  # If no valid j found, skip this attempt
                else:
                    # If all other degrees are zero, pick a non-self node uniformly
                    j = rng.integers(n0 - 1)
                    j = j + 1 if j >= i else j

            # Add edge only once (simple graph)
            adj[i].add(j)
            adj[j].add(i)
            degrees[i] += 1
            degrees[j] += 1

        k_all[:, t] = degrees

    # Build degree evolution dictionary for all nodes
    degree_evolution = defaultdict(list)
    for node in range(n0):
        degree_evolution[node] = k_all[node, :].tolist()

    final_degree_sequence = degrees.tolist()

    return degree_evolution, final_degree_sequence