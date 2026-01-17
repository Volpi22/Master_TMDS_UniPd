import networkx as nx
import random
import copy

def simulate_SIR(G_original, beta, gamma, w, p_rand, max_steps, stop_on_extinction=True):
    """
    Simulates an SIR epidemic on a dynamic network involving two types of rewiring.

    The simulation proceeds in discrete time steps. At each step, the network topology 
    may change due to global random noise or adaptive behavior (susceptible nodes 
    avoiding infected neighbors), followed by the standard SIR infection and recovery processes.

    Parameters:
        G_original (nx.Graph): The initial network structure. Nodes must have a 'state' 
                               attribute set to 'S', 'I', or 'R'.
        beta (float): Infection probability (transmission rate per edge).
        gamma (float): Recovery probability (rate at which I becomes R).
        w (float): Adaptive Rewiring probability. The likelihood that a Susceptible node 
                   disconnects from an Infected neighbor and rewires to a safer node.
        p_rand (float): Global Random Rewiring probability. The likelihood that any node 
                        rewires a link randomly (background topological noise).
        max_steps (int): The maximum number of simulation time steps.
        stop_on_extinction (bool): If True, simulation stops early if the number of 
                                   Infected nodes drops to zero.

    Returns:
        list[nx.Graph]: A history of graph snapshots at each time step.
    """
    # Create a deep copy to ensure the original graph object remains unmodified
    G = copy.deepcopy(G_original)
    
    # Pre-calculate state sets for efficient O(1) lookups
    infected_nodes = {n for n, d in G.nodes(data=True) if d.get("state") == "I"}
    recovered_nodes = {n for n, d in G.nodes(data=True) if d.get("state") == "R"}
    
    history = []
    
    for step in range(max_steps):
        # Archive current state
        history.append(copy.deepcopy(G))
        
        # Early exit condition: Epidemic has died out
        if stop_on_extinction and not infected_nodes:
            break

        # ---------------------------------------------------------
        # Phase 1: Global Random Rewiring (Background Noise)
        # ---------------------------------------------------------
        if p_rand > 0:
            # Iterate over a static list of nodes to allow graph modification during iteration
            for u in list(G.nodes()):
                if random.random() < p_rand:
                    neighbors = list(G.neighbors(u))
                    
                    if neighbors:
                        # Select a random neighbor to disconnect
                        v = random.choice(neighbors)
                        G.remove_edge(u, v)
                        
                        # Find a new connection target
                        # We look for a node that is not 'u' and not already a neighbor
                        possible_target = None
                        while True:
                            candidate = random.choice(list(G.nodes()))
                            if candidate != u and not G.has_edge(u, candidate):
                                possible_target = candidate
                                break
                        
                        G.add_edge(u, possible_target)

        # ---------------------------------------------------------
        # Phase 2: Adaptive Rewiring (Behavioral Response)
        # ---------------------------------------------------------
        # Logic: Susceptible nodes attempt to break links with Infected neighbors
        if w > 0:
            for u in list(G.nodes()):
                # Only Susceptible nodes perform this adaptive action
                if G.nodes[u].get("state") == "S": 
                    
                    # Identify neighbors that are currently Infected
                    infected_neighbors = [n for n in G.neighbors(u) if G.nodes[n].get("state") == "I"]
                    
                    for v in infected_neighbors:
                        if random.random() < w:
                            # Search for "safe" rewiring targets (Susceptible or Recovered nodes)
                            # to which 'u' is not already connected.
                            valid_candidates = [
                                node for node in G.nodes()
                                if node != u 
                                and not G.has_edge(u, node)
                                and G.nodes[node].get("state") in ["S", "R"]
                            ]
                            
                            # Crucial Check: Only sever the tie if a valid replacement exists.
                            # If no safe node is available, the risky link remains.
                            if valid_candidates:
                                G.remove_edge(u, v)
                                candidate = random.choice(valid_candidates)
                                G.add_edge(u, candidate)

        # ---------------------------------------------------------
        # Phase 3: Infection & Recovery Dynamics
        # ---------------------------------------------------------
        new_infected = set()
        new_recovered = set()
        
        # A. Transmission (I -> S)
        for u in list(infected_nodes):
            for v in G.neighbors(u):
                # Identify Susceptible neighbors (State is not I and not R)
                if v not in infected_nodes and v not in recovered_nodes:
                    if random.random() < beta:
                        new_infected.add(v)
            
            # B. Recovery (I -> R)
            if random.random() < gamma:
                new_recovered.add(u)
        
        # Apply state updates synchronously
        for n in new_infected:
            G.nodes[n]["state"] = "I"
            infected_nodes.add(n)
            
        for n in new_recovered:
            G.nodes[n]["state"] = "R"
            infected_nodes.remove(n)
            recovered_nodes.add(n)

    return history