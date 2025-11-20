import os
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

from src.utils import LANG_DICT

# ============================================================================
# CoNLL-U PARSING
# ============================================================================

def parse_ud_conllu(data_folder: str = "./data/PUD", lang: str = "en"):
    """
    Parse a Universal Dependencies CoNLL-U file and extract dependency graph information.
    
    For each sentence, computes:
    - Node degrees: k_i = out_degree(i) + in_degree(i) for each token
    - Dependency lengths: d_i = |head_index - dependent_index| for each edge
    - Number of odd-degree nodes: q (used in Euler path analysis)
    
    Args:
        data_folder: path to directory containing CoNLL-U files
        lang: language code (e.g., 'en', 'ar', 'zh')
    
    Returns:
        Dictionary mapping sentence_id to:
        {
            "degrees": list of node degrees [k_1, k_2, ..., k_n],
            "dep_lengths": list of dependency edge lengths [d_1, d_2, ..., d_{n-1}],
            "q": number of nodes with odd degree
        }
    """
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join(data_folder, f"{lang}_pud-ud-test.conllu")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sentence_id = 0
    language_data = defaultdict(set)

    # Parse CoNLL-U format
    for line in lines:
        # Skip comment lines
        if line.startswith("#"):
            continue
        
        # Empty line indicates sentence boundary
        if len(line.strip()) == 0:
            sentence_id += 1
            continue

        # Parse token line (10 tab-separated fields)
        words = line.strip().split("\t")
        
        # Skip multiword tokens (e.g., "1-2") and empty nodes (e.g., "1.1")
        if "-" in words[0] or "." in words[0]:
            continue
        
        # Extract head relationship (field 6 is HEAD index)
        parent_id = int(words[6])
        
        # Only add edge if not root (parent_id != 0)
        if parent_id != 0:
            token_id = int(words[0])
            language_data[sentence_id].add((parent_id, token_id))

    # Process each sentence's dependency graph
    processed_data = {}
    for pid, edges in language_data.items():
        # Build directed graph from edges
        G = nx.DiGraph()
        G.add_edges_from(edges)

        # Compute dependency lengths: |head - dependent|
        dependency_lengths = [abs(parent - child) for parent, child in edges]

        # Compute total degree (in + out) for each node
        degrees = [G.out_degree(node) + G.in_degree(node) for node in G.nodes()]

        # Count nodes with odd degree (relevant for Eulerian path theory)
        q = len([deg for deg in degrees if deg % 2 == 1])

        processed_data[pid] = {
            "degrees": degrees,
            "dep_lengths": dependency_lengths,
            "q": q
        }

    return processed_data

# ============================================================================
# STATISTICS COMPUTATION
# ============================================================================

def compute_stats():
    """
    Compute and save dependency tree metrics for all languages.
    
    For each language, processes all sentences and computes:
    - n: sentence length (number of tokens)
    - ⟨k²⟩: mean squared degree = (1/n) Σ k_i²
    - ⟨d⟩: mean dependency length = (1/(n-1)) Σ d_i
    - q: number of odd-degree nodes
    
    Output files:
        data/dependency_metrics/{language}_dependency_tree_metrics.csv
        Columns: phrase_id, n, ⟨k2⟩, <d>, q
    """
    os.makedirs("data/dependency_metrics", exist_ok=True)
    
    for lang in LANG_DICT.keys():
        # Parse CoNLL-U file for this language
        language_data = parse_ud_conllu(lang=lang)

        # Collect statistics for each sentence
        ids = []
        len_phrases = []
        mean_k2_values = []
        mean_dep_lengths = []
        q_values = []
        
        for pid, data in language_data.items():
            ids.append(pid)
            
            degs = data.get("degrees", [])
            lens = data.get("dep_lengths", [])
            
            # Sentence length
            n = len(degs)
            len_phrases.append(n)
            
            # Mean squared degree: ⟨k²⟩ = (1/n) Σ k_i²
            mean_k2 = np.mean([deg**2 for deg in degs]) if degs else 0
            mean_k2_values.append(mean_k2)
            
            # Mean dependency length: ⟨d⟩ = (1/(n-1)) Σ d_i
            mean_d = np.mean(lens) if lens else 0
            mean_dep_lengths.append(mean_d)
            
            # Number of odd-degree nodes
            q_values.append(data.get("q", 0))

        # Create DataFrame and save
        df = pd.DataFrame({
            "n": len_phrases,
            "⟨k2⟩": mean_k2_values,
            "<d>": mean_dep_lengths,
            "q": q_values
        }, index=ids)
        
        df.index.name = "phrase_id"
        df.sort_index(inplace=True)
        
        output_path = f"data/dependency_metrics/{LANG_DICT[lang]}_dependency_tree_metrics.csv"
        df.to_csv(output_path)
        print(f"Saved: {output_path}")

# ============================================================================
# DATA VALIDATION
# ============================================================================

def ensure_validity():
    """
    Validate computed metrics against theoretical bounds.
    
    Checks two inequalities for each sentence:
    
    1. Mean squared degree bound:
       4 - 6/n ≤ ⟨k²⟩ ≤ n - 1
    
    2. Mean dependency length bound (using mean of d_i):
       (1/4) · (n·⟨k²⟩ + q)/(n-1) ≤ ⟨d⟩ ≤ (1/4) · (3(n-1)² + 1 - n mod 2)/(n-1)
    
    Prints warnings for any sentences violating these bounds.
    """
    for lang in LANG_DICT.keys():
        filepath = f"data/dependency_metrics/{LANG_DICT[lang]}_dependency_tree_metrics.csv"
        df = pd.read_csv(filepath)
        
        violations_k2 = 0
        violations_d = 0
        
        for index, row in df.iterrows():
            n = row["n"]
            mean_k2 = row["⟨k2⟩"]
            mean_d = row["<d>"]
            q = row["q"]

            # Check ⟨k²⟩ bounds: 4 - 6/n ≤ ⟨k²⟩ ≤ n - 1
            lower_k2 = 4 - 6/n
            upper_k2 = n - 1
            
            if not (round(lower_k2, 7) <= round(mean_k2, 7) <= round(upper_k2, 7)): 
                print(f"⟨k²⟩ violation in {LANG_DICT[lang]} phrase {index}: "
                      f"{lower_k2:.4f} ≤ {mean_k2:.4f} ≤ {upper_k2:.4f}")
                violations_k2 += 1
            
            # Check ⟨d⟩ bounds (using mean of dependency lengths)
            # Lower bound: (1/4) · (n·⟨k²⟩ + q) / (n-1)
            # Upper bound: (1/4) · (3(n-1)² + 1 - n mod 2) / (n-1)
            if n > 1:  # Avoid division by zero for single-token sentences
                lower_d = (1/4) * (n * mean_k2 + q) / (n - 1)
                upper_d = (1/4) * (3 * (n - 1)**2 + 1 - n % 2) / (n - 1)
                
                if not (round(lower_d, 7) <= round(mean_d, 7) <= round(upper_d, 7)):
                    print(f"⟨d⟩ violation in {LANG_DICT[lang]} phrase {index}: "
                          f"{lower_d:.4f} ≤ {mean_d:.4f} ≤ {upper_d:.4f}")
                    violations_d += 1
        
        if violations_k2 == 0 and violations_d == 0:
            print(f"✓ {LANG_DICT[lang]}: All bounds satisfied")
        else:
            print(f"✗ {LANG_DICT[lang]}: {violations_k2} ⟨k²⟩ violations, {violations_d} ⟨d⟩ violations")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def summary_table():
    """
    Generate summary statistics table for all languages.
    
    Computes aggregate statistics across all sentences in each language:
    - N: number of sentences
    - mean_n, std_n: mean and standard deviation of sentence length
    - mean_k2, std_k2: mean and standard deviation of ⟨k²⟩
    
    Output file:
        data/summary_statistics.csv
        Columns: language, N, mean_n, std_n, mean_k2, std_k2
    """
    os.makedirs("data", exist_ok=True)

    results = {
        'language': [], 
        'N': [], 
        'mean_n': [], 
        'std_n': [], 
        'mean_k2': [], 
        'std_k2': []
    }
    
    for lang in LANG_DICT.keys():
        filepath = f"data/dependency_metrics/{LANG_DICT[lang]}_dependency_tree_metrics.csv"
        df = pd.read_csv(filepath)
        
        # Number of sentences
        n_sentences = df.shape[0]
        
        # Sentence length statistics
        mean_n = round(np.mean(df["n"]), 2)
        std_n = round(np.std(df["n"]), 2)

        # Mean squared degree statistics
        mean_k2 = round(np.mean(df["⟨k2⟩"]), 2)
        std_k2 = round(np.std(df["⟨k2⟩"]), 2)

        # Append to results
        results['language'].append(LANG_DICT[lang])
        results['N'].append(n_sentences)
        results['mean_n'].append(mean_n)
        results['std_n'].append(std_n)
        results['mean_k2'].append(mean_k2)
        results['std_k2'].append(std_k2)

    # Create and save summary DataFrame
    summary_df = pd.DataFrame(results)
    summary_df.to_csv("data/summary_statistics.csv", index=False)
    print("\nSaved: data/summary_statistics.csv")