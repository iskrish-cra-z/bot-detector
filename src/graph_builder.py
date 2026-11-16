"""
Graph Construction and Analysis Module.

Provides utilities for:
1. Building and analyzing the social graph from MGTAB data
2. Computing graph statistics (degree distribution, clustering, etc.)
3. Extracting subgraphs for visualization
4. Converting between different graph representations
"""

import numpy as np
import torch
import networkx as nx
from collections import Counter
from torch_geometric.utils import (
    to_networkx, 
    to_undirected, 
    degree,
    subgraph,
    is_undirected,
)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def analyze_graph(data, info=None):
    """
    Compute comprehensive graph statistics.
    
    Args:
        data: PyG Data object
        info: Optional dataset info dict
        
    Returns:
        Dict with graph statistics
    """
    edge_index = data.edge_index
    num_nodes = data.x.shape[0]
    num_edges = edge_index.shape[1]
    
    print(f"\n[GraphBuilder] Analyzing graph structure...")
    
    # Degree statistics
    row = edge_index[0]
    deg = degree(row, num_nodes=num_nodes)
    
    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'is_undirected': bool(is_undirected(edge_index)),
        'density': float(2 * num_edges / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0,
        'avg_degree': float(deg.mean()),
        'median_degree': float(deg.median()),
        'max_degree': int(deg.max()),
        'min_degree': int(deg.min()),
        'std_degree': float(deg.std()),
        'isolated_nodes': int((deg == 0).sum()),
    }
    
    # Degree distribution for bot vs human
    if hasattr(data, 'y') and data.y is not None:
        labels = data.y.numpy()
        for cls_idx in range(data.num_classes):
            cls_name = 'human' if cls_idx == 0 else 'bot'
            cls_mask = labels == cls_idx
            cls_deg = deg[cls_mask]
            stats[f'{cls_name}_avg_degree'] = float(cls_deg.mean())
            stats[f'{cls_name}_max_degree'] = int(cls_deg.max()) if len(cls_deg) > 0 else 0
            stats[f'{cls_name}_isolated'] = int((cls_deg == 0).sum())
    
    # Edge type distribution
    if hasattr(data, 'edge_type') and data.edge_type is not None:
        edge_types = data.edge_type.numpy()
        type_counts = Counter(edge_types.tolist())
        stats['edge_type_distribution'] = dict(type_counts)
    
    # Print summary
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Density: {stats['density']:.6f}")
    print(f"  Avg degree: {stats['avg_degree']:.2f} ± {stats['std_degree']:.2f}")
    print(f"  Max degree: {stats['max_degree']}")
    print(f"  Isolated nodes: {stats['isolated_nodes']}")
    if 'human_avg_degree' in stats:
        print(f"  Human avg degree: {stats['human_avg_degree']:.2f}")
        print(f"  Bot avg degree: {stats['bot_avg_degree']:.2f}")
    
    return stats


def build_networkx_graph(data, max_nodes=None, sample_seed=42):
    """
    Convert PyG Data to NetworkX graph for visualization and analysis.
    
    Args:
        data: PyG Data object
        max_nodes: Maximum nodes to include (for large graphs).
                  If None, uses all nodes.
        sample_seed: Random seed for node sampling
        
    Returns:
        NetworkX graph with node attributes
    """
    num_nodes = data.x.shape[0]
    
    if max_nodes is not None and num_nodes > max_nodes:
        # Sample a subgraph around high-degree nodes
        np.random.seed(sample_seed)
        
        # Sample mix of bot and human nodes
        labels = data.y.numpy()
        human_idx = np.where(labels == 0)[0]
        bot_idx = np.where(labels == 1)[0]
        
        n_human = min(max_nodes // 2, len(human_idx))
        n_bot = min(max_nodes - n_human, len(bot_idx))
        
        selected_human = np.random.choice(human_idx, n_human, replace=False)
        selected_bot = np.random.choice(bot_idx, n_bot, replace=False)
        selected = np.concatenate([selected_human, selected_bot])
        
        subset = torch.tensor(selected, dtype=torch.long)
        sub_edge_index, _, edge_mask = subgraph(
            subset, data.edge_index, relabel_nodes=True, 
            num_nodes=num_nodes, return_edge_mask=True
        )
        
        sub_labels = data.y[subset]
        sub_features = data.x[subset]
        
        G = nx.Graph()
        for i in range(len(subset)):
            G.add_node(i, 
                       label=int(sub_labels[i]),
                       original_id=int(subset[i]))
        
        edges = sub_edge_index.t().numpy()
        for src, dst in edges:
            if src != dst:  # Skip self-loops
                G.add_edge(int(src), int(dst))
        
        print(f"[GraphBuilder] Created subgraph: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges")
    else:
        # Convert full graph
        G = nx.Graph()
        labels = data.y.numpy() if data.y is not None else np.zeros(num_nodes)
        
        for i in range(num_nodes):
            G.add_node(i, label=int(labels[i]))
        
        edges = data.edge_index.t().numpy()
        for src, dst in edges:
            if src != dst:
                G.add_edge(int(src), int(dst))
        
        print(f"[GraphBuilder] Created full graph: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges")
    
    return G


def get_degree_distribution(data):
    """
    Compute degree distribution separated by class.
    
    Returns:
        Dict with degree arrays for each class
    """
    deg = degree(data.edge_index[0], num_nodes=data.x.shape[0])
    labels = data.y.numpy()
    
    result = {
        'all': deg.numpy(),
        'human': deg[labels == 0].numpy(),
        'bot': deg[labels == 1].numpy(),
    }
    
    return result


def get_neighbor_label_distribution(data):
    """
    For each node, compute the fraction of its neighbors that are bots.
    This is a key indicator — bots tend to be connected to other bots (homophily).
    
    Returns:
        numpy array of shape [N] with bot-neighbor ratios
    """
    num_nodes = data.x.shape[0]
    labels = data.y.numpy()
    edge_index = data.edge_index.numpy()
    
    bot_neighbor_ratio = np.zeros(num_nodes)
    
    for i in range(num_nodes):
        # Find all neighbors of node i
        neighbor_mask = edge_index[0] == i
        neighbors = edge_index[1, neighbor_mask]
        
        if len(neighbors) > 0:
            bot_ratio = labels[neighbors].mean()
            bot_neighbor_ratio[i] = bot_ratio
    
    return bot_neighbor_ratio


def make_undirected(data):
    """
    Convert a directed graph to undirected by adding reverse edges.
    
    Args:
        data: PyG Data object
        
    Returns:
        Modified Data object with undirected edges
    """
    if is_undirected(data.edge_index):
        print("[GraphBuilder] Graph is already undirected")
        return data
    
    edge_index = to_undirected(data.edge_index)
    data.edge_index = edge_index
    
    print(f"[GraphBuilder] Made graph undirected: {edge_index.shape[1]} edges")
    return data


if __name__ == "__main__":
    """Test graph analysis with dummy data."""
    # Create dummy graph
    num_nodes = 100
    num_edges = 500
    
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    x = torch.randn(num_nodes, 16)
    y = torch.randint(0, 2, (num_nodes,))
    
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_classes = 2
    data.edge_type = torch.randint(0, 2, (num_edges,))
    
    stats = analyze_graph(data)
    G = build_networkx_graph(data, max_nodes=50)
    deg_dist = get_degree_distribution(data)
    
    print(f"\nDegree distribution:")
    print(f"  All: mean={deg_dist['all'].mean():.2f}")
    print(f"  Human: mean={deg_dist['human'].mean():.2f}")
    print(f"  Bot: mean={deg_dist['bot'].mean():.2f}")
