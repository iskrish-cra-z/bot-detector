"""
Visualization Module for Bot Detection Project.

Generates publication-quality visualizations:
1. Graph structure (NetworkX plot with bot/human coloring)
2. Embedding space (t-SNE/PCA of GAT node embeddings)
3. Attention weight analysis
4. Feature distributions (bot vs human)
5. Degree distribution analysis
6. Cluster analysis
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.graph_builder import build_networkx_graph, get_degree_distribution
from src.feature_engineering import MGTAB_FEATURE_NAMES

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_graph_structure(data, save_dir=None, max_nodes=500):
    """
    Visualize the social graph with bot/human node coloring.
    
    Creates a spring-layout plot where:
    - Green nodes = Human accounts
    - Red nodes = Bot accounts
    - Node size proportional to degree
    - Edges shown with transparency
    """
    if save_dir is None:
        save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    print("[Visualize] Creating graph structure visualization...")
    
    G = build_networkx_graph(data, max_nodes=max_nodes)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Compute layout
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Separate nodes by label
    human_nodes = [n for n, d in G.nodes(data=True) if d.get('label', 0) == 0]
    bot_nodes = [n for n, d in G.nodes(data=True) if d.get('label', 0) == 1]
    
    # Node sizes based on degree
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    
    human_sizes = [20 + 200 * (degrees[n] / max_deg) for n in human_nodes]
    bot_sizes = [20 + 200 * (degrees[n] / max_deg) for n in bot_nodes]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.05, width=0.5, ax=ax)
    
    # Draw human nodes
    nx.draw_networkx_nodes(G, pos, nodelist=human_nodes, node_color=config.VIS_CONFIG['color_human'],
                          node_size=human_sizes, alpha=0.7, ax=ax, label='Human')
    
    # Draw bot nodes
    nx.draw_networkx_nodes(G, pos, nodelist=bot_nodes, node_color=config.VIS_CONFIG['color_bot'],
                          node_size=bot_sizes, alpha=0.8, ax=ax, label='Bot')
    
    ax.set_title(f'Twitter Social Graph — Bot vs Human Accounts\n'
                f'({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, markerscale=2, loc='upper left')
    ax.axis('off')
    
    filepath = os.path.join(save_dir, 'graph_structure.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved to {filepath}")


def plot_embeddings_tsne(model, data, model_name='GAT', save_dir=None):
    """
    Visualize node embeddings using t-SNE dimensionality reduction.
    
    Extracts embeddings from the last hidden layer of the GNN model,
    projects them to 2D using t-SNE, and colors by true label.
    """
    if save_dir is None:
        save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[Visualize] Computing t-SNE for {model_name} embeddings...")
    
    # Get embeddings
    device = config.DEVICE
    model = model.to(device)
    data_gpu = data.to(device)
    
    embeddings = model.get_embeddings(data_gpu.x, data_gpu.edge_index,
                                      data_gpu.edge_type if hasattr(data_gpu, 'edge_type') else None)
    embeddings = embeddings.cpu().numpy()
    labels = data.y.cpu().numpy()
    
    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=config.VIS_CONFIG['tsne_perplexity'],
        n_iter=config.VIS_CONFIG['tsne_n_iter'],
        random_state=42,
        learning_rate='auto',
        init='pca',
    )
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Colored by TRUE label
    ax = axes[0]
    for cls_idx, cls_name, color in [(0, 'Human', config.VIS_CONFIG['color_human']),
                                      (1, 'Bot', config.VIS_CONFIG['color_bot'])]:
        mask = labels == cls_idx
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=color, label=cls_name, alpha=0.5, s=10, edgecolors='none')
    
    ax.set_title(f'{model_name} Embeddings (t-SNE) — True Labels',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, markerscale=3)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # Right: Colored by PREDICTED label
    model.eval()
    with torch.no_grad():
        out = model(data_gpu.x, data_gpu.edge_index,
                   data_gpu.edge_type if hasattr(data_gpu, 'edge_type') else None)
        preds = out.argmax(dim=1).cpu().numpy()
    
    ax = axes[1]
    for cls_idx, cls_name, color in [(0, 'Human', config.VIS_CONFIG['color_human']),
                                      (1, 'Bot', config.VIS_CONFIG['color_bot'])]:
        mask = preds == cls_idx
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=color, label=cls_name, alpha=0.5, s=10, edgecolors='none')
    
    ax.set_title(f'{model_name} Embeddings (t-SNE) — Predicted Labels',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, markerscale=3)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    filepath = os.path.join(save_dir, f'{model_name.lower()}_tsne_embeddings.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {filepath}")


def plot_embeddings_pca(model, data, model_name='GAT', save_dir=None):
    """
    Visualize node embeddings using PCA (faster alternative to t-SNE).
    """
    if save_dir is None:
        save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[Visualize] Computing PCA for {model_name} embeddings...")
    
    device = config.DEVICE
    model = model.to(device)
    data_gpu = data.to(device)
    
    embeddings = model.get_embeddings(data_gpu.x, data_gpu.edge_index,
                                      data_gpu.edge_type if hasattr(data_gpu, 'edge_type') else None)
    embeddings = embeddings.cpu().numpy()
    labels = data.y.cpu().numpy()
    
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for cls_idx, cls_name, color in [(0, 'Human', config.VIS_CONFIG['color_human']),
                                      (1, 'Bot', config.VIS_CONFIG['color_bot'])]:
        mask = labels == cls_idx
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=color, label=cls_name, alpha=0.5, s=10, edgecolors='none')
    
    ax.set_title(f'{model_name} Node Embeddings (PCA)\n'
                f'Explained variance: {pca.explained_variance_ratio_.sum():.2%}',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, markerscale=3)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    
    plt.tight_layout()
    filepath = os.path.join(save_dir, f'{model_name.lower()}_pca_embeddings.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {filepath}")


def plot_degree_distribution(data, save_dir=None):
    """
    Plot degree distribution separated by bot/human class.
    """
    if save_dir is None:
        save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    print("[Visualize] Plotting degree distribution...")
    
    deg_dist = get_degree_distribution(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Histogram
    ax = axes[0]
    max_deg = min(int(np.percentile(deg_dist['all'], 99)), 200)
    bins = np.linspace(0, max_deg, 50)
    
    ax.hist(deg_dist['human'], bins=bins, alpha=0.6, color=config.VIS_CONFIG['color_human'],
           label=f'Human (mean={deg_dist["human"].mean():.1f})', density=True)
    ax.hist(deg_dist['bot'], bins=bins, alpha=0.6, color=config.VIS_CONFIG['color_bot'],
           label=f'Bot (mean={deg_dist["bot"].mean():.1f})', density=True)
    
    ax.set_xlabel('Degree', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Node Degree Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    
    # Right: Log-log plot
    ax = axes[1]
    for subset, label, color in [
        (deg_dist['human'], 'Human', config.VIS_CONFIG['color_human']),
        (deg_dist['bot'], 'Bot', config.VIS_CONFIG['color_bot']),
    ]:
        deg_vals, counts = np.unique(subset.astype(int), return_counts=True)
        deg_vals = deg_vals[deg_vals > 0]
        counts = counts[len(counts)-len(deg_vals):]
        if len(deg_vals) > 0:
            ax.scatter(deg_vals, counts, alpha=0.6, color=color, label=label, s=15)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Degree (log)', fontsize=12)
    ax.set_ylabel('Count (log)', fontsize=12)
    ax.set_title('Degree Distribution (Log-Log)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    filepath = os.path.join(save_dir, 'degree_distribution.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {filepath}")


def plot_feature_distributions(data, save_dir=None, top_k=8):
    """
    Box plots comparing key feature distributions between bot and human accounts.
    """
    if save_dir is None:
        save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    print("[Visualize] Plotting feature distributions...")
    
    features = data.x.numpy()
    labels = data.y.numpy()
    
    # Select interpretable features (first 21 = metadata)
    n_props = min(features.shape[1], len(MGTAB_FEATURE_NAMES))
    feature_indices = list(range(n_props))[:top_k]
    
    n_cols = 4
    n_rows = (len(feature_indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten()
    
    for idx, feat_idx in enumerate(feature_indices):
        ax = axes[idx]
        feat_name = MGTAB_FEATURE_NAMES.get(feat_idx, f'Feature {feat_idx}')
        
        human_vals = features[labels == 0, feat_idx]
        bot_vals = features[labels == 1, feat_idx]
        
        # Clip extreme values for better visualization
        clip_val = np.percentile(np.abs(features[:, feat_idx]), 99)
        human_vals = np.clip(human_vals, -clip_val, clip_val)
        bot_vals = np.clip(bot_vals, -clip_val, clip_val)
        
        bp = ax.boxplot([human_vals, bot_vals], 
                       labels=['Human', 'Bot'],
                       patch_artist=True,
                       widths=0.6)
        
        bp['boxes'][0].set_facecolor(config.VIS_CONFIG['color_human'])
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor(config.VIS_CONFIG['color_bot'])
        bp['boxes'][1].set_alpha(0.6)
        
        ax.set_title(feat_name, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide empty axes
    for idx in range(len(feature_indices), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Feature Distributions: Human vs Bot Accounts',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    filepath = os.path.join(save_dir, 'feature_distributions.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {filepath}")


def plot_attention_analysis(model, data, save_dir=None, sample_nodes=5):
    """
    Visualize GAT attention weights for sample bot and human nodes.
    Shows which neighbors each node attends to most.
    """
    if save_dir is None:
        save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    print("[Visualize] Analyzing attention weights...")
    
    device = config.DEVICE
    model = model.to(device)
    data_gpu = data.to(device)
    
    try:
        attn_weights = model.get_attention_weights(data_gpu.x, data_gpu.edge_index,
                                                    data_gpu.edge_type if hasattr(data_gpu, 'edge_type') else None)
    except Exception as e:
        print(f"  Warning: Could not extract attention weights: {e}")
        return
    
    if not attn_weights:
        print("  No attention weights available")
        return
    
    # Use last layer's attention
    alpha = attn_weights[-1].cpu().numpy()  # [E, num_heads]
    edge_index = data.edge_index.cpu().numpy()
    labels = data.y.cpu().numpy()
    
    # Average across heads
    if alpha.ndim > 1:
        alpha_avg = alpha.mean(axis=1)
    else:
        alpha_avg = alpha
    
    # Sample some human and bot nodes
    human_nodes = np.where(labels == 0)[0][:sample_nodes]
    bot_nodes = np.where(labels == 1)[0][:sample_nodes]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, nodes, title, color in [
        (axes[0], human_nodes, 'Human Accounts', config.VIS_CONFIG['color_human']),
        (axes[1], bot_nodes, 'Bot Accounts', config.VIS_CONFIG['color_bot']),
    ]:
        all_attn_values = []
        for node_idx in nodes:
            # Find edges where this node is the target
            edge_mask = edge_index[1] == node_idx
            node_attn = alpha_avg[edge_mask]
            if len(node_attn) > 0:
                all_attn_values.extend(node_attn.tolist())
        
        if all_attn_values:
            ax.hist(all_attn_values, bins=30, alpha=0.7, color=color, edgecolor='white')
            ax.set_title(f'Attention Distribution — {title}',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Attention Weight')
            ax.set_ylabel('Count')
            ax.axvline(np.mean(all_attn_values), color='black', linestyle='--',
                      label=f'Mean: {np.mean(all_attn_values):.4f}')
            ax.legend()
    
    plt.tight_layout()
    filepath = os.path.join(save_dir, 'attention_analysis.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {filepath}")


def generate_all_visualizations(data, models_dict, histories=None, save_dir=None):
    """
    Generate all visualizations in one call.
    
    Args:
        data: PyG Data object
        models_dict: Dict of {model_name: trained_model}
        histories: Dict of {model_name: training_history}
        save_dir: Output directory for figures
    """
    if save_dir is None:
        save_dir = config.FIGURES_DIR
    
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    # 1. Graph structure
    try:
        plot_graph_structure(data, save_dir=save_dir)
    except Exception as e:
        print(f"  ⚠️ Graph structure plot failed: {e}")
    
    # 2. Degree distribution
    try:
        plot_degree_distribution(data, save_dir=save_dir)
    except Exception as e:
        print(f"  ⚠️ Degree distribution plot failed: {e}")
    
    # 3. Feature distributions
    try:
        plot_feature_distributions(data, save_dir=save_dir)
    except Exception as e:
        print(f"  ⚠️ Feature distribution plot failed: {e}")
    
    # 4. Embedding visualizations for each GNN model
    for model_name, model in models_dict.items():
        try:
            plot_embeddings_tsne(model, data, model_name=model_name, save_dir=save_dir)
        except Exception as e:
            print(f"  ⚠️ t-SNE for {model_name} failed: {e}")
        
        try:
            plot_embeddings_pca(model, data, model_name=model_name, save_dir=save_dir)
        except Exception as e:
            print(f"  ⚠️ PCA for {model_name} failed: {e}")
    
    # 5. Attention analysis (GAT only)
    if 'GAT' in models_dict:
        try:
            plot_attention_analysis(models_dict['GAT'], data, save_dir=save_dir)
        except Exception as e:
            print(f"  ⚠️ Attention analysis failed: {e}")
    
    print(f"\n✅ All visualizations saved to {save_dir}")
