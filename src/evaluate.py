"""
Model Evaluation Module.

Computes test-set metrics for all models and generates:
- Confusion matrices
- ROC curves
- Classification reports
- Side-by-side comparison tables
- Statistical analysis across multiple seeds
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc,
    confusion_matrix, classification_report
)
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def evaluate_gnn(model, data, model_name='GAT'):
    """
    Evaluate a trained GNN model on the test set.
    
    Args:
        model: Trained PyTorch model
        data: PyG Data object
        model_name: Name for logging
        
    Returns:
        Dict with all evaluation metrics
    """
    device = config.DEVICE
    model = model.to(device)
    data = data.to(device)
    
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index,
                    data.edge_type if hasattr(data, 'edge_type') else None)
        
        probs = torch.softmax(out, dim=1)
    
    # Test set predictions
    test_pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
    test_true = data.y[data.test_mask].cpu().numpy()
    test_probs = probs[data.test_mask].cpu().numpy()
    
    # Compute metrics
    metrics = {
        'model_name': model_name,
        'model_type': 'Graph-based',
        'accuracy': accuracy_score(test_true, test_pred),
        'precision': precision_score(test_true, test_pred, average='macro', zero_division=0),
        'recall': recall_score(test_true, test_pred, average='macro', zero_division=0),
        'f1_score': f1_score(test_true, test_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(test_true, test_pred).tolist(),
        'classification_report': classification_report(
            test_true, test_pred,
            target_names=['human', 'bot'],
            output_dict=True
        ),
        'y_true': test_true,
        'y_pred': test_pred,
        'y_proba': test_probs,
    }
    
    # ROC-AUC
    if test_probs.shape[1] == 2:
        metrics['roc_auc'] = roc_auc_score(test_true, test_probs[:, 1])
        fpr, tpr, _ = roc_curve(test_true, test_probs[:, 1])
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
    
    print(f"\n{'='*50}")
    print(f"Test Results: {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def compare_all_models(gnn_results, baseline_results=None):
    """
    Create a comprehensive comparison table of all models.
    
    Args:
        gnn_results: Dict of {model_name: metrics} for GNN models
        baseline_results: Dict of {model_name: metrics} for baseline models
        
    Returns:
        comparison_table: List of dicts
        summary_text: Formatted string
    """
    rows = []
    
    # Add GNN results
    for name, metrics in gnn_results.items():
        rows.append({
            'Model': name,
            'Type': 'Graph-based',
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics.get('roc_auc', None),
        })
    
    # Add baseline results
    if baseline_results:
        for name, metrics in baseline_results.items():
            rows.append({
                'Model': name,
                'Type': 'Feature-based',
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics.get('roc_auc', None),
            })
    
    # Sort by F1-Score descending
    rows.sort(key=lambda x: x['F1-Score'], reverse=True)
    
    # Format for display
    display_rows = []
    for row in rows:
        display_rows.append({
            'Model': row['Model'],
            'Type': row['Type'],
            'Accuracy': f"{row['Accuracy']:.4f}",
            'Precision': f"{row['Precision']:.4f}",
            'Recall': f"{row['Recall']:.4f}",
            'F1-Score': f"{row['F1-Score']:.4f}",
            'ROC-AUC': f"{row['ROC-AUC']:.4f}" if row['ROC-AUC'] else 'N/A',
        })
    
    # Print table
    print(f"\n{'='*85}")
    print("MODEL COMPARISON (sorted by F1-Score)")
    print(f"{'='*85}")
    print(tabulate(display_rows, headers='keys', tablefmt='grid'))
    
    return rows, display_rows


def plot_confusion_matrices(all_results, save_dir=None):
    """
    Plot confusion matrices for all models in a grid.
    """
    if save_dir is None:
        save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    n_models = len(all_results)
    cols = min(3, n_models)
    rows_count = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows_count, cols, figsize=(5*cols, 4*rows_count))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (name, metrics) in enumerate(all_results.items()):
        ax = axes[idx]
        cm = np.array(metrics['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Human', 'Bot'],
                   yticklabels=['Human', 'Bot'])
        ax.set_title(f'{name}\n(Acc: {metrics["accuracy"]:.3f}, F1: {metrics["f1_score"]:.3f})',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    # Hide empty axes
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    filepath = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Evaluate] Confusion matrices saved to {filepath}")


def plot_roc_curves(all_results, save_dir=None):
    """
    Plot ROC curves for all models on a single plot.
    """
    if save_dir is None:
        save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))
    
    for (name, metrics), color in zip(all_results.items(), colors):
        if 'fpr' in metrics and 'tpr' in metrics:
            roc_auc_val = metrics.get('roc_auc', auc(metrics['fpr'], metrics['tpr']))
            ax.plot(metrics['fpr'], metrics['tpr'],
                   label=f'{name} (AUC = {roc_auc_val:.3f})',
                   color=color, linewidth=2)
    
    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Bot Detection Model Comparison',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    filepath = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Evaluate] ROC curves saved to {filepath}")


def plot_training_curves(histories, save_dir=None):
    """
    Plot training & validation loss/accuracy/F1 curves for GNN models.
    
    Args:
        histories: Dict of {model_name: history_dict}
    """
    if save_dir is None:
        save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics_to_plot = [
        ('train_loss', 'val_loss', 'Loss', 'Loss'),
        ('train_acc', 'val_acc', 'Accuracy', 'Accuracy'),
        ('train_f1', 'val_f1', 'F1-Score', 'F1-Score'),
    ]
    
    colors = {'GAT': '#e74c3c', 'GCN': '#3498db', 'GraphSAGE': '#2ecc71'}
    
    for ax, (train_key, val_key, ylabel, title) in zip(axes, metrics_to_plot):
        for model_name, history in histories.items():
            color = colors.get(model_name, '#95a5a6')
            epochs = range(1, len(history[train_key]) + 1)
            
            ax.plot(epochs, history[train_key], '--', color=color, alpha=0.5,
                   label=f'{model_name} (train)')
            ax.plot(epochs, history[val_key], '-', color=color, linewidth=2,
                   label=f'{model_name} (val)')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Training {title}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Evaluate] Training curves saved to {filepath}")


def plot_f1_comparison_bar(all_results, save_dir=None):
    """
    Bar chart comparing F1-scores across all models.
    """
    if save_dir is None:
        save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    models = list(all_results.keys())
    f1_scores = [all_results[m]['f1_score'] for m in models]
    accuracies = [all_results[m]['accuracy'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy',
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score',
                   color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    filepath = os.path.join(save_dir, 'model_comparison_bar.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Evaluate] Comparison bar chart saved to {filepath}")


def save_all_results(all_results, save_dir=None):
    """Save all evaluation results to JSON."""
    if save_dir is None:
        save_dir = config.METRICS_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    serializable = {}
    for name, metrics in all_results.items():
        serializable[name] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metrics.items()
            if k not in ['y_true', 'y_pred', 'y_proba', 'fpr', 'tpr']
        }
    
    filepath = os.path.join(save_dir, 'all_results.json')
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    
    print(f"[Evaluate] All results saved to {filepath}")
