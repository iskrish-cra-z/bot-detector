"""
Training Pipeline for GNN Models (GAT, GCN, GraphSAGE).

Implements:
- Full training loop with validation
- Early stopping with patience
- Learning rate scheduling
- Model checkpointing (best validation F1)
- Training history logging
- Weight initialization
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.models.gat_model import create_gat_model
from src.models.gcn_model import create_gcn_model
from src.models.graphsage_model import create_sage_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=30, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value):
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def train_gnn(data, model_name='GAT', seed=42, verbose=True):
    """
    Train a GNN model (GAT, GCN, or GraphSAGE) on the MGTAB dataset.
    
    Args:
        data: PyG Data object with x, edge_index, y, masks
        model_name: 'GAT', 'GCN', or 'GraphSAGE'
        seed: Random seed
        verbose: Print training progress
        
    Returns:
        model: Trained model
        history: Dict with training history
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    device = config.DEVICE
    tcfg = config.TRAINING_CONFIG
    
    in_channels = data.x.shape[1]
    num_classes = data.num_classes
    
    # ================================================================
    # Create model
    # ================================================================
    if model_name.upper() == 'GAT':
        model = create_gat_model(in_channels, num_classes)
    elif model_name.upper() == 'GCN':
        model = create_gcn_model(in_channels, num_classes)
    elif model_name.upper() in ['GRAPHSAGE', 'SAGE']:
        model = create_sage_model(in_channels, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose GAT, GCN, or GraphSAGE")
    
    model = model.to(device)
    data = data.to(device)
    
    # ================================================================
    # Optimizer, loss, scheduler
    # ================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg['lr'],
        weight_decay=tcfg['weight_decay'],
    )
    
    criterion = nn.CrossEntropyLoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=tcfg['scheduler_factor'],
        patience=tcfg['scheduler_patience'],
        verbose=verbose,
    )
    
    early_stopping = EarlyStopping(
        patience=tcfg['patience'],
        min_delta=tcfg['min_delta'],
        mode='max',
    )
    
    # ================================================================
    # Training loop
    # ================================================================
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'lr': [],
    }
    
    best_val_f1 = 0
    best_model_state = None
    best_epoch = 0
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} | Device: {device}")
    print(f"Epochs: {tcfg['epochs']}, LR: {tcfg['lr']}, "
          f"Patience: {tcfg['patience']}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for epoch in range(tcfg['epochs']):
        # --- Train step ---
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, 
                     data.edge_type if hasattr(data, 'edge_type') else None)
        
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # --- Compute metrics ---
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index,
                       data.edge_type if hasattr(data, 'edge_type') else None)
            
            # Training metrics
            train_pred = out[data.train_mask].argmax(dim=1).cpu().numpy()
            train_true = data.y[data.train_mask].cpu().numpy()
            train_loss = criterion(out[data.train_mask], data.y[data.train_mask]).item()
            train_acc = accuracy_score(train_true, train_pred)
            train_f1 = f1_score(train_true, train_pred, average='macro', zero_division=0)
            train_prec = precision_score(train_true, train_pred, average='macro', zero_division=0)
            train_rec = recall_score(train_true, train_pred, average='macro', zero_division=0)
            
            # Validation metrics
            val_pred = out[data.val_mask].argmax(dim=1).cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask]).item()
            val_acc = accuracy_score(val_true, val_pred)
            val_f1 = f1_score(val_true, val_pred, average='macro', zero_division=0)
            val_prec = precision_score(val_true, val_pred, average='macro', zero_division=0)
            val_rec = recall_score(val_true, val_pred, average='macro', zero_division=0)
        
        # Update history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_precision'].append(train_prec)
        history['val_precision'].append(val_prec)
        history['train_recall'].append(train_rec)
        history['val_recall'].append(val_rec)
        history['lr'].append(current_lr)
        
        # Update scheduler
        scheduler.step(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{tcfg['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"LR: {current_lr:.6f}")
        
        # Early stopping
        if early_stopping(val_f1):
            print(f"\n  ⚡ Early stopping at epoch {epoch+1} "
                  f"(best F1: {best_val_f1:.4f} at epoch {best_epoch})")
            break
    
    training_time = time.time() - start_time
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)
    
    # Save model checkpoint
    checkpoint_path = os.path.join(config.MODELS_DIR, f'{model_name.lower()}_best.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'in_channels': in_channels,
        'num_classes': num_classes,
        'config': getattr(config, f'{model_name.upper()}_CONFIG', {}),
    }, checkpoint_path)
    print(f"\n  ✅ Best model saved to {checkpoint_path}")
    
    # Add metadata to history
    history['best_epoch'] = best_epoch
    history['best_val_f1'] = best_val_f1
    history['training_time'] = training_time
    history['model_name'] = model_name
    history['total_epochs'] = epoch + 1
    
    print(f"\n{'='*60}")
    print(f"Training Complete: {model_name}")
    print(f"  Best epoch: {best_epoch}, Best val F1: {best_val_f1:.4f}")
    print(f"  Training time: {training_time:.1f}s")
    print(f"{'='*60}")
    
    return model, history


def train_all_gnns(data, seeds=None):
    """
    Train all GNN models (GAT, GCN, GraphSAGE).
    
    Args:
        data: PyG Data object
        seeds: List of random seeds (default: uses first seed from config)
        
    Returns:
        Dict of {model_name: (model, history)}
    """
    if seeds is None:
        seeds = [config.DEFAULT_SEED]
    
    results = {}
    
    for model_name in ['GAT', 'GCN', 'GraphSAGE']:
        all_histories = []
        best_model = None
        best_f1 = 0
        
        for seed in seeds:
            model, history = train_gnn(data, model_name=model_name, seed=seed)
            all_histories.append(history)
            
            if history['best_val_f1'] > best_f1:
                best_f1 = history['best_val_f1']
                best_model = model
        
        # Use the best model and first history for visualization
        results[model_name] = (best_model, all_histories[0])
    
    return results


if __name__ == "__main__":
    """Test training with dummy data."""
    from torch_geometric.data import Data
    
    N, F, E = 200, 16, 1000
    x = torch.randn(N, F)
    edge_index = torch.randint(0, N, (2, E))
    y = torch.randint(0, 2, (N,))
    
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[:int(0.7*N)] = True
    val_mask = torch.zeros(N, dtype=torch.bool)
    val_mask[int(0.7*N):int(0.9*N)] = True
    test_mask = torch.zeros(N, dtype=torch.bool)
    test_mask[int(0.9*N):] = True
    
    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.num_classes = 2
    data.edge_type = torch.randint(0, 2, (E,))
    
    model, history = train_gnn(data, model_name='GAT', seed=42)
    print(f"\nFinal model: {type(model).__name__}")
    print(f"Best val F1: {history['best_val_f1']:.4f}")
