"""
Configuration file for Bot Detection using Graph Attention Networks.
All hyperparameters, paths, and settings are centralized here.
"""

import os
import torch

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MGTAB_DIR = os.path.join(DATA_DIR, "MGTAB")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

# Create directories if they don't exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MGTAB_DIR,
          RESULTS_DIR, FIGURES_DIR, MODELS_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# DEVICE CONFIGURATION
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Config] Using device: {DEVICE}")

# ============================================================
# DATASET CONFIGURATION (MGTAB)
# ============================================================
DATASET_NAME = "MGTAB"
TASK = "bot"  # 'bot' for bot detection, 'stance' for stance detection
NUM_CLASSES = 2  # bot detection: 2 (human=0, bot=1)

# MGTAB has 7 relation types:
# 0: followers, 1: friends, 2: mention, 3: reply, 4: quoted, 5: url, 6: hashtag
RELATION_DICT = {
    0: 'followers',
    1: 'friends',
    2: 'mention',
    3: 'reply',
    4: 'quoted',
    5: 'url',
    6: 'hashtag'
}
# Which relations to use for graph construction
RELATION_SELECT = [0, 1]  # Default: followers + friends (best results per paper)

# ============================================================
# DATA SPLIT
# ============================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1  # 0.7 to 0.8 (original uses 0.7 to 0.9 for val, we split finer)
TEST_RATIO = 0.2  # remaining 0.8 to 1.0

# ============================================================
# GAT MODEL HYPERPARAMETERS
# ============================================================
GAT_CONFIG = {
    'hidden_dim': 256,
    'num_heads': 8,          # Multi-head attention
    'num_layers': 2,         # Number of GAT layers
    'dropout': 0.3,
    'use_gatv2': True,       # Use GATv2Conv (fixes static attention)
    'residual': True,        # Residual connections
    'layer_norm': True,      # Layer normalization
}

# ============================================================
# GCN MODEL HYPERPARAMETERS
# ============================================================
GCN_CONFIG = {
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.3,
}

# ============================================================
# GraphSAGE MODEL HYPERPARAMETERS
# ============================================================
SAGE_CONFIG = {
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.3,
    'aggr': 'mean',  # Aggregation: 'mean', 'max', 'lstm'
}

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
TRAINING_CONFIG = {
    'epochs': 200,
    'lr': 1e-3,
    'weight_decay': 5e-4,
    'patience': 30,          # Early stopping patience
    'min_delta': 1e-4,       # Minimum improvement for early stopping
    'scheduler_patience': 10, # LR scheduler patience
    'scheduler_factor': 0.5,  # LR reduction factor
}

# ============================================================
# BASELINE MODEL HYPERPARAMETERS
# ============================================================
LR_CONFIG = {
    'C': 0.1,
    'max_iter': 500,
    'solver': 'lbfgs',
}

RF_CONFIG = {
    'n_estimators': 100,
    'n_jobs': -1,
    'max_depth': None,
}

SVM_CONFIG = {
    'kernel': 'rbf',
    'C': 10,
    'probability': True,
}

# ============================================================
# RANDOM SEEDS (for reproducibility across runs)
# ============================================================
RANDOM_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_SEED = 42

# ============================================================
# VISUALIZATION
# ============================================================
VIS_CONFIG = {
    'max_nodes_display': 500,   # Max nodes in graph visualization
    'tsne_perplexity': 30,
    'tsne_n_iter': 1000,
    'figsize': (12, 8),
    'dpi': 150,
    'font_size': 12,
    'color_human': '#2ecc71',   # Green
    'color_bot': '#e74c3c',     # Red
    'color_palette': 'viridis',
}
