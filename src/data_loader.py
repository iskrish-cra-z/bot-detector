"""
MGTAB Dataset Loader for Bot Detection using Graph Attention Networks.

Loads the MGTAB dataset which contains:
- 10,199 expert-annotated Twitter users
- 7 types of relationships (followers, friends, mention, reply, quoted, URL, hashtag)
- Bot labels (human=0, bot=1) and stance labels
- Pre-computed node feature vectors

The dataset files are PyTorch tensors (.pt):
- features.pt:      Node features [N × F]
- edge_index.pt:    Edge indices [2 × E]
- edge_type.pt:     Edge type labels [E]
- edge_weight.pt:   Edge weights [E]
- labels_bot.pt:    Bot detection labels [N]
- labels_stance.pt: Stance detection labels [N]
"""

import os
import sys
import torch
import numpy as np
from sklearn.utils import shuffle as sklearn_shuffle
from torch_geometric.data import Data

# Add parent dir to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def sample_mask(idx, num_nodes):
    """Create a boolean mask tensor from indices.
    
    Args:
        idx: Indices to set to True
        num_nodes: Total number of nodes
        
    Returns:
        Boolean mask tensor of shape [num_nodes]
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = True
    return mask


def load_mgtab(data_dir=None, task='bot', relation_select=None, seed=42):
    """
    Load the MGTAB dataset and construct a PyG Data object.
    
    The MGTAB dataset should be downloaded from Google Drive and extracted
    into the data/MGTAB directory. Expected files:
    - features.pt, edge_index.pt, edge_type.pt, edge_weight.pt
    - labels_bot.pt, labels_stance.pt
    
    Args:
        data_dir: Path to MGTAB directory containing .pt files
        task: 'bot' for bot detection, 'stance' for stance detection
        relation_select: List of relation indices to use (0-6).
                        Default uses config.RELATION_SELECT
        seed: Random seed for train/val/test split
        
    Returns:
        data: PyG Data object with x, edge_index, y, masks, and metadata
        info: Dict with dataset statistics
    """
    if data_dir is None:
        data_dir = config.MGTAB_DIR
    
    if relation_select is None:
        relation_select = config.RELATION_SELECT
    
    # ================================================================
    # Step 1: Load raw tensor files
    # ================================================================
    print(f"[DataLoader] Loading MGTAB dataset from: {data_dir}")
    
    required_files = ['features.pt', 'edge_index.pt', 'edge_type.pt', 
                      'edge_weight.pt', 'labels_bot.pt']
    
    for f in required_files:
        filepath = os.path.join(data_dir, f)
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Required file not found: {filepath}\n"
                f"Please download the MGTAB dataset from:\n"
                f"  https://drive.google.com/uc?export=download&id=1gbWNOoU1JB8RrTu2a5j9KMNVa9wX72Fe\n"
                f"And extract it into: {data_dir}"
            )
    
    # Load tensors (using weights_only=False for compatibility with older torch saves)
    features = torch.load(os.path.join(data_dir, "features.pt"), 
                         map_location='cpu', weights_only=False)
    features = features.to(torch.float32)
    
    edge_index = torch.load(os.path.join(data_dir, "edge_index.pt"), 
                           map_location='cpu', weights_only=False)
    edge_index = torch.tensor(edge_index, dtype=torch.int64) if not isinstance(edge_index, torch.Tensor) else edge_index.to(torch.int64)
    
    edge_type = torch.load(os.path.join(data_dir, "edge_type.pt"), 
                          map_location='cpu', weights_only=False)
    
    edge_weight = torch.load(os.path.join(data_dir, "edge_weight.pt"), 
                            map_location='cpu', weights_only=False)
    
    bot_label = torch.load(os.path.join(data_dir, "labels_bot.pt"), 
                          map_location='cpu', weights_only=False)
    
    # Load stance labels if available
    stance_label = None
    stance_path = os.path.join(data_dir, "labels_stance.pt")
    if os.path.exists(stance_path):
        stance_label = torch.load(stance_path, map_location='cpu', weights_only=False)
    
    print(f"  Features shape: {features.shape}")
    print(f"  Edge index shape: {edge_index.shape}")
    print(f"  Edge types: {edge_type.unique().tolist()}")
    print(f"  Bot labels distribution: {torch.bincount(bot_label.long()).tolist()}")
    
    # ================================================================
    # Step 2: Select relations (filter edges by type)
    # ================================================================
    print(f"  Relations selected: {[config.RELATION_DICT[r] for r in relation_select]}")
    
    # Build selection mask for edges
    edge_mask = torch.zeros(edge_type.shape[0], dtype=torch.bool)
    for rel_idx in relation_select:
        edge_mask = edge_mask | (edge_type == rel_idx)
    
    filtered_edge_index = edge_index[:, edge_mask]
    filtered_edge_type = edge_type[edge_mask]
    filtered_edge_weight = edge_weight[edge_mask]
    
    # Re-map edge types to consecutive indices (0, 1, 2, ...)
    type_remap = {old_type: new_type for new_type, old_type in enumerate(sorted(relation_select))}
    remapped_edge_type = torch.zeros_like(filtered_edge_type)
    for old_t, new_t in type_remap.items():
        remapped_edge_type[filtered_edge_type == old_t] = new_t
    
    print(f"  Filtered edges: {filtered_edge_index.shape[1]} "
          f"(from {edge_index.shape[1]} total)")
    
    # ================================================================
    # Step 3: Select labels based on task
    # ================================================================
    if task == 'bot':
        labels = bot_label.long()
        num_classes = 2
    elif task == 'stance':
        if stance_label is None:
            raise ValueError("Stance labels not found in dataset")
        labels = stance_label.long()
        num_classes = 3
    else:
        raise ValueError(f"Unknown task: {task}. Choose 'bot' or 'stance'.")
    
    # ================================================================
    # Step 4: Create train/val/test masks with stratified splitting
    # ================================================================
    num_nodes = len(labels)
    shuffled_idx = sklearn_shuffle(np.arange(num_nodes), random_state=seed)
    
    train_end = int(config.TRAIN_RATIO * num_nodes)
    val_end = int((config.TRAIN_RATIO + config.VAL_RATIO) * num_nodes)
    
    train_idx = shuffled_idx[:train_end]
    val_idx = shuffled_idx[train_end:val_end]
    test_idx = shuffled_idx[val_end:]
    
    train_mask = sample_mask(train_idx, num_nodes)
    val_mask = sample_mask(val_idx, num_nodes)
    test_mask = sample_mask(test_idx, num_nodes)
    
    print(f"  Split: train={train_mask.sum().item()}, "
          f"val={val_mask.sum().item()}, test={test_mask.sum().item()}")
    
    # ================================================================
    # Step 5: Construct PyG Data object
    # ================================================================
    data = Data(
        x=features,
        edge_index=filtered_edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    data.edge_type = remapped_edge_type
    data.edge_weight = filtered_edge_weight
    data.num_classes = num_classes
    data.num_relations = len(relation_select)
    
    # ================================================================
    # Step 6: Compute dataset statistics
    # ================================================================
    info = {
        'dataset': 'MGTAB',
        'task': task,
        'num_nodes': num_nodes,
        'num_edges': filtered_edge_index.shape[1],
        'num_features': features.shape[1],
        'num_classes': num_classes,
        'num_relations': len(relation_select),
        'relations_used': [config.RELATION_DICT[r] for r in relation_select],
        'label_counts': torch.bincount(labels).tolist(),
        'train_count': train_mask.sum().item(),
        'val_count': val_mask.sum().item(),
        'test_count': test_mask.sum().item(),
        'avg_degree': filtered_edge_index.shape[1] / num_nodes,
        'feature_dim': features.shape[1],
    }
    
    # Class balance info
    for cls_idx in range(num_classes):
        cls_name = 'human' if cls_idx == 0 else 'bot'
        info[f'class_{cls_name}_count'] = (labels == cls_idx).sum().item()
        info[f'class_{cls_name}_ratio'] = (labels == cls_idx).float().mean().item()
    
    print(f"\n[DataLoader] Dataset Summary:")
    print(f"  Nodes: {info['num_nodes']}, Edges: {info['num_edges']}")
    print(f"  Features: {info['num_features']}, Classes: {info['num_classes']}")
    print(f"  Human: {info.get('class_human_count', 'N/A')}, "
          f"Bot: {info.get('class_bot_count', 'N/A')}")
    print(f"  Avg degree: {info['avg_degree']:.2f}")
    
    return data, info


def get_feature_vectors(data):
    """
    Extract feature vectors for traditional ML models (no graph structure).
    
    Args:
        data: PyG Data object
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test as numpy arrays
    """
    x = data.x.numpy()
    y = data.y.numpy()
    
    X_train = x[data.train_mask.numpy()]
    y_train = y[data.train_mask.numpy()]
    X_val = x[data.val_mask.numpy()]
    y_val = y[data.val_mask.numpy()]
    X_test = x[data.test_mask.numpy()]
    y_test = y[data.test_mask.numpy()]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    """Test data loading."""
    try:
        data, info = load_mgtab(task='bot', seed=42)
        print(f"\nData object: {data}")
        print(f"\nInfo: {info}")
        
        X_train, X_val, X_test, y_train, y_val, y_test = get_feature_vectors(data)
        print(f"\nFeature vectors:")
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("\nTo download the MGTAB dataset:")
        print("1. Visit: https://drive.google.com/uc?export=download&id=1gbWNOoU1JB8RrTu2a5j9KMNVa9wX72Fe")
        print(f"2. Extract the zip into: {config.MGTAB_DIR}")
