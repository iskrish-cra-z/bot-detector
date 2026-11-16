"""
Feature Engineering Module for Bot Detection.

Provides feature analysis and transformation utilities for the MGTAB dataset.
The MGTAB dataset comes with pre-computed features, but this module provides:
1. Feature analysis and importance visualization
2. Feature normalization
3. Feature description mapping

MGTAB Feature Vector Structure (per the paper):
The feature vector contains:
- Categorical properties (binary/categorical user metadata)
- Numerical properties (continuous user metadata like follower counts)
- Description embeddings (text embeddings of user bio)
- Tweet embeddings (text embeddings of user tweets)

Total feature dimension depends on dataset version.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif


# MGTAB feature column descriptions (first 21 columns based on BotRGCN paper)
# The remaining columns are typically text embeddings (768-dim from BERT/RoBERTa)
MGTAB_FEATURE_NAMES = {
    0: 'id_encoded',
    1: 'is_protected',
    2: 'is_verified',
    3: 'has_default_profile',
    4: 'followers_count',
    5: 'has_default_profile_image',
    6: 'friends_count',
    7: 'listed_count',
    8: 'favourites_count',
    9: 'has_geo_enabled',
    10: 'statuses_count',
    11: 'avg_tweet_length',
    12: 'avg_hashtags',
    13: 'avg_urls',
    14: 'avg_mentions',
    15: 'account_age_hours',
    16: 'has_description',
    17: 'has_url',
    18: 'has_location',
    19: 'username_length',
    20: 'screen_name_length',
}

# Feature categories for BotRGCN-style feature separation
CATEGORICAL_FEATURES = [1, 2, 3, 5, 9, 16, 17, 18, 19, 20]  # Binary/categorical
NUMERICAL_FEATURES = [4, 6, 7, 8, 10, 11, 12, 13, 14, 15]    # Continuous numerical
PROPERTY_FEATURES = list(range(21))                            # All metadata
EMBEDDING_START = 21                                            # Text embeddings start


def analyze_features(features, labels, top_k=20):
    """
    Analyze feature importance using mutual information.
    
    Args:
        features: Feature matrix [N × F] as numpy array or tensor
        labels: Label vector [N] as numpy array or tensor
        top_k: Number of top features to return
        
    Returns:
        Dict with feature importance scores and analysis
    """
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    # Replace NaN/Inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    num_features = features.shape[1]
    print(f"[FeatureEng] Analyzing {num_features} features...")
    
    # Compute mutual information with labels (only for metadata features if too many)
    n_features_for_mi = min(num_features, 100)  # Limit for speed
    mi_scores = mutual_info_classif(
        features[:, :n_features_for_mi], 
        labels, 
        random_state=42
    )
    
    # Get feature names
    feature_names = []
    for i in range(n_features_for_mi):
        if i in MGTAB_FEATURE_NAMES:
            feature_names.append(MGTAB_FEATURE_NAMES[i])
        elif i >= EMBEDDING_START:
            feature_names.append(f'embedding_{i - EMBEDDING_START}')
        else:
            feature_names.append(f'feature_{i}')
    
    # Sort by importance
    sorted_indices = np.argsort(mi_scores)[::-1]
    
    top_features = []
    for rank, idx in enumerate(sorted_indices[:top_k]):
        top_features.append({
            'rank': rank + 1,
            'index': int(idx),
            'name': feature_names[idx],
            'mi_score': float(mi_scores[idx]),
        })
    
    # Compute per-class statistics for metadata features
    class_stats = {}
    unique_labels = np.unique(labels)
    for cls in unique_labels:
        cls_mask = labels == cls
        cls_name = 'human' if cls == 0 else 'bot'
        stats = {}
        for i in range(min(num_features, len(MGTAB_FEATURE_NAMES))):
            feat_name = MGTAB_FEATURE_NAMES.get(i, f'feature_{i}')
            feat_vals = features[cls_mask, i]
            stats[feat_name] = {
                'mean': float(np.mean(feat_vals)),
                'std': float(np.std(feat_vals)),
                'min': float(np.min(feat_vals)),
                'max': float(np.max(feat_vals)),
            }
        class_stats[cls_name] = stats
    
    return {
        'num_features': num_features,
        'top_features': top_features,
        'mi_scores': mi_scores.tolist(),
        'class_stats': class_stats,
        'feature_names': feature_names,
    }


def normalize_features(features, method='standard'):
    """
    Normalize feature matrix.
    
    Args:
        features: Feature matrix [N × F] as tensor
        method: 'standard' (zero mean, unit var) or 'minmax' (0-1 range)
        
    Returns:
        Normalized features as tensor, fitted scaler
    """
    if isinstance(features, torch.Tensor):
        x = features.numpy()
    else:
        x = features.copy()
    
    # Replace NaN/Inf
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    x_normalized = scaler.fit_transform(x)
    
    return torch.tensor(x_normalized, dtype=torch.float32), scaler


def get_property_features(features):
    """
    Extract only the user metadata/property features (first 21 columns).
    Useful for baseline ML models that don't need text embeddings.
    
    Args:
        features: Full feature matrix [N × F] as tensor
        
    Returns:
        Property features [N × 21] as tensor
    """
    return features[:, :len(MGTAB_FEATURE_NAMES)]


def compute_derived_features(features):
    """
    Compute additional hand-crafted features from the raw MGTAB properties.
    
    Args:
        features: Feature matrix as tensor (at least 21 columns)
        
    Returns:
        Extended feature matrix with derived features appended
    """
    f = features.numpy() if isinstance(features, torch.Tensor) else features.copy()
    
    derived = []
    
    # Followers-to-friends ratio (avoid division by zero)
    friends = np.maximum(f[:, 6], 1)  # friends_count
    followers = f[:, 4]  # followers_count
    derived.append((followers / friends).reshape(-1, 1))
    
    # Tweets per hour of account age
    age_hours = np.maximum(f[:, 15], 1)  # account_age_hours
    derived.append((f[:, 10] / age_hours).reshape(-1, 1))  # statuses_count / age
    
    # Engagement ratio (favourites / statuses)
    statuses = np.maximum(f[:, 10], 1)
    derived.append((f[:, 8] / statuses).reshape(-1, 1))  # favourites / statuses
    
    # Listed-to-followers ratio
    followers_safe = np.maximum(f[:, 4], 1)
    derived.append((f[:, 7] / followers_safe).reshape(-1, 1))  # listed / followers
    
    # Stack derived features
    derived_array = np.hstack(derived)
    derived_array = np.nan_to_num(derived_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Append to original features
    extended = np.hstack([f, derived_array])
    
    return torch.tensor(extended, dtype=torch.float32)


if __name__ == "__main__":
    """Test feature engineering."""
    # Create dummy data for testing
    N = 100
    F = 789  # 21 metadata + 768 embeddings
    features = torch.randn(N, F)
    labels = torch.randint(0, 2, (N,))
    
    # Test analysis
    analysis = analyze_features(features, labels, top_k=10)
    print(f"\nTop features by mutual information:")
    for feat in analysis['top_features']:
        print(f"  #{feat['rank']}: {feat['name']} (MI={feat['mi_score']:.4f})")
    
    # Test normalization
    norm_features, scaler = normalize_features(features)
    print(f"\nNormalized features shape: {norm_features.shape}")
    print(f"Mean: {norm_features.mean():.6f}, Std: {norm_features.std():.6f}")
    
    # Test derived features
    extended = compute_derived_features(features)
    print(f"\nExtended features shape: {extended.shape} (original: {features.shape})")
