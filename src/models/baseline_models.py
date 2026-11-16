"""
Baseline Machine Learning Models for Bot Detection.

These models use ONLY the node feature vectors — they do NOT use graph structure.
This serves as a crucial comparison point to demonstrate the value of graph-based
learning (GAT/GCN/GraphSAGE).

Models included:
1. Logistic Regression — linear baseline
2. Random Forest — ensemble tree-based model
3. SVM (RBF kernel) — non-linear classifier
4. AdaBoost — boosting ensemble
5. KNN — instance-based learning

Feature importance from Random Forest is used to interpret which features
are most discriminative for bot detection.
"""

import numpy as np
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class BaselineModels:
    """
    Container for all baseline ML models.
    
    Usage:
        baselines = BaselineModels()
        results = baselines.train_and_evaluate(X_train, X_test, y_train, y_test)
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.fitted = False
    
    def _create_models(self):
        """Create all baseline model instances."""
        return {
            'Logistic Regression': LogisticRegression(
                C=config.LR_CONFIG['C'],
                max_iter=config.LR_CONFIG['max_iter'],
                solver=config.LR_CONFIG.get('solver', 'lbfgs'),
                random_state=self.seed,
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=config.RF_CONFIG['n_estimators'],
                max_depth=config.RF_CONFIG.get('max_depth', None),
                n_jobs=config.RF_CONFIG.get('n_jobs', -1),
                random_state=self.seed,
            ),
            'SVM': SVC(
                kernel=config.SVM_CONFIG['kernel'],
                C=config.SVM_CONFIG['C'],
                probability=config.SVM_CONFIG.get('probability', True),
                random_state=self.seed,
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=50,
                learning_rate=1.0,
                random_state=self.seed,
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=7,
            ),
        }
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, 
                           X_val=None, y_val=None, models_to_run=None):
        """
        Train all baseline models and evaluate on test set.
        
        Args:
            X_train, X_test: Feature matrices (numpy arrays)
            y_train, y_test: Label vectors
            X_val, y_val: Optional validation set
            models_to_run: Optional list of model names to run (default: all)
            
        Returns:
            Dict of results per model
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        all_models = self._create_models()
        
        if models_to_run:
            all_models = {k: v for k, v in all_models.items() if k in models_to_run}
        
        self.results = {}
        self.models = {}
        
        for name, model in all_models.items():
            print(f"\n{'='*50}")
            print(f"Training: {name}")
            print(f"{'='*50}")
            
            # Train
            model.fit(X_train_scaled, y_train)
            self.models[name] = model
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Get probabilities (for ROC-AUC)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_scaled)
                if y_proba.shape[1] == 2:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            else:
                y_proba = None
                roc_auc = None
            
            # Compute metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='macro', zero_division=0),
                'roc_auc': roc_auc,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, 
                                                                target_names=['human', 'bot'],
                                                                output_dict=True),
                'y_pred': y_pred,
                'y_proba': y_proba,
            }
            
            # Validation metrics
            if X_val is not None:
                y_val_pred = model.predict(X_val_scaled)
                metrics['val_accuracy'] = accuracy_score(y_val, y_val_pred)
                metrics['val_f1'] = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
            
            self.results[name] = metrics
            
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            if roc_auc:
                print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        self.fitted = True
        return self.results
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance from Random Forest.
        
        Returns:
            Dict of {feature_name: importance_score}, sorted by importance
        """
        if 'Random Forest' not in self.models:
            return None
        
        rf = self.models['Random Forest']
        importances = rf.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        result = []
        for rank, idx in enumerate(indices[:30]):  # Top 30 features
            result.append({
                'rank': rank + 1,
                'index': int(idx),
                'name': feature_names[idx] if idx < len(feature_names) else f'feature_{idx}',
                'importance': float(importances[idx]),
            })
        
        return result
    
    def get_comparison_table(self):
        """
        Generate a comparison table of all model results.
        
        Returns:
            List of dicts suitable for tabulate/pandas
        """
        rows = []
        for name, metrics in self.results.items():
            rows.append({
                'Model': name,
                'Type': 'Feature-based',
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else 'N/A',
            })
        return rows
    
    def save_results(self, save_dir=None):
        """Save all results to JSON."""
        if save_dir is None:
            save_dir = config.METRICS_DIR
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for name, metrics in self.results.items():
            serializable_results[name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
                if k not in ['y_pred', 'y_proba']  # Skip large arrays
            }
        
        filepath = os.path.join(save_dir, 'baseline_results.json')
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"[Baselines] Results saved to {filepath}")


if __name__ == "__main__":
    """Test baseline models with dummy data."""
    np.random.seed(42)
    
    N_train, N_test = 500, 200
    F = 789
    
    X_train = np.random.randn(N_train, F)
    y_train = np.random.randint(0, 2, N_train)
    X_test = np.random.randn(N_test, F)
    y_test = np.random.randint(0, 2, N_test)
    
    baselines = BaselineModels(seed=42)
    results = baselines.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    for row in baselines.get_comparison_table():
        print(f"  {row['Model']:25s} Acc={row['Accuracy']}  F1={row['F1-Score']}")
    
    importance = baselines.get_feature_importance()
    if importance:
        print("\nTop 5 features (Random Forest):")
        for feat in importance[:5]:
            print(f"  #{feat['rank']}: {feat['name']} ({feat['importance']:.4f})")
