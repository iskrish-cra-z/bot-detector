"""
Main Entry Point — Bot Detection using Graph Attention Networks (GAT)

This is the master script that orchestrates the entire pipeline:
1. Load MGTAB dataset
2. Analyze graph structure
3. Train baseline ML models (Logistic Regression, Random Forest, SVM, etc.)
4. Train GNN models (GAT, GCN, GraphSAGE)
5. Evaluate all models on test set
6. Generate visualizations (graph, t-SNE, ROC curves, confusion matrices)
7. Save all results

Usage:
    python main.py                          # Run full pipeline
    python main.py --mode train             # Only train
    python main.py --mode evaluate          # Only evaluate (load saved models)
    python main.py --mode visualize         # Only generate visualizations
    python main.py --model GAT              # Train only GAT
    python main.py --model all              # Train all GNN models
    python main.py --relations 0 1 2        # Use followers + friends + mention
    python main.py --seed 42               # Set random seed
"""

import os
import sys
import argparse
import time
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from src.data_loader import load_mgtab, get_feature_vectors
from src.graph_builder import analyze_graph
from src.feature_engineering import analyze_features
from src.models.baseline_models import BaselineModels
from src.train import train_gnn
from src.evaluate import (
    evaluate_gnn, compare_all_models,
    plot_confusion_matrices, plot_roc_curves,
    plot_training_curves, plot_f1_comparison_bar,
    save_all_results
)
from src.visualize import generate_all_visualizations


def parse_args():
    parser = argparse.ArgumentParser(
        description='Bot Detection on Twitter using Graph Attention Networks (GAT)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Full pipeline
  python main.py --mode train --model GAT
  python main.py --mode all --relations 0 1 2 3
  python main.py --mode visualize
        """
    )
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'evaluate', 'visualize', 'all'],
                       help='Pipeline mode (default: all)')
    parser.add_argument('--model', type=str, default='all',
                       choices=['GAT', 'GCN', 'GraphSAGE', 'all'],
                       help='GNN model to train (default: all)')
    parser.add_argument('--task', type=str, default='bot',
                       choices=['bot', 'stance'],
                       help='Detection task (default: bot)')
    parser.add_argument('--relations', type=int, nargs='+', default=None,
                       help='Relation indices to use (0-6). Default: 0 1 (followers+friends)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override training epochs')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to MGTAB dataset directory')
    parser.add_argument('--skip_baselines', action='store_true',
                       help='Skip baseline ML models')
    parser.add_argument('--skip_viz', action='store_true',
                       help='Skip visualization generation')
    
    return parser.parse_args()


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║     🤖 Bot Detection using Graph Attention Networks (GAT)    ║
║     Dataset: MGTAB (Multi-relational Graph-Based Twitter     ║
║              Account Detection Benchmark)                    ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    args = parse_args()
    print_banner()
    
    # Override config if args provided
    if args.relations:
        config.RELATION_SELECT = args.relations
    if args.epochs:
        config.TRAINING_CONFIG['epochs'] = args.epochs
    
    start_time = time.time()
    
    # ================================================================
    # STEP 1: Load Dataset
    # ================================================================
    print("\n" + "="*60)
    print("STEP 1: Loading MGTAB Dataset")
    print("="*60)
    
    data, info = load_mgtab(
        data_dir=args.data_dir,
        task=args.task,
        relation_select=config.RELATION_SELECT,
        seed=args.seed,
    )
    
    # ================================================================
    # STEP 2: Graph Analysis
    # ================================================================
    print("\n" + "="*60)
    print("STEP 2: Graph Structure Analysis")
    print("="*60)
    
    graph_stats = analyze_graph(data, info)
    
    # Save graph stats
    stats_path = os.path.join(config.METRICS_DIR, 'graph_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({k: v for k, v in graph_stats.items() 
                   if not isinstance(v, np.ndarray)}, f, indent=2, default=str)
    
    # ================================================================
    # STEP 3: Feature Analysis
    # ================================================================
    print("\n" + "="*60)
    print("STEP 3: Feature Analysis")
    print("="*60)
    
    feat_analysis = analyze_features(data.x, data.y, top_k=10)
    print(f"\nTop 5 discriminative features (by mutual information):")
    for feat in feat_analysis['top_features'][:5]:
        print(f"  #{feat['rank']}: {feat['name']} (MI={feat['mi_score']:.4f})")
    
    if args.mode in ['train', 'all']:
        # ============================================================
        # STEP 4: Train Baseline Models
        # ============================================================
        baseline_results = {}
        
        if not args.skip_baselines:
            print("\n" + "="*60)
            print("STEP 4: Training Baseline ML Models")
            print("="*60)
            
            X_train, X_val, X_test, y_train, y_val, y_test = get_feature_vectors(data)
            print(f"Feature vectors: train={X_train.shape}, "
                  f"val={X_val.shape}, test={X_test.shape}")
            
            baselines = BaselineModels(seed=args.seed)
            baseline_results = baselines.train_and_evaluate(
                X_train, X_test, y_train, y_test,
                X_val=X_val, y_val=y_val,
            )
            
            # Feature importance
            importance = baselines.get_feature_importance()
            if importance:
                print("\nTop 10 features (Random Forest importance):")
                for feat in importance[:10]:
                    print(f"  #{feat['rank']}: {feat['name']} ({feat['importance']:.4f})")
            
            baselines.save_results()
        
        # ============================================================
        # STEP 5: Train GNN Models
        # ============================================================
        print("\n" + "="*60)
        print("STEP 5: Training GNN Models")
        print("="*60)
        
        gnn_models = {}
        gnn_histories = {}
        
        models_to_train = ['GAT', 'GCN', 'GraphSAGE'] if args.model == 'all' else [args.model]
        
        for model_name in models_to_train:
            model, history = train_gnn(data, model_name=model_name, seed=args.seed)
            gnn_models[model_name] = model
            gnn_histories[model_name] = history
        
        # ============================================================
        # STEP 6: Evaluate All Models
        # ============================================================
        print("\n" + "="*60)
        print("STEP 6: Model Evaluation")
        print("="*60)
        
        gnn_eval_results = {}
        for model_name, model in gnn_models.items():
            gnn_eval_results[model_name] = evaluate_gnn(model, data, model_name)
        
        # Compare all models
        all_results = {**gnn_eval_results}
        if baseline_results:
            all_results.update(baseline_results)
        
        comparison, display = compare_all_models(gnn_eval_results, baseline_results)
        
        # Save all results
        save_all_results(all_results)
        
        # ============================================================
        # STEP 7: Generate Plots
        # ============================================================
        print("\n" + "="*60)
        print("STEP 7: Generating Evaluation Plots")
        print("="*60)
        
        # Training curves
        plot_training_curves(gnn_histories)
        
        # Confusion matrices
        plot_confusion_matrices(all_results)
        
        # ROC curves
        plot_roc_curves(all_results)
        
        # Comparison bar chart
        plot_f1_comparison_bar(all_results)
        
        # ============================================================
        # STEP 8: Visualizations
        # ============================================================
        if not args.skip_viz:
            print("\n" + "="*60)
            print("STEP 8: Generating Visualizations")
            print("="*60)
            
            generate_all_visualizations(data, gnn_models, gnn_histories)
    
    elif args.mode == 'evaluate':
        print("\n⚠️  Evaluation mode requires trained models.")
        print("Run with --mode train first, or use --mode all")
    
    elif args.mode == 'visualize':
        print("\n⚠️  Visualization mode requires trained models.")
        print("Run with --mode all for full pipeline")
    
    # ================================================================
    # DONE
    # ================================================================
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Results:    {config.METRICS_DIR}")
    print(f"  Figures:    {config.FIGURES_DIR}")
    print(f"  Models:     {config.MODELS_DIR}")
    print(f"\n  Run 'streamlit run app/dashboard.py' for interactive dashboard")


if __name__ == "__main__":
    main()
