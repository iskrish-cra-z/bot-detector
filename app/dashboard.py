"""
Interactive Streamlit Dashboard for Bot Detection Results.

Provides an interactive interface to explore:
- Dataset statistics and graph properties
- Model performance comparison
- Node-level predictions
- Embedding visualizations
- Feature importance analysis

Run with: streamlit run app/dashboard.py
"""

import os
import sys
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config


# ================================================================
# Page Configuration
# ================================================================
st.set_page_config(
    page_title="🤖 Bot Detector — GAT Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


def load_results():
    """Load saved results from disk."""
    results = {}
    
    # Load evaluation results
    results_path = os.path.join(config.METRICS_DIR, 'all_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results['evaluation'] = json.load(f)
    
    # Load baseline results
    baseline_path = os.path.join(config.METRICS_DIR, 'baseline_results.json')
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            results['baselines'] = json.load(f)
    
    # Load graph stats
    stats_path = os.path.join(config.METRICS_DIR, 'graph_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            results['graph_stats'] = json.load(f)
    
    return results


def page_overview():
    """Overview page with dataset and graph statistics."""
    st.markdown('<h1 class="main-header">🤖 Bot Detection Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("**Twitter Bot Detection using Graph Attention Networks (GAT)** "
                "on the MGTAB Dataset")
    
    results = load_results()
    
    if 'graph_stats' in results:
        stats = results['graph_stats']
        
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Nodes", f"{stats.get('num_nodes', 'N/A'):,}")
        with col2:
            st.metric("Total Edges", f"{stats.get('num_edges', 'N/A'):,}")
        with col3:
            st.metric("Avg Degree", f"{stats.get('avg_degree', 0):.2f}")
        with col4:
            st.metric("Graph Density", f"{stats.get('density', 0):.6f}")
        
        st.subheader("🔗 Graph Properties")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Human Avg Degree", f"{stats.get('human_avg_degree', 0):.2f}")
        with col2:
            st.metric("Bot Avg Degree", f"{stats.get('bot_avg_degree', 0):.2f}")
        with col3:
            st.metric("Isolated Nodes", stats.get('isolated_nodes', 0))
    else:
        st.warning("⚠️ No results found. Run `python main.py` first to generate results.")
    
    # Show saved figures
    st.subheader("📈 Generated Visualizations")
    figures_dir = config.FIGURES_DIR
    
    if os.path.exists(figures_dir):
        images = sorted([f for f in os.listdir(figures_dir) if f.endswith('.png')])
        
        if images:
            cols = st.columns(2)
            for idx, img_name in enumerate(images):
                with cols[idx % 2]:
                    img_path = os.path.join(figures_dir, img_name)
                    st.image(img_path, caption=img_name.replace('_', ' ').replace('.png', '').title(),
                            use_container_width=True)
        else:
            st.info("No visualization images found. Run `python main.py` to generate them.")


def page_model_comparison():
    """Model comparison page."""
    st.markdown('<h1 class="main-header">📊 Model Comparison</h1>', 
                unsafe_allow_html=True)
    
    results = load_results()
    
    if 'evaluation' not in results:
        st.warning("⚠️ No evaluation results found. Run `python main.py` first.")
        return
    
    eval_results = results['evaluation']
    
    # Build comparison DataFrame
    rows = []
    for model_name, metrics in eval_results.items():
        rows.append({
            'Model': model_name,
            'Type': metrics.get('model_type', 'Unknown'),
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1-Score': metrics.get('f1_score', 0),
            'ROC-AUC': metrics.get('roc_auc', 0),
        })
    
    df = pd.DataFrame(rows).sort_values('F1-Score', ascending=False)
    
    # Display table
    st.subheader("Performance Metrics")
    st.dataframe(df.style.format({
        'Accuracy': '{:.4f}',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1-Score': '{:.4f}',
        'ROC-AUC': '{:.4f}',
    }).highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                     color='lightgreen'),
        use_container_width=True)
    
    # Interactive bar chart
    st.subheader("Visual Comparison")
    
    metric_choice = st.selectbox("Select Metric", 
                                  ['F1-Score', 'Accuracy', 'Precision', 'Recall', 'ROC-AUC'])
    
    fig = px.bar(df, x='Model', y=metric_choice,
                 color='Type',
                 color_discrete_map={'Graph-based': '#e74c3c', 'Feature-based': '#3498db'},
                 title=f'{metric_choice} by Model',
                 text=metric_choice)
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart
    st.subheader("Radar Comparison")
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    for _, row in df.iterrows():
        values = [row[cat] for cat in categories]
        values.append(values[0])  # Close the radar
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=row['Model'],
            opacity=0.6,
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='Multi-Metric Model Comparison (Radar)',
    )
    st.plotly_chart(fig, use_container_width=True)


def page_visualizations():
    """Visualizations gallery page."""
    st.markdown('<h1 class="main-header">🎨 Visualizations</h1>', 
                unsafe_allow_html=True)
    
    figures_dir = config.FIGURES_DIR
    
    if not os.path.exists(figures_dir):
        st.warning("No figures directory found. Run `python main.py` first.")
        return
    
    images = sorted([f for f in os.listdir(figures_dir) if f.endswith('.png')])
    
    if not images:
        st.info("No visualization images found.")
        return
    
    # Categorize images
    categories = {
        'Graph Structure': [img for img in images if 'graph' in img.lower()],
        'Embeddings': [img for img in images if 'tsne' in img.lower() or 'pca' in img.lower()],
        'Training': [img for img in images if 'training' in img.lower() or 'curve' in img.lower()],
        'Evaluation': [img for img in images if 'confusion' in img.lower() or 'roc' in img.lower() or 'comparison' in img.lower()],
        'Features': [img for img in images if 'feature' in img.lower() or 'degree' in img.lower() or 'attention' in img.lower()],
    }
    
    for category, imgs in categories.items():
        if imgs:
            st.subheader(f"📌 {category}")
            cols = st.columns(min(2, len(imgs)))
            for idx, img_name in enumerate(imgs):
                with cols[idx % len(cols)]:
                    img_path = os.path.join(figures_dir, img_name)
                    st.image(img_path, 
                            caption=img_name.replace('_', ' ').replace('.png', '').title(),
                            use_container_width=True)


def page_about():
    """About page with project information."""
    st.markdown('<h1 class="main-header">ℹ️ About</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Bot/Fake Account Detection on Twitter using Graph Attention Networks
    
    ### Problem Statement
    Social media bot accounts manipulate public opinion, spread misinformation, 
    and inflate metrics. This project uses **Graph Attention Networks (GAT)** to 
    detect bot accounts by leveraging both user features AND social network structure.
    
    ### Why Graph Neural Networks?
    Traditional ML models treat each user independently. But bot accounts often 
    form **coordinated networks** — they follow each other, retweet each other, 
    and exhibit similar patterns. Graph Neural Networks can capture these 
    **relational patterns** that tabular models miss.
    
    ### Dataset: MGTAB
    - **10,199 expert-annotated Twitter users**
    - **7 types of relationships**: followers, friends, mention, reply, quoted, URL, hashtag
    - **73% human, 27% bot** accounts
    - Includes user metadata features + text embeddings
    
    ### Models Compared
    | Model | Type | Uses Graph? |
    |-------|------|-------------|
    | Logistic Regression | Linear ML | ❌ |
    | Random Forest | Ensemble ML | ❌ |
    | SVM | Kernel ML | ❌ |
    | GCN | Graph Neural Network | ✅ |
    | GraphSAGE | Graph Neural Network | ✅ |
    | **GAT** | **Graph Attention Network** | ✅ |
    
    ### Key Innovation
    GAT uses **multi-head attention** to learn which neighbors are most important 
    for each node's classification — enabling it to focus on the most informative 
    relationships in the social network.
    
    ### References
    - MGTAB: Shi et al., "MGTAB: A Multi-Relational Graph-Based Twitter Account Detection Benchmark"
    - GAT: Veličković et al., "Graph Attention Networks" (ICLR 2018)
    - GATv2: Brody et al., "How Attentive are Graph Attention Networks?" (ICLR 2022)
    """)


# ================================================================
# Main App Router
# ================================================================
def main():
    st.sidebar.title("🤖 Bot Detector")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Overview", "🏆 Model Comparison", "🎨 Visualizations", "ℹ️ About"],
    )
    
    if page == "📊 Overview":
        page_overview()
    elif page == "🏆 Model Comparison":
        page_model_comparison()
    elif page == "🎨 Visualizations":
        page_visualizations()
    elif page == "ℹ️ About":
        page_about()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Bot Detection using GAT**\n\n"
        "Final Year Project\n\n"
        "Dataset: MGTAB"
    )


if __name__ == "__main__":
    main()
