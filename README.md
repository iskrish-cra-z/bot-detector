# 🤖 Bot/Fake Account Detection on Twitter using Graph Attention Networks (GAT)

> Final Year Computer Science Engineering Project

## 📋 Project Overview

This project detects bot/fake accounts on Twitter by framing the problem as **node classification on a social graph** using **Graph Attention Networks (GAT)**. It leverages the [MGTAB](https://github.com/GraphDetec/MGTAB) benchmark dataset, which provides 10,199 expert-annotated Twitter users with 7 types of social relationships.

### Why Graph Neural Networks?

Traditional ML models treat each user independently. But bot accounts often form **coordinated networks** — they follow each other, retweet each other, and exhibit similar patterns. GAT captures these **relational patterns** through learned attention-weighted message passing, determining which neighbors are most informative for each node.

## 🏗️ Architecture

```
                    ┌──────────────────┐
                    │   MGTAB Dataset   │
                    │  10,199 users     │
                    │  7 relation types │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Graph Builder    │
                    │  Edge filtering   │
                    │  Feature prep     │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼─────┐ ┌─────▼──────┐ ┌─────▼──────┐
    │  Baseline ML   │ │  GCN/SAGE  │ │    GAT     │
    │  LR, RF, SVM   │ │  (Compare) │ │  (Primary) │
    └─────────┬─────┘ └─────┬──────┘ └─────┬──────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼─────────┐
                    │    Evaluation     │
                    │  Metrics, Plots   │
                    │  Visualization    │
                    └──────────────────┘
```

### GAT Model Architecture

```
Input Features [N × F]
    ↓
Linear + LeakyReLU [N × 256]
    ↓
GATv2Conv (8 heads) + LayerNorm + ELU + Residual + Dropout [N × 256]
    ↓
GATv2Conv (8 heads) + LayerNorm + ELU + Residual + Dropout [N × 256]
    ↓
Linear Classifier [N × 2]
    ↓
Output: Bot/Human prediction per node
```

## 📊 Dataset: MGTAB

| Property | Value |
|----------|-------|
| Total users | 10,199 |
| Human accounts | 7,451 (73.06%) |
| Bot accounts | 2,748 (26.94%) |
| Relation types | 7 (followers, friends, mention, reply, quoted, URL, hashtag) |
| Feature dimension | Pre-computed (metadata + text embeddings) |

### Download

1. Download MGTAB from [Google Drive](https://drive.google.com/uc?export=download&id=1gbWNOoU1JB8RrTu2a5j9KMNVa9wX72Fe)
2. Extract the zip into `data/MGTAB/`
3. Verify you have: `features.pt`, `edge_index.pt`, `edge_type.pt`, `edge_weight.pt`, `labels_bot.pt`, `labels_stance.pt`

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** For PyTorch Geometric, follow the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) matching your PyTorch and CUDA versions.

### 2. Download Dataset

Download MGTAB and extract into `data/MGTAB/`.

### 3. Run Full Pipeline

```bash
python main.py
```

This will:
1. Load the MGTAB dataset
2. Analyze graph structure
3. Train baseline ML models (LR, RF, SVM, AdaBoost, KNN)
4. Train GNN models (GAT, GCN, GraphSAGE)
5. Evaluate all models
6. Generate all visualizations
7. Save results

### 4. Launch Dashboard

```bash
streamlit run app/dashboard.py
```

## 📁 Project Structure

```
Bot detector/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── config.py                     # All hyperparameters and paths
├── main.py                       # Main entry point (CLI)
│
├── data/                         # Data directory
│   └── MGTAB/                    # MGTAB dataset (.pt files)
│
├── src/                          # Source code
│   ├── data_loader.py            # MGTAB dataset loading & preprocessing
│   ├── feature_engineering.py    # Feature analysis & normalization
│   ├── graph_builder.py          # Graph construction & analysis
│   ├── train.py                  # GNN training pipeline
│   ├── evaluate.py               # Evaluation & comparison metrics
│   ├── visualize.py              # Graph & embedding visualization
│   │
│   └── models/                   # Model definitions
│       ├── gat_model.py          # GAT (primary model)
│       ├── gcn_model.py          # GCN (comparison)
│       ├── graphsage_model.py    # GraphSAGE (comparison)
│       └── baseline_models.py    # LR, RF, SVM, AdaBoost, KNN
│
├── app/                          # Streamlit dashboard
│   └── dashboard.py
│
└── results/                      # Generated outputs
    ├── figures/                  # Plots (PNG)
    ├── models/                   # Saved model checkpoints
    └── metrics/                  # Evaluation results (JSON)
```

## ⚙️ CLI Options

```bash
python main.py --help

# Train only GAT
python main.py --model GAT

# Use specific relations (0=followers, 1=friends, 2=mention, ...)
python main.py --relations 0 1 2

# Custom epochs
python main.py --epochs 100

# Skip baseline models
python main.py --skip_baselines

# Skip visualizations (faster)
python main.py --skip_viz
```

## 📈 Expected Results (MGTAB paper benchmarks)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 88.49 | 85.69 | 84.41 | 84.97 |
| Random Forest | 89.52 | 88.92 | 86.72 | 86.83 |
| GCN | 85.81 | 77.40 | 84.37 | 78.33 |
| GraphSAGE | 88.71 | 85.33 | 86.15 | 85.44 |
| **GAT** | **86.96** | **79.71** | **84.88** | **82.33** |

## 🔧 Technical Details

### Evaluation Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: Of predicted bots, how many are actually bots
- **Recall**: Of actual bots, how many are detected
- **F1-Score**: Harmonic mean of precision and recall (primary metric)
- **ROC-AUC**: Area under the ROC curve

### Training Details
- Optimizer: AdamW (lr=1e-3, weight_decay=5e-4)
- Early stopping: patience=30 epochs on validation F1
- LR scheduler: ReduceLROnPlateau (factor=0.5, patience=10)
- Gradient clipping: max_norm=1.0
- Split: 70% train / 10% val / 20% test (shuffled)

## 📝 References

1. Shi et al., "MGTAB: A Multi-Relational Graph-Based Twitter Account Detection Benchmark" (2023)
2. Veličković et al., "Graph Attention Networks" (ICLR 2018)
3. Brody et al., "How Attentive are Graph Attention Networks?" (ICLR 2022)
4. Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017)
5. Hamilton et al., "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)

## 📄 License

This project uses the MGTAB dataset under CC BY-NC-ND 4.0 license.
Code is for educational purposes (final year project).
