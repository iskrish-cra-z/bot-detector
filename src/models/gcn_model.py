"""
Graph Convolutional Network (GCN) for Bot Detection.

Serves as a comparison baseline to GAT. GCN uses spectral graph convolutions
with a fixed neighborhood aggregation (no attention mechanism).

Key difference from GAT:
- GCN: All neighbors contribute equally (normalized by degree)
- GAT: Neighbors contribute proportionally to learned attention weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class GCNBotDetector(nn.Module):
    """
    Graph Convolutional Network for Bot Detection.
    
    Architecture (matches MGTAB paper):
        Input [N × F]
          → Linear + LeakyReLU [N × hidden_dim]
          → GCNConv + Dropout [N × hidden_dim]
          → GCNConv [N × hidden_dim]
          → Linear [N × num_classes]
    """
    
    def __init__(self, in_channels, hidden_channels=256, out_channels=2,
                 num_layers=2, dropout=0.3):
        super(GCNBotDetector, self).__init__()
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gcn_layers.append(
                GCNConv(hidden_channels, hidden_channels)
            )
        
        # Output classifier
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self._embeddings = None
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, edge_type=None):
        h = self.input_proj(x.to(torch.float32))
        
        for i, gcn in enumerate(self.gcn_layers):
            h = gcn(h, edge_index)
            if i < len(self.gcn_layers) - 1:
                h = F.leaky_relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        self._embeddings = h.detach()
        out = self.classifier(h)
        return out
    
    def get_embeddings(self, x, edge_index, edge_type=None):
        self.eval()
        with torch.no_grad():
            self.forward(x, edge_index, edge_type)
        return self._embeddings


def create_gcn_model(in_channels, num_classes=2, cfg=None):
    if cfg is None:
        cfg = config.GCN_CONFIG
    
    model = GCNBotDetector(
        in_channels=in_channels,
        hidden_channels=cfg['hidden_dim'],
        out_channels=num_classes,
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[GCN] Created model: {total_params:,} trainable params")
    
    return model
