"""
GraphSAGE Model for Bot Detection.

GraphSAGE (SAmple and aggreGatE) learns node representations by sampling
and aggregating features from a node's local neighborhood.

Key difference from GAT and GCN:
- GCN: Fixed spectral aggregation
- GAT: Learned attention-weighted aggregation
- GraphSAGE: Sampling-based aggregation (mean/max/LSTM) — more scalable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class GraphSAGEBotDetector(nn.Module):
    """
    GraphSAGE for Bot Detection.
    
    Architecture (matches MGTAB paper):
        Input [N × F]
          → Linear + LeakyReLU [N × hidden_dim]
          → SAGEConv + Dropout [N × hidden_dim]
          → SAGEConv [N × hidden_dim]
          → Linear [N × num_classes]
    """
    
    def __init__(self, in_channels, hidden_channels=256, out_channels=2,
                 num_layers=2, dropout=0.3, aggr='mean'):
        super(GraphSAGEBotDetector, self).__init__()
        self.dropout = dropout
        
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )
        
        self.sage_layers = nn.ModuleList()
        for i in range(num_layers):
            self.sage_layers.append(
                SAGEConv(hidden_channels, hidden_channels, aggr=aggr)
            )
        
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
        
        for i, sage in enumerate(self.sage_layers):
            h = sage(h, edge_index)
            if i < len(self.sage_layers) - 1:
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


def create_sage_model(in_channels, num_classes=2, cfg=None):
    if cfg is None:
        cfg = config.SAGE_CONFIG
    
    model = GraphSAGEBotDetector(
        in_channels=in_channels,
        hidden_channels=cfg['hidden_dim'],
        out_channels=num_classes,
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
        aggr=cfg.get('aggr', 'mean'),
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[GraphSAGE] Created model: {total_params:,} trainable params")
    
    return model
