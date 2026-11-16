"""
Graph Attention Network (GAT) for Bot Detection.

This is the CORE model of the project. Implements GAT with:
- GATv2Conv layers (fixes the static attention problem of original GAT)
- Multi-head attention mechanism
- Residual connections
- Layer normalization
- Dropout for regularization
- Attention weight extraction for interpretability

Mathematical Formulation:
========================
GAT computes attention coefficients between connected nodes:

    e_ij = a(W*h_i, W*h_j)          # Attention coefficient
    α_ij = softmax_j(e_ij)           # Normalized attention (over neighbors j)
    h'_i = σ(Σ_j α_ij * W * h_j)    # Updated node embedding

GATv2 improvement:
    e_ij = a^T * LeakyReLU(W * [h_i || h_j])   # Dynamic attention

Multi-head (K heads):
    h'_i = ||_{k=1}^{K} σ(Σ_j α_ij^k * W^k * h_j)  # Concatenation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, GATv2Conv
except ImportError:
    raise ImportError(
        "torch_geometric is required. Install with:\n"
        "pip install torch-geometric"
    )

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class GATBotDetector(nn.Module):
    """
    Graph Attention Network for Twitter Bot Detection.
    
    Architecture:
        Input [N × F] 
          → Linear + LeakyReLU [N × hidden_dim]
          → GATv2Conv (multi-head) + ELU + Dropout [N × (hidden_dim)]
          → GATv2Conv (multi-head) + ELU + Dropout [N × (hidden_dim)]
          → Linear [N × num_classes]
    
    The model uses the MGTAB-style architecture adapted for GATv2Conv:
    - Input linear projection to hidden_dim
    - Two GATv2Conv layers with multi-head attention
    - Output linear classifier
    
    Args:
        in_channels: Number of input features per node
        hidden_channels: Hidden layer dimension
        out_channels: Number of output classes (2 for bot detection)
        num_heads: Number of attention heads
        num_layers: Number of GAT layers (default 2)
        dropout: Dropout probability
        use_gatv2: Whether to use GATv2Conv (recommended)
        residual: Whether to use residual connections
        layer_norm: Whether to use layer normalization
    """
    
    def __init__(
        self,
        in_channels,
        hidden_channels=256,
        out_channels=2,
        num_heads=8,
        num_layers=2,
        dropout=0.3,
        use_gatv2=True,
        residual=True,
        layer_norm=True,
    ):
        super(GATBotDetector, self).__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        self.residual = residual
        self.use_layer_norm = layer_norm
        
        # Input projection: map raw features to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )
        
        # Select GAT convolution type
        GATLayer = GATv2Conv if use_gatv2 else GATConv
        
        # Build GAT layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                # First GAT layer: hidden_dim → hidden_dim/heads * heads
                self.gat_layers.append(
                    GATLayer(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True,  # Concatenate heads → output dim = hidden_channels
                        add_self_loops=True,
                    )
                )
            else:
                # Subsequent layers
                self.gat_layers.append(
                    GATLayer(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True,
                        add_self_loops=True,
                    )
                )
            
            if layer_norm:
                self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Output classifier
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
        # Store attention weights for interpretability
        self._attention_weights = []
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize linear layers with Kaiming uniform."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, edge_type=None, return_attention=False):
        """
        Forward pass.
        
        Args:
            x: Node feature matrix [N × F]
            edge_index: Edge indices [2 × E]
            edge_type: Edge type labels (unused in vanilla GAT, kept for API consistency)
            return_attention: If True, return attention weights
            
        Returns:
            out: Class logits [N × num_classes]
            attention_weights: (optional) List of attention weight tensors
        """
        self._attention_weights = []
        
        # Input projection
        h = self.input_proj(x.to(torch.float32))
        
        # GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            h_residual = h
            
            if return_attention:
                h, (edge_idx, alpha) = gat_layer(
                    h, edge_index, return_attention_weights=True
                )
                self._attention_weights.append(alpha)
            else:
                h = gat_layer(h, edge_index)
            
            # Layer normalization
            if self.use_layer_norm:
                h = self.norms[i](h)
            
            # Activation
            h = F.elu(h)
            
            # Residual connection
            if self.residual and h_residual.shape == h.shape:
                h = h + h_residual
            
            # Dropout
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Store embeddings before classification (for t-SNE visualization)
        self._embeddings = h.detach()
        
        # Classification
        out = self.classifier(h)
        
        if return_attention:
            return out, self._attention_weights
        return out
    
    def get_embeddings(self, x, edge_index, edge_type=None):
        """
        Get node embeddings from the last hidden layer (before classifier).
        Useful for t-SNE / PCA visualization.
        
        Returns:
            Embeddings [N × hidden_channels]
        """
        self.eval()
        with torch.no_grad():
            self.forward(x, edge_index, edge_type)
        return self._embeddings
    
    def get_attention_weights(self, x, edge_index, edge_type=None):
        """
        Get attention weights for all GAT layers.
        
        Returns:
            List of attention weight tensors, one per layer
        """
        self.eval()
        with torch.no_grad():
            self.forward(x, edge_index, edge_type, return_attention=True)
        return self._attention_weights
    
    def __repr__(self):
        return (
            f"GATBotDetector(\n"
            f"  input_proj={self.input_proj},\n"
            f"  gat_layers={self.gat_layers},\n"
            f"  classifier={self.classifier},\n"
            f"  dropout={self.dropout}, residual={self.residual}\n"
            f")"
        )


def create_gat_model(in_channels, num_classes=2, cfg=None):
    """
    Factory function to create a GAT model from config.
    
    Args:
        in_channels: Number of input features
        num_classes: Number of output classes
        cfg: Optional config dict (uses config.GAT_CONFIG if None)
        
    Returns:
        GATBotDetector model
    """
    if cfg is None:
        cfg = config.GAT_CONFIG
    
    model = GATBotDetector(
        in_channels=in_channels,
        hidden_channels=cfg['hidden_dim'],
        out_channels=num_classes,
        num_heads=cfg['num_heads'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
        use_gatv2=cfg.get('use_gatv2', True),
        residual=cfg.get('residual', True),
        layer_norm=cfg.get('layer_norm', True),
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[GAT] Created model: {trainable_params:,} trainable params "
          f"(total: {total_params:,})")
    
    return model


if __name__ == "__main__":
    """Test GAT model."""
    # Dummy data
    N, F, E = 100, 789, 500
    x = torch.randn(N, F)
    edge_index = torch.randint(0, N, (2, E))
    
    model = create_gat_model(in_channels=F, num_classes=2)
    print(model)
    
    # Forward pass
    out = model(x, edge_index)
    print(f"\nOutput shape: {out.shape}")
    print(f"Predictions: {out.argmax(dim=1)[:10]}")
    
    # Get embeddings
    embeddings = model.get_embeddings(x, edge_index)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Get attention weights
    attn = model.get_attention_weights(x, edge_index)
    print(f"Attention weights: {len(attn)} layers")
    for i, a in enumerate(attn):
        print(f"  Layer {i}: {a.shape}")
