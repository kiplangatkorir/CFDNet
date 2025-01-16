import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    """Projection layer to transform data into different dimensions."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class ResidualLayer(nn.Module):
    """Residual connection wrapper for layers."""
    def __init__(self, sub_layer, dropout=0.1):
        super().__init__()
        self.sub_layer = sub_layer
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(sub_layer.out_features)

    def forward(self, x):
        return x + self.dropout(self.norm(self.sub_layer(x)))
