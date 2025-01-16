import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CubicSplinePositionalEncoding(nn.Module):
    """Learnable positional encoding using cubic B-splines."""
    def __init__(self, max_seq_len, d_model, num_control_points=32):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_control_points = num_control_points
        self.control_points = nn.Parameter(torch.randn(num_control_points, d_model) / np.sqrt(num_control_points))
        self.register_buffer('basis_functions', self._compute_basis_functions())
        
    def _compute_basis_functions(self):
        t = torch.linspace(0, 1, self.max_seq_len)
        knots = torch.linspace(0, 1, self.num_control_points - 2)
        basis = torch.zeros(self.max_seq_len, self.num_control_points)
        for i in range(self.max_seq_len):
            for j in range(self.num_control_points):
                basis[i, j] = self._cubic_bspline(t[i], j, knots)
        return basis

    def _cubic_bspline(self, t, i, knots):
        x = torch.zeros_like(t)
        if i >= 2 and i < len(knots):
            if knots[i-2] <= t <= knots[i-1]:
                x = (t - knots[i-2])**3
        return x
        
    def forward(self, x):
        seq_len = x.size(1)
        pos_encodings = torch.matmul(self.basis_functions[:seq_len], self.control_points)
        return x + pos_encodings

class DynamicPositionalEncoding(nn.Module):
    """Dynamic positional encoding for variable sequence lengths."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.positional_embedding = nn.Parameter(torch.randn(1, 1024, d_model))  # Up to 1024 tokens
        
    def forward(self, x):
        seq_len = x.size(1)
        pos_encodings = self.positional_embedding[:, :seq_len, :]
        return x + pos_encodings


class AdaptiveUnivariateFunction(nn.Module):
    """Learnable univariate function using adaptive spline interpolation."""
    def __init__(self, num_control_points=32):
        super().__init__()
        self.num_control_points = num_control_points
        self.control_points = nn.Parameter(torch.randn(num_control_points) / np.sqrt(num_control_points))
        self.knots = nn.Parameter(torch.linspace(0, 1, num_control_points))
        
    def forward(self, x):
        x_normalized = (x - x.min()) / (x.max() - x.min() + 1e-6)
        output = torch.zeros_like(x)
        for i in range(self.num_control_points - 1):
            mask = (x_normalized >= self.knots[i]) & (x_normalized <= self.knots[i + 1])
            t = (x_normalized[mask] - self.knots[i]) / (self.knots[i + 1] - self.knots[i])
            output[mask] = (1 - t) * self.control_points[i] + t * self.control_points[i + 1]
        return output


class DecompositionBlock(nn.Module):
    """Decomposition block with multi-head attention, GLU gating, and residual connections."""
    def __init__(self, d_model, num_heads=8, num_functions=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_functions = num_functions
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        
        # Univariate transformation functions
        self.psi_functions = nn.ModuleList([AdaptiveUnivariateFunction() for _ in range(num_functions)])
        
        # GLU gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GLU(),
            nn.LayerNorm(d_model)
        )
        
        # Learnable projection matrices
        self.input_projection = nn.Linear(d_model, num_functions)
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x)
        
        # Project input to function space
        projected = self.input_projection(x)
        
        # Apply univariate transformations
        transformed = torch.stack([f(projected[..., i]) for i, f in enumerate(self.psi_functions)], dim=-1)
        
        # Apply GLU gating
        gated_output = self.gate(transformed)
        
        # Final projection with skip connection
        return x + attn_output + self.output_projection(gated_output)


class CFDNet(nn.Module):
    """Enhanced Continuous Function Decomposition Network with multi-head attention, GLU, and residuals."""
    def __init__(self, vocab_size, d_model, num_layers, max_seq_len, num_functions=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_scale = np.sqrt(d_model)
        
        # Enhanced positional encoding
        self.positional_encoding = DynamicPositionalEncoding(d_model)
        
        # Decomposition layers with dropout
        self.layers = nn.ModuleList([DecompositionBlock(d_model, num_functions=num_functions) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize network parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x, mask=None):
        # Embed and scale tokens
        x = self.embedding(x) * self.embedding_scale
        x = self.positional_encoding(x)
        
        # Apply mask if provided
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        
        # Pass through decomposition layers
        residual = x
        for layer in self.layers:
            x = self.dropout(layer(x)) + residual
            residual = x  # Residual for next layer
                
        # Project to vocabulary
        output = self.output_layer(x)
        return F.log_softmax(output, dim=-1)


def create_pad_mask(seq, pad_idx):
    """Create padding mask."""
    return (seq == pad_idx).unsqueeze(-2)
