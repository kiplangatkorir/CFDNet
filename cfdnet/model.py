import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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
        self.positional_embedding = nn.Parameter(torch.randn(1, 1024, d_model))  
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
    def __init__(self, d_model, num_heads=8, num_functions=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_functions = num_functions
        
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.attention_norm = nn.BatchNorm1d(d_model)
        
        self.psi_functions = nn.ModuleList([AdaptiveUnivariateFunction() for _ in range(num_functions)])
        
        self.gate = nn.Sequential(
            nn.Linear(num_functions, d_model * 2),
            nn.BatchNorm1d(d_model * 2),
            nn.GLU(),
            nn.LayerNorm(d_model)
        )
        
        self.input_projection = nn.Linear(d_model, num_functions)
        self.projection_norm = nn.BatchNorm1d(num_functions)
        
        self.output_projection = nn.Linear(d_model, d_model)
        self.output_norm = nn.BatchNorm1d(d_model)
        
    def forward(self, x):
        # Reshape for batch norm
        batch_size, seq_len, d_model = x.shape
        
        # Attention with batch norm
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.transpose(1, 2)
        attn_output = self.attention_norm(attn_output)
        attn_output = attn_output.transpose(1, 2)
        
        # Projection with batch norm
        projected = self.input_projection(x)
        projected = projected.transpose(1, 2)
        projected = self.projection_norm(projected)
        projected = projected.transpose(1, 2)
        
        # Transform and gate
        transformed = torch.stack([f(projected[..., i]) for i, f in enumerate(self.psi_functions)], dim=-1)
        gated_output = self.gate(transformed)
        
        # Output projection with batch norm
        output = self.output_projection(gated_output)
        output = output.transpose(1, 2)
        output = self.output_norm(output)
        output = output.transpose(1, 2)
        
        return x + attn_output + output

class CFDNetScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.compute_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        
    def compute_rate(self):
        step = self._step
        factor = self.factor
        d_model = self.d_model
        warmup = self.warmup_steps
        return factor * (d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

class CFDNet(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, max_seq_len, 
                 num_functions=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_scale = np.sqrt(d_model)
        self.positional_encoding = CubicSplinePositionalEncoding(max_seq_len, d_model)
        self.layers = nn.ModuleList([DecompositionBlock(d_model, num_functions) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self.optimizer = None
        self.scheduler = None
        
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def configure_optimizer(self, warmup_steps=4000, factor=1.0):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = CFDNetScheduler(self.optimizer, self.d_model, warmup_steps, factor)
        return self.optimizer, self.scheduler
                
    def forward(self, x, mask=None):
        x = self.embedding(x) * self.embedding_scale
        x = self.positional_encoding(x)
        
        if x.dim() == 4:
            print(f"Reshaping x from 4D to 3D: {x.shape}")
            x = x.view(x.size(0), x.size(1), -1)  
        print(f"x shape after reshaping: {x.shape}")
        
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)  
        
        for layer in self.layers:
            x = self.dropout(layer(x))
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1), 0)
        
        output = self.output_layer(x)
        return F.log_softmax(output, dim=-1)



def create_pad_mask(seq, pad_idx):
    """Create padding mask."""
    return (seq == pad_idx).unsqueeze(-2)  
