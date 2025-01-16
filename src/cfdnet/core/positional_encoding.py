import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import numpy as np

class CubicSplinePositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model, num_control_points=32):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_control_points = num_control_points

        # Learnable control points
        self.control_points = nn.Parameter(
            torch.randn(num_control_points, d_model) / np.sqrt(num_control_points)
        )

        # Precompute basis functions
        self.register_buffer('basis_functions', self._compute_basis_functions())

    def _compute_basis_functions(self):
        t = torch.linspace(0, 1, self.max_seq_len)
        knots = torch.linspace(0, 1, self.num_control_points - 2)
        basis = torch.zeros(self.max_seq_len, self.num_control_points)

        # Generate basis
        for i in range(self.max_seq_len):
            for j in range(self.num_control_points):
                basis[i, j] = self._cubic_bspline(t[i], j, knots)

        return basis

    @staticmethod
    def _cubic_bspline(t, i, knots):
        # Approximate cubic B-spline
        return (t - knots[i])**3 if 0 <= t - knots[i] < 1 else 0

    def forward(self, x):
        seq_len = x.size(1)
        pos_encodings = torch.matmul(
            self.basis_functions[:seq_len],
            self.control_points
        )
        return x + pos_encodings
