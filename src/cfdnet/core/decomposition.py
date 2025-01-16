import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
from .univariate_functions import AdaptiveUnivariateFunction

class DecompositionBlock(nn.Module):
    def __init__(self, d_model, num_functions=8):
        super().__init__()
        self.d_model = d_model
        self.num_functions = num_functions

        # Functions and mixing
        self.psi_functions = nn.ModuleList([AdaptiveUnivariateFunction() for _ in range(num_functions)])
        self.phi_function = nn.Sequential(
            nn.Linear(num_functions, d_model * 2),
            nn.GLU(),
            nn.LayerNorm(d_model)
        )

        # Projections
        self.input_projection = nn.Linear(d_model, num_functions)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        projected = self.input_projection(x)
        transformed = torch.stack([f(projected[..., i]) for i, f in enumerate(self.psi_functions)], dim=-1)
        mixed = self.phi_function(transformed)
        return x + self.output_projection(mixed)
