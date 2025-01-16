import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
from core.model import CFDNet

class PDESolver(nn.Module):
    """
    Solve Partial Differential Equations (PDEs) using CFDNet.
    """
    def __init__(self, input_dim, d_model, num_layers, max_seq_len, output_dim, num_functions=8, dropout=0.1):
        super().__init__()
        self.cfdnet = CFDNet(input_dim, d_model, num_layers, max_seq_len, num_functions, dropout)
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        Forward pass for PDE solving.
        """
        x = self.cfdnet(x)  
        solution = self.output_layer(x)
        return solution
