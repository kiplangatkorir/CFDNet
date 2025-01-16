import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import numpy as np

class AdaptiveUnivariateFunction(nn.Module):
    def __init__(self, num_control_points=32):
        super().__init__()
        self.num_control_points = num_control_points

        # Learnable control points and knots
        self.control_points = nn.Parameter(
            torch.randn(num_control_points) / np.sqrt(num_control_points)
        )
        self.knots = nn.Parameter(
            torch.linspace(0, 1, num_control_points)
        )

    def forward(self, x):
        x_normalized = (x - x.min()) / (x.max() - x.min() + 1e-6)

        output = torch.zeros_like(x)
        for i in range(self.num_control_points - 1):
            mask = (x_normalized >= self.knots[i]) & (x_normalized <= self.knots[i + 1])
            t = (x_normalized[mask] - self.knots[i]) / (self.knots[i + 1] - self.knots[i])
            output[mask] = (1 - t) * self.control_points[i] + t * self.control_points[i + 1]

        return output
