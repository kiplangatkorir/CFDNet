import torch
import torch.nn as nn


class SplineFunction(nn.Module):
    def __init__(self, num_control_points):
        super(SplineFunction, self).__init__()
        self.control_points = nn.Parameter(torch.randn(num_control_points))  # Learnable control points
        self.knots = torch.linspace(0, 1, num_control_points).unsqueeze(-1)  # Shape (num_control_points, 1)

    def forward(self, x):
        x = x.float().unsqueeze(-1)  # Convert x to float and add feature dimension
        distances = torch.cdist(x, self.knots, p=2)  # Compute pairwise distances
        weights = torch.softmax(-distances, dim=-1)  # Compute softmax over control points
        return torch.matmul(weights, self.control_points)  # Weighted sum


class ContinuousPositionalEmbedding(nn.Module):
    def __init__(self, d_model, num_control_points=10):
        super(ContinuousPositionalEmbedding, self).__init__()
        self.spline_function = SplineFunction(num_control_points)  
        self.linear = nn.Linear(1, d_model)  

    def forward(self, positions):
        positions = positions.unsqueeze(-1)  
        embeddings = self.spline_function(positions)  
        return self.linear(embeddings)  

class DecompositionBlock(nn.Module):
    def __init__(self, d_model, num_channels=4):
        super(DecompositionBlock, self).__init__()
        self.num_channels = num_channels
        self.psi = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU()
            ) for _ in range(num_channels)
        ])  
        self.theta = nn.Sequential(
            nn.Linear(num_channels * d_model, d_model),  
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        transformed = [psi(x) for psi in self.psi]
        combined = torch.cat(transformed, dim=-1)  
        return self.theta(combined)  
