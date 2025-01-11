import torch
import torch.nn as nn

# Spline Function Parameterization
class SplineFunction(nn.Module):
    def __init__(self, num_control_points):
        super(SplineFunction, self).__init__()
        self.control_points = nn.Parameter(torch.randn(num_control_points))  # Learnable control points
        self.knots = torch.linspace(0, 1, num_control_points).unsqueeze(-1)

    def forward(self, x):
        x = x.unsqueeze(-1)  # Reshape input
        distances = torch.cdist(x, self.knots, p=2)  # Compute distances to knots
        weights = torch.softmax(-distances, dim=-1)  # Softmax over control points
        return torch.matmul(weights, self.control_points)  # Weighted sum of control points

# Continuous Positional Embedding
class ContinuousPositionalEmbedding(nn.Module):
    def __init__(self, d_model, num_control_points=10):
        super(ContinuousPositionalEmbedding, self).__init__()
        self.spline_function = SplineFunction(num_control_points)  # Use the spline function for continuous embeddings
        self.linear = nn.Linear(1, d_model)  # Linear layer to project to embedding space

    def forward(self, positions):
        positions = positions.unsqueeze(-1)  # Add feature dimension
        embeddings = self.spline_function(positions)  # Get embeddings from spline
        return self.linear(embeddings)  # Project to the desired embedding dimension

# Decomposition Block
class DecompositionBlock(nn.Module):
    def __init__(self, d_model, num_channels=4):
        super(DecompositionBlock, self).__init__()
        self.num_channels = num_channels
        self.psi = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU()
            ) for _ in range(num_channels)
        ])  # Univariate transformations (per channel)
        self.theta = nn.Sequential(
            nn.Linear(num_channels * d_model, d_model),  # Learnable mixing
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        # Apply transformations and combine them
        transformed = [psi(x) for psi in self.psi]
        combined = torch.cat(transformed, dim=-1)  # Concatenate along the feature dimension
        return self.theta(combined)  # Apply learnable mixing and normalization
