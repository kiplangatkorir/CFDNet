import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
from core.model import CFDNet

class TimeSeriesForecasting(nn.Module):
    """
    Time series forecasting using CFDNet.
    """
    def __init__(self, input_dim, d_model, num_layers, max_seq_len, output_dim, num_functions=8, dropout=0.1):
        super().__init__()
        self.cfdnet = CFDNet(input_dim, d_model, num_layers, max_seq_len, num_functions, dropout)
        self.forecaster = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        Forward pass for time-series forecasting.
        """
        x = self.cfdnet(x)  # Extract temporal features
        forecasts = self.forecaster(x[:, -1, :])  # Use the last timestep's features for prediction
        return forecasts
