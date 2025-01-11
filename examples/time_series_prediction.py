import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
from cfdnet.models import CFDNet
from cfdnet.utils import save_model, load_model

# Model configuration
input_size = 1 
output_size = 1  # Predicting the next value in the time series
vocab_size = 1000  # Simulated tokenization of time series data (could be quantized values)
d_model = 64  # Embedding dimension
num_layers = 4  # Number of decomposition layers
max_seq_len = 50  # Max sequence length for time-series data (e.g., using past 50 timesteps)

# Create CFDNet model for time-series
model = CFDNet(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, max_seq_len=max_seq_len)

# Define a prediction head to forecast the next value in the series
class TimeSeriesModel(nn.Module):
    def __init__(self, d_model, input_size, output_size):
        super(TimeSeriesModel, self).__init__()
        self.embedding = nn.Embedding(input_size, d_model)  
        self.predictor = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.embedding(x)  
        x = self.predictor(x)
        return x



# Instantiate the time series prediction model
ts_model = TimeSeriesModel(d_model=d_model, input_size=input_size, output_size=output_size)

# Simulate a batch of time series data (2 sequences, each with 50 time steps)
input_series = torch.randint(0, vocab_size, (2, max_seq_len))

# Forward pass
predicted_next_value = ts_model(input_series)

print("Predicted next value shape:", predicted_next_value.size())

# Simulate training: Define loss and optimizer
true_next_value = torch.randn(2, input_size)  # Simulated true values for the next time step
criterion = nn.MSELoss()  # Use mean squared error for regression
optimizer = torch.optim.Adam(ts_model.parameters())

# Compute loss
loss = criterion(predicted_next_value, true_next_value)
print("Loss:", loss.item())

# Backward pass and optimization step
loss.backward()
optimizer.step()

# Save the time series prediction model
save_model(ts_model, "ts_prediction_model.pth")

# Load the model back (example of model loading)
loaded_ts_model = load_model(TimeSeriesModel, "ts_prediction_model.pth", d_model, input_size)
