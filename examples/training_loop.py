import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import torch.optim as optim
import cfdnet as cf

# Initialize the CFDNet model
vocab_size = 5000  # Size of vocabulary
d_model = 128      # Dimensionality of the model
num_layers = 6     # Number of decomposition layers
max_seq_len = 1024 # Maximum sequence length

model = cf.CFDNet(vocab_size, d_model, num_layers, max_seq_len)

# Example input sequence (batch size: 2, sequence length: 256)
input_seq = torch.randint(0, vocab_size, (2, 256))

# Create a padding mask where 0 indicates padding
pad_idx = 0
mask = cf.create_pad_mask(input_seq, pad_idx)

# Get the model's output
output = model(input_seq, mask=mask)

# Print the shape of the output
print("Output shape with mask:", output.size())  # Should print torch.Size([2, 256, 5000])
