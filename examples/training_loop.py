import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import torch.optim as optim
import cfdnet as cf

# Initialize the CFDNet model
vocab_size = 5000
d_model = 128
num_layers = 6
max_seq_len = 1024
model = cf.CFDNet(vocab_size, d_model, num_layers, max_seq_len)

# Example input and target sequences (batch size: 2, sequence length: 256)
input_seq = torch.randint(0, vocab_size, (2, 256))
target_seq = torch.randint(0, vocab_size, (2, 256))

# Create a mask for padding tokens
pad_idx = 0
mask = cf.create_pad_mask(input_seq, pad_idx)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Training loop (1 iteration example)
model.train()
optimizer.zero_grad()

# Forward pass
output = model(input_seq, mask=mask)

# Compute the loss (ignoring padding tokens)
loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))

# Backward pass and optimization
loss.backward()
optimizer.step()

print("Training loss:", loss.item())
