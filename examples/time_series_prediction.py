import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
from src.cfdnet.applications.text import TextClassification

# Parameters
VOCAB_SIZE = 10000
D_MODEL = 128
NUM_LAYERS = 4
MAX_SEQ_LEN = 50
NUM_CLASSES = 5
BATCH_SIZE = 32

# Model
model = TextClassification(VOCAB_SIZE, D_MODEL, NUM_LAYERS, MAX_SEQ_LEN, NUM_CLASSES)

# Dummy Data
input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))  # Simulated input IDs
labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

# Loss and Optimization
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training step
outputs = model(input_ids)
loss = criterion(outputs, labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Training step complete. Loss: {loss.item():.4f}")
