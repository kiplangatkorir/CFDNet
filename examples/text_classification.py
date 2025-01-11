import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
from cfdnet.models import CFDNet
from cfdnet.utils import save_model, load_model

# Model configuration
vocab_size = 5000  # Size of the vocabulary (number of unique tokens)
d_model = 128  # Embedding dimension
num_layers = 6  # Number of decomposition layers
max_seq_len = 256  # Maximum sequence length
num_classes = 10  # Number of output classes (for classification task)

# Create the CFDNet model
model = CFDNet(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, max_seq_len=max_seq_len)

# Define a classification head on top of the CFDNet model
class ClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super(ClassificationHead, self).__init__()
        self.cfdnet = model
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Forward pass through the CFDNet model
        features = self.cfdnet(x)
        # Take the mean of the feature representations along the sequence length dimension
        pooled_features = features.mean(dim=1)
        # Classification layer to predict the class logits
        return self.classifier(pooled_features)

# Instantiate the classification model
classification_model = ClassificationHead(d_model=d_model, num_classes=num_classes)

# Example input: A batch of 2 sequences, each with length 256 (tokenized text)
input_seq = torch.randint(0, vocab_size, (2, max_seq_len))

# Forward pass
output = classification_model(input_seq)

# Output shape should be [batch_size, num_classes] -> torch.Size([2, 10])
print("Output shape:", output.size())

# Simulate training: Define loss and optimizer
labels = torch.randint(0, num_classes, (2,))  # Simulated labels for the 2 sequences
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classification_model.parameters())

# Compute loss
loss = criterion(output, labels)
print("Loss:", loss.item())

# Backward pass and optimization step
loss.backward()
optimizer.step()

# Save the model
save_model(classification_model, "classification_model.pth")

# Load the model back (example of model loading)
loaded_model = load_model(ClassificationHead, "classification_model.pth", d_model, num_classes)
