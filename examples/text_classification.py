import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
from cfdnet.models import CFDNet

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
        self.cfdnet = CFDNet(vocab_size, d_model, num_layers, max_seq_len)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Forward pass through the CFDNet model to get feature representations
        features = self.cfdnet.embedding(x) + self.cfdnet.positional_embedding(torch.arange(x.size(1), device=x.device))
        
        # Pass through all blocks without applying the final output layer
        for block in self.cfdnet.blocks:
            features = features + block(features)
        
        # Pool the features (take the mean of the feature representations along the sequence length dimension)
        pooled_features = features.mean(dim=1)  # Shape: [batch_size, d_model]

        # Classification layer to predict the class logits
        return self.classifier(pooled_features)  # Output shape: [batch_size, num_classes]

# Instantiate the classification model
classification_model = ClassificationHead(d_model=d_model, num_classes=num_classes)

# Example input: A batch of 2 sequences, each with length 256 (tokenized text)
input_seq = torch.randint(0, vocab_size, (2, max_seq_len))

# Forward pass through the classification model
output = classification_model(input_seq)

# Output shape should be [batch_size, num_classes] -> torch.Size([2, 10])
print("Output shape:", output.size())
