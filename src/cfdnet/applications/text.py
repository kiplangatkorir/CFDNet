import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextClassification(nn.Module):
    """
    A wrapper for text classification tasks using CFDNet.
    """
    def __init__(self, vocab_size, d_model, num_layers, max_seq_len, num_classes, num_functions=8, dropout=0.1):
        super().__init__()
        self.cfdnet = CFDNet(vocab_size, d_model, num_layers, max_seq_len, num_functions, dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        """
        Forward pass for text classification.
        """
        x = self.cfdnet(x, mask)  # Use CFDNet as feature extractor
        logits = self.classifier(x[:, 0, :])  # Classify based on [CLS] token
        return F.log_softmax(logits, dim=-1)
