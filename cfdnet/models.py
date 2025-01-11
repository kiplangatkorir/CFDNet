import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import DecompositionBlock
from .utils import PositionalEmbedding

class CFDNet(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, max_seq_len, num_channels=4):
        """
        Continuous Function Decomposition Network (CFDNet)

        Args:
            vocab_size (int): Vocabulary size for input sequences.
            d_model (int): Dimension of the embedding space.
            num_layers (int): Number of decomposition layers.
            max_seq_len (int): Maximum sequence length.
            num_channels (int): Number of decomposition channels.
        """
        super(CFDNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model, max_seq_len)
        self.decomposition_blocks = nn.ModuleList([
            DecompositionBlock(d_model, num_channels) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Forward pass for CFDNet.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size).
        """
        x = self.embedding(x)  
        x = x + self.positional_embedding(x)

        for block in self.decomposition_blocks:
            x = block(x)

        x = self.output_layer(x)  
        return x
