import torch
import torch.nn as nn
from cfdnet.core.univariate_functions import ContinuousPositionalEmbedding, DecompositionBlock

class CFDNet(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, max_seq_len, num_control_points=10):
        super(CFDNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  
        self.positional_embedding = ContinuousPositionalEmbedding(d_model, num_control_points)  
        self.blocks = nn.ModuleList([
            DecompositionBlock(d_model) for _ in range(num_layers)
        ])  
        self.output_layer = nn.Linear(d_model, vocab_size)  

    def forward(self, x):
        batch_size, seq_len = x.size()  
        positions = torch.arange(seq_len).float().to(x.device)  
        x = self.embedding(x) + self.positional_embedding(positions)  
        for block in self.blocks:
            x = x + block(x)  
        return self.output_layer(x)  
