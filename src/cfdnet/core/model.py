import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .positional_encoding import CubicSplinePositionalEncoding
from .decomposition import DecompositionBlock

class CFDNet(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, max_seq_len, num_functions=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_scale = np.sqrt(d_model)
        self.positional_encoding = CubicSplinePositionalEncoding(max_seq_len, d_model)
        self.layers = nn.ModuleList([DecompositionBlock(d_model, num_functions) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, vocab_size)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        x = self.embedding(x) * self.embedding_scale
        x = self.positional_encoding(x)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        for layer in self.layers:
            x = self.dropout(layer(x))
        return F.log_softmax(self.output_layer(x), dim=-1)
