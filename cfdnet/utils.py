import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        """
        Positional embedding using learnable cubic splines.

        Args:
            max_seq_len (int): Maximum sequence length.
            d_model (int): Dimension of the embeddings.
        """
        super(PositionalEmbedding, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Learnable spline control points for each position dimension
        # Shape: (max_seq_len, d_model)
        self.control_points = nn.Parameter(torch.randn(max_seq_len, d_model))

        # Layer normalization for better gradient flow
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Forward pass for positional embedding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Positional embeddings of shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len = x.size()
        assert seq_len <= self.max_seq_len, "Sequence length exceeds max_seq_len"

        # Retrieve positional embeddings for the given sequence length
        pos_emb = self.control_points[:seq_len]  # Shape: (seq_len, d_model)

        # Expand embeddings to match batch size
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, seq_len, d_model)

        # Normalize embeddings
        pos_emb = self.layer_norm(pos_emb)

        return pos_emb
