import torch
import torch.nn as nn
from .utils import PositionalEmbedding
from .layers import DecompositionBlock

class CFDNet(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, max_seq_len):
        """
        CFDNet model definition, consisting of positional embeddings and decomposition blocks.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the embeddings.
            num_layers (int): Number of decomposition layers.
            max_seq_len (int): Maximum sequence length.
        """
        super(CFDNet, self).__init__()

        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_layers = num_layers

        # Embedding layer for the input
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional embedding layer
        self.pos_embedding = PositionalEmbedding(max_seq_len, d_model)

        # List of decomposition blocks
        self.decomposition_blocks = nn.ModuleList(
            [DecompositionBlock(d_model) for _ in range(num_layers)]
        )

        # Final linear layer to map to the vocabulary size
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Forward pass of CFDNet.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = x.size()

        # Input embedding
        x_embedded = self.embedding(x)  # Shape: (batch_size, seq_len, d_model)

        # Add positional encoding
        pos_embedded = self.pos_embedding(x)
        x = x_embedded + pos_embedded  # Add the positional embeddings

        # Pass through the decomposition layers
        for layer in self.decomposition_blocks:
            x = layer(x)

        # Final output layer
        output = self.output_layer(x)  # Shape: (batch_size, seq_len, vocab_size)
        
        return output
if __name__ == "__main__":
    # Parameters
    vocab_size = 5000
    d_model = 256
    num_layers = 4
    max_seq_len = 256

    # Initialize CFDNet model
    model = CFDNet(vocab_size, d_model, num_layers, max_seq_len)

    # Create a mock input tensor (batch_size, seq_len)
    batch_size = 2
    input_seq = torch.randint(0, vocab_size, (batch_size, 128))  # Random sequence

    # Forward pass
    output = model(input_seq)

    # Output shape should be (batch_size, seq_len, vocab_size)
    print(f"Output shape: {output.shape}")
