import torch
import torch.nn as nn

class DecompositionBlock(nn.Module):
    def __init__(self, d_model, num_channels):
        """
        A decomposition block that uses univariate transformations to process input.

        Args:
            d_model (int): Dimension of the input embeddings.
            num_channels (int): Number of decomposition channels.
        """
        super(DecompositionBlock, self).__init__()
        self.num_channels = num_channels

        self.univariate_transforms = nn.ModuleList([
            nn.Linear(d_model // num_channels, d_model // num_channels) for _ in range(num_channels)
        ])

        self.mixing_functions = nn.Linear(d_model, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Forward pass through the decomposition block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Processed tensor of the same shape as input.
        """
        batch_size, seq_len, d_model = x.size()
        assert d_model % self.num_channels == 0, "d_model must be divisible by num_channels"

        split_size = d_model // self.num_channels
        channels = torch.split(x, split_size, dim=-1)  

        transformed_channels = [
            transform(channel) for transform, channel in zip(self.univariate_transforms, channels)
        ]

        recomposed = torch.cat(transformed_channels, dim=-1)  
        mixed = self.mixing_functions(recomposed)  

        output = self.layer_norm(x + mixed)

        return output
