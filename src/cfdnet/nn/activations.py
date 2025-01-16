import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch.nn as nn

class GLUActivation(nn.Module):
    """Gated Linear Unit activation."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.glu(x, dim=-1)
