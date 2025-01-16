import torch.nn as nn

def initialize_weights(model):
    """
    Initialize the weights of the model using Xavier uniform initialization.
    """
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
