import torch

def create_pad_mask(seq, pad_idx):
    """
    Creates a mask for padding tokens in a sequence.

    Args:
        seq (torch.Tensor): The input sequence tensor of shape [batch_size, seq_len].
        pad_idx (int): The index used for padding tokens.
    
    Returns:
        torch.Tensor: A mask tensor of shape [batch_size, seq_len] with 0 for
        non-padding tokens and 1 for padding tokens.
    """
    return (seq == pad_idx)
