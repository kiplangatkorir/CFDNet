import torch
import torch.nn.functional as F

def calculate_accuracy(output, target, pad_idx=None):
    """
    Calculate token-level accuracy.

    Args:
        output (torch.Tensor): Model predictions of shape [batch_size, seq_len, vocab_size].
        target (torch.Tensor): Ground truth labels of shape [batch_size, seq_len].
        pad_idx (int, optional): Index used for padding tokens.

    Returns:
        float: Accuracy value.
    """
    predictions = output.argmax(dim=-1)
    correct = (predictions == target).float()

    if pad_idx is not None:
        mask = (target != pad_idx)
        correct *= mask
        return correct.sum() / mask.sum()
    else:
        return correct.mean()

def calculate_perplexity(output, target, pad_idx=None):
    """
    Calculate sequence perplexity.

    Args:
        output (torch.Tensor): Model predictions (logits) of shape [batch_size, seq_len, vocab_size].
        target (torch.Tensor): Ground truth labels of shape [batch_size, seq_len].
        pad_idx (int, optional): Index used for padding tokens.

    Returns:
        float: Perplexity value.
    """
    if pad_idx is not None:
        loss = F.cross_entropy(
            output.view(-1, output.size(-1)),
            target.view(-1),
            ignore_index=pad_idx,
            reduction='sum'
        )
        mask = (target != pad_idx).sum()
    else:
        loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1), reduction='mean')
        mask = target.numel()
    
    perplexity = torch.exp(loss / mask)
    return perplexity.item()
