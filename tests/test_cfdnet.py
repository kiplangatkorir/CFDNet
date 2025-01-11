import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
from cfdnet.models import CFDNet

def main():
    vocab_size = 5000
    d_model = 256
    num_layers = 4
    max_seq_len = 128

    model = CFDNet(vocab_size, d_model, num_layers, max_seq_len)

    batch_size = 2
    input_seq = torch.randint(0, vocab_size, (batch_size, max_seq_len))

    output = model(input_seq)

    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()
