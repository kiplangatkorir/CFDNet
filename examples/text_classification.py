import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
from src.cfdnet.core.model import CFDNet
from torch.utils.data import DataLoader, Dataset

class PDESolutionDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        
    def __len__(self):
        return len(self.input_data)
        
    def __getitem__(self, idx):
        return torch.tensor(self.input_data[idx]), torch.tensor(self.output_data[idx])

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        output = model(inputs)
        
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(data_loader)

if __name__ == "__main__":
    # Example input-output pairs (mock-up for PDE)
    input_data = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]  # Example feature input for PDE (e.g., x, t)
    output_data = [[1, 1.5, 2], [1.2, 1.6, 2.2], [1.5, 2, 2.5]]  # Example solution of PDE
    
    # Define model
    vocab_size = 10  # Not actually used for PDE, but needed for CFDNet
    d_model = 64
    max_seq_len = 3  # Size of the input sequence (e.g., coordinates or time steps)
    num_layers = 2
    model = CFDNet(vocab_size, d_model, num_layers, max_seq_len)
    
    # DataLoader setup
    dataset = PDESolutionDataset(input_data, output_data)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    
    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loss = train(model, data_loader, optimizer, criterion, device)
    
    print(f"Training loss: {train_loss:.4f}")
