import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
from datetime import datetime



file_path = 'data/SPX_1m_TrainingData_HL_80_200.txt'
save_path = 'GRU_model_with_LH_fixlen_data_200.pth'

import numpy as np

def load_data(file_path):
    data = []
    signals = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Remove any surrounding whitespace and newline characters
            line = line.strip()
            
            # Split the line into 3 parts: signal, discard second number, and data part
            parts = line.split(',', 2)  # Split into 3 parts: signal, discard, and the rest
            
            # Extract the first part as the signal (keep the first number only)
            data_signal = int(parts[0])  # Either '1' or '0'
            
            # The third part is the data series (normalized data)
            data_part = parts[2]  # The rest of the line after dropping the second number
            
            # Convert the data part (comma-separated string) to a list of floats
            data_row = [float(x) for x in data_part.split(',')]
            
            # Append the signal and data row to respective lists
            signals.append(data_signal)
            data.append(data_row)
            
    # Convert lists to NumPy arrays
    signals_np = np.array(signals).reshape(-1, 1)  # Reshape to (6883, 1)
    data_np = np.array(data)  # Should already be of shape (6883, 80)
    
    return data_np, signals_np

# Example usage
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"1. Load training data from {file_path}")
data, signals = load_data(file_path)

print("Data shape:", data.shape)
print("Targets shape:", signals.shape)
print(data)
print(signals)



# Custom dataset class for loading signals and data
class TimeSeriesDataset(Dataset):
    def __init__(self, signals, data):
        self.signals = signals
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return the data and its corresponding signal
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.signals[idx], dtype=torch.float32)

# Instantiate the dataset
dataset = TimeSeriesDataset(signals, data)

# Create DataLoader for batching
batch_size = 32  # You can change the batch size as needed
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):  # Default set to 2 layers now
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer with multiple layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer to map hidden state to output
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state with zeros for all layers
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Pass through GRU
        out, _ = self.gru(x, h0)
        
        # Pass the last hidden state output to the fully connected layer
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out

# Instantiate the model, define the loss function and the optimizer
print("3. Instantiate the model, define the loss function and the optimize")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Define hyperparameters
input_size = 80  # Number of features (length of each input sequence)
hidden_size = 64  # Number of hidden units in the GRU
output_size = 1   # Output size (1 for binary classification)
num_layers = 2    # Number of GRU layers
learning_rate = 0.0001  # Learning rate

# Instantiate the model
model = GRUModel(input_size, hidden_size, output_size, num_layers)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function: Binary Cross Entropy Loss
criterion = nn.BCEWithLogitsLoss()  # Use with sigmoid for binary classification

# Optimizer: Adam
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Add a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR by 10x every 10 epochs

# Training loop
print("4. Start training loop")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Define a list to keep track of losses over epochs
losses = []

# Training loop
num_epochs = 50  # Change as needed

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    for data, signals in dataloader:
        # Move data and signals to the same device as the model
        data, signals = data.to(device), signals.to(device)
        
        # Reshape data for GRU input (batch_size, sequence_length, input_size)
        data = data.unsqueeze(1)  # Add a time dimension since GRU expects 3D input
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, signals)
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Zero the gradient buffers
        loss.backward()  # Compute the gradients
        optimizer.step()  # Update the model parameters
        
        running_loss += loss.item() * data.size(0)
    
    # Compute epoch loss
    epoch_loss = running_loss / len(dataset)
    losses.append(epoch_loss)  # Store the loss for this epoch
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')

# Save the model, optimizer state, and losses
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"5. Save the model, optimizer state, and losses to {save_path}")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': losses
}, save_path)


print(f"Training model saved to {save_path}")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")