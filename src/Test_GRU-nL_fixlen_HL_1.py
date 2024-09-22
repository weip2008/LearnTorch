import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
from datetime import datetime



file_path = 'data/SPX_1m_TrainingData_HL_80_400.txt'
save_path = 'GRU_model_with_LH_fixlen_data_205.pth'

import numpy as np


def load_training_data(file_path):
    data = []
    signals = []

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into data and target parts
            signals_part, data_part = line.strip().split(',[')
            
            signal = int(signals_part.split(',')[0])
            signals.append(signal)
            
            # Add the beginning bracket to the data part and opening bracket to the target part
            data_part = '[' + data_part
            
            # Convert the string representations to actual lists
            data_row = eval(data_part)
            
            # Append to the respective lists
            data.append(data_row)
            #targets.append(target_row[0])  # Ensure target_row is a 1D array
    
    # Convert lists to numpy arrays
    data_np = np.array(data)
    signals_np = np.array(signals).reshape(-1, 1)  # Reshape to (6883, 1)
    
    return data_np, signals_np

def load_training_data1(file_path):
    data = []
    signals = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into signal part and data part
            parts = line.strip().split('), (')
            signal = int(parts[0].split(',')[0])
            signals.append(signal)
            
            # Process the tuples
            tuples = []
            for part in parts[1:]:
                part = part.replace('(', '').replace(')', '')
                elements = part.split(',')
                elements = [float(e) for e in elements]  # Convert each element to float
                tuples.append(elements)
            
            data.append(tuples)
    
    # Convert lists to numpy arrays
    signals = np.array(signals)
    data = np.array([np.array(row, dtype=float) for row in data], dtype=float)
    
    return data, signals


# Example usage
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"1. Load training data from {file_path}")
data, signals = load_training_data(file_path)

print("Data shape:", data.shape)
print("Targets shape:", signals.shape)
#print(data)
#print(signals)



# Custom dataset class for loading signals and data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, signals):
        self.data = data
        self.signals = signals

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.signals[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Instantiate the dataset
print("2. Define dataset and dataloader")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
dataset = TimeSeriesDataset(data, signals)

# Create DataLoader for batching
batch_size = 32  # You can change the batch size as needed
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the GRU model with 2 layers
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Instantiate the model, define the loss function and the optimizer
print("3. Instantiate the model, define the loss function and the optimize")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Define hyperparameters
input_size = 5  # Number of features in each tuple
hidden_size = 64  # Number of features in the hidden state
output_size = 1  # Number of output features (signal)
num_layers = 5    # Number of GRU layers
learning_rate = 0.0001  # Learning rate

# Instantiate the model
print(f"Number of layers: {num_layers}")
model = GRUModel(input_size, hidden_size, output_size, num_layers)

# Move model to GPU if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

# Loss function: Binary Cross Entropy Loss
#criterion = nn.BCEWithLogitsLoss()  # Use with sigmoid for binary classification
criterion = nn.MSELoss()

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

# Training loop
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    epoch_loss = 0
    for inputs, targets in dataloader:
        
        # Move tensors to the appropriate device
        #inputs, targets = inputs.to(model.device), targets.to(model.device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_epoch_loss = epoch_loss / len(dataloader)
    losses.append(avg_epoch_loss)
    #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time  # Duration in seconds
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}, Duration: {epoch_duration:.2f} seconds')

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