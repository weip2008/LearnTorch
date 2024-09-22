import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
from datetime import datetime



training_file_path = 'data/SPX_1m_TrainingData_HL_80_400.txt'
testing_file_path  = 'data/SPX_1m_TestingData_HL_80_400.txt'
save_path = 'GRU_model_with_LH_fixlen_data_205.pth'

import numpy as np


def load_training_data(training_file_path):
    data = []
    signals = []

    with open(training_file_path, 'r') as file:
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


def load_testing_data(training_file_path):
    data = []
    signals = []

    with open(training_file_path, 'r') as file:
        for line in file:
            # Split the line into data and target parts
            signals_part, data_part = line.strip().split(',[')
            
            signal = int(signals_part[0])
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


# Example usage
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"1.1 Load training data from {training_file_path}")
training_data, training_signals = load_training_data(training_file_path)

print("Data shape:", training_data.shape)
print("Targets shape:", training_signals.shape)
#print(data)
#print(signals)

print(f"1.2 Load testing data from {training_file_path}")
testing_data, testing_signals = load_testing_data(testing_file_path)

print("Data shape:", testing_data.shape)
print("Targets shape:", testing_signals.shape)

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
training_dataset = TimeSeriesDataset(training_data, training_signals)
testing_dataset = TimeSeriesDataset(testing_data, testing_signals)

# Create DataLoader for batching
batch_size = 32  # You can change the batch size as needed
# Training dataloader with shuffling
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
# Testing dataloader without shuffling
testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

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

# Hyperparameters
learning_rate = 0.001
num_epochs = 10

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# List to store losses
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    epoch_loss = 0
    for inputs, targets in training_dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_epoch_loss = epoch_loss / len(training_dataloader)
    train_losses.append(avg_epoch_loss)
    #print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}')
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time  # Duration in seconds
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}, Duration: {epoch_duration:.2f} seconds')
    
    
    # Testing loss
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for test_inputs, test_targets in testing_dataloader:
            test_outputs = model(test_inputs)
            test_loss += criterion(test_outputs, test_targets).item()
    avg_test_loss = test_loss / len(testing_dataloader)
    test_losses.append(avg_test_loss)
    print(f'    Testing Loss: {avg_test_loss:.6f}')
    
    
# Save the model, optimizer state, and losses
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"5. Save the model, optimizer state, and losses to {save_path}")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'test_losses': test_losses
}, save_path)


print(f"Training model saved to {save_path}")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")