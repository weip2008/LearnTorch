import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
from datetime import datetime



file_path = 'data/SPX_TrainingData_FixLenGRU_180_900.txt'
save_path = 'GRU_model_with_fixed_length_data_903.pth'

def load_data(file_path):
    data = []
    targets = []

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into data and target parts
            data_part, target_part = line.strip().split('][')
            
            # Add the closing bracket to the data part and opening bracket to the target part
            data_part += ']'
            target_part = '[' + target_part
            
            # Convert the string representations to actual lists
            data_row = eval(data_part)
            target_row = eval(target_part)
            
            # Append to the respective lists
            data.append(data_row)
            targets.append(target_row)
            #targets.append(target_row[0])  # Ensure target_row is a 1D array
    
    # Convert lists to numpy arrays
    data = np.array(data)
    targets = np.array(targets)
    
    return data, targets

# Example usage
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"1. Load training data from {file_path}")
data, targets = load_data(file_path)

print("Data shape:", data.shape)
print("Targets shape:", targets.shape)
#print(targets)


# Create a custom dataset
class FixedLengthDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

print("2. Define dataset and dataloader")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

dataset = FixedLengthDataset(data, targets)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, h_n = self.gru(x)
        output = self.fc(h_n[-1])
        return output

# Instantiate the model, define the loss function and the optimizer
print("3. Instantiate the model, define the loss function and the optimize")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

model = GRUModel(input_size=3, hidden_size=100, output_size=3)  # Output size is now 3
#model = GRUModel(input_size=3, hidden_size=120, output_size=3)  # Output size is now 3
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1.5e-4)

# Training loop
print("4. Start training loop")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

num_epochs = 50
losses = []
model.train()
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    epoch_loss = 0.0
    num_batches = 0
    
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    losses.append(avg_loss)
    epoch_end_time = time.time()
    epoch_duration = (epoch_end_time - epoch_start_time)  # convert to minutes
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.8f}, Duration: {epoch_duration:.2f} seconds')

# Save the model, optimizer state, and losses
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"5. Save the model, optimizer state, and losses to {save_path}")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': losses
}, save_path)

print(f"Training results saved to {save_path}")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")