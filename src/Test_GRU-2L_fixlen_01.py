import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
from datetime import datetime

training_file_path = 'data\SPX_30m_TrainingData_FixLenGRU_150_1001.txt'
model_save_path = 'GRU_2layer_fixlen_30m_150_1001.pth'

# Define the function to load data
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
    
    # Convert lists to numpy arrays
    data = np.array(data)
    targets = np.array(targets)
    
    return data, targets

# Define the custom dataset
class FixedLengthDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

# Define the GRU model with 2 layers
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initialize hidden state
        output, h_n = self.gru(x, h0)
        output = self.fc(h_n[-1])
        return output

print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"1. Load training data from {training_file_path}")
# Load the training data
#training_file_path = 'data/SPX_TrainingData_FixLenGRU_180_900.txt'
train_data, train_targets = load_data(training_file_path)

print("Data shape:", train_data.shape)
print("Targets shape:", train_targets.shape)


# Create DataLoader for training data
print("2. Define dataset and dataloader")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
train_dataset = FixedLengthDataset(train_data, train_targets)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)


# Initialize the model, loss function, and optimizer
print("3. Instantiate the model, define the loss function and the optimizer")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
model = GRUModel(input_size=3, hidden_size=50, output_size=3)  # Change input_size to 3
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Adjust learning rate every 10 epochs

# Training loop
print("4. Start training loop")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

''' ### Changes Made:
1. **Learning Rate Scheduler**: Added `scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)` to decrease the learning rate every 10 epochs.
2. **Early Stopping**: Implemented early stopping to stop training if there's no improvement in validation loss for a given number of epochs (`patience`).
 '''
''' num_epochs = 20
losses = []
patience = 5  # Number of epochs to wait for improvement before stopping
best_loss = float('inf')
epochs_no_improve = 0
early_stop = False
model.train()  # Set the model to training mode

for epoch in range(num_epochs):
    if early_stop:
        print("Early stopping")
        break

    epoch_start_time = time.time()
    epoch_loss = 0.0
    num_batches = 0
    
    for inputs, targets in train_dataloader:
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
    scheduler.step()  # Update the learning rate
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            early_stop = True

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.8f}, Duration: {epoch_duration:.2f} seconds')
 '''
 
num_epochs = 20
losses = []
model.train()
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    epoch_loss = 0.0
    num_batches = 0
    
    for inputs, targets in train_dataloader:
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
print(f"5. Save the model, optimizer state, and losses to {model_save_path}")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': losses,
}, model_save_path)



