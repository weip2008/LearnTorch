import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

# Example fixed-length data
fixed_length = 120
data = [np.random.rand(fixed_length, 5) for _ in range(1000)]
targets = np.random.rand(1000, 3)  # Each target is an array of length 3

# Create a custom dataset
class FixedLengthDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

dataset = FixedLengthDataset(data, targets)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the GRU model with dropout
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0):  # Set dropout to 0 for single layer
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, h_n = self.gru(x)
        h_n = self.dropout(h_n[-1])
        output = self.fc(h_n)
        return output

model = GRUModel(input_size=5, hidden_size=50, output_size=3, dropout=0.0)  # Ensure dropout is 0 for single layer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# Weight initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                param.data.fill_(0.01)

model.apply(init_weights)

# Training loop
num_epochs = 20
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
    scheduler.step(avg_loss)  # Adjust the learning rate based on the average loss
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time  # Duration in seconds
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.8f}, Duration: {epoch_duration:.2f} seconds')

# Save the model, optimizer state, and losses
save_path = 'GRU_model_with_fixed_length_data.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': losses
}, save_path)

print(f"Training results saved to {save_path}")
