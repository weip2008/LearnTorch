import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Example variable-length data
data = [np.random.rand(np.random.randint(10, 1000), 5) for _ in range(100)]
targets = np.random.rand(100)

# Create a custom dataset
class VariableLengthDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

# Custom collate function for dynamic batching
def collate_fn(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    data, targets = zip(*batch)
    lengths = [seq.shape[0] for seq in data]
    padded_data = nn.utils.rnn.pad_sequence(data, batch_first=True)
    targets = torch.stack(targets)
    return padded_data, targets, lengths

dataset = VariableLengthDataset(data, targets)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Pack the padded sequences
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        output = self.fc(h_n[-1])
        return output

# Instantiate the model, define the loss function and the optimizer
model = LSTMModel(input_size=5, hidden_size=50, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    for inputs, targets, lengths in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
