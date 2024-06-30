#
# Explanation for GRU Model:
#1.	Preprocess the Data:
#   o	Same as before, sequences are dynamically padded within each batch.
#2.	Create Dataset and DataLoader:
#   o	VariableLengthDataset class handles the data and targets.
#   o	DataLoader with a custom collate_fn handles dynamic batching and padding.
#3.	Define the Model:
#   o	A GRU model with a fully connected layer is defined.
#   o	pack_padded_sequence is used to handle the variable-length sequences during the forward pass.
#4.	Train the Model:
#   o	A standard training loop is used with a mean squared error loss function and the Adam optimizer.
#   o	Packed sequences and dynamic batching efficiently handle the variable-length sequences.
# By using a GRU model, you benefit from a simpler architecture compared to LSTM, 
# often resulting in faster training times while still effectively handling variable-length sequential data.
#
#To increase the batch size in the GRU model example, you need to modify the batch_size parameter 
#   in the DataLoader. Here’s how you can do it:
#Steps:
#1.	Adjust the batch_size parameter in the DataLoader.
#2.	Optionally, you can also adjust other hyperparameters if necessary, such as the learning rate 
#   or the number of epochs, to accommodate the larger batch size.



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt

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
# Increase batch size from 16 to 64
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Pack the padded sequences
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
        packed_output, h_n = self.gru(packed_input)
        output = self.fc(h_n[-1])
        return output

# Instantiate the model, define the loss function and the optimizer
model = GRUModel(input_size=5, hidden_size=50, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce LR every 5 epochs

# Training loop with scheduler and loss tracking
num_epochs = 20
losses = []
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, targets, lengths in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()  # Adjust the learning rate
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')

# Plotting the training loss over epochs
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()
