import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset
import random
import matplotlib.pyplot as plt


# Function to generate synthetic data
def generate_data(num_samples, min_len, max_len, input_size):
    data = []
    for _ in range(num_samples):
        seq_len = random.randint(min_len, max_len)
        src = [random.random() for _ in range(seq_len)]
        tgt = [x + 0.1 for x in src]  # Target is just a shifted version of source for simplicity
        data.append((src, tgt))
    return data

# Sample dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        src, tgt = self.data[index]
        return torch.tensor(src, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)

# Collate function to pad sequences and create masks
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lengths = [len(src) for src in src_batch]
    tgt_lengths = [len(tgt) for tgt in tgt_batch]

    src_padded = rnn_utils.pad_sequence(src_batch, batch_first=True)
    tgt_padded = rnn_utils.pad_sequence(tgt_batch, batch_first=True)

    src_mask = (src_padded == 0)
    tgt_mask = (tgt_padded == 0)

    return src_padded, tgt_padded, src_mask, tgt_mask

# Generate data
num_samples = 1000
min_len = 10
max_len = 100
input_size = 1
data = generate_data(num_samples, min_len, max_len, input_size)

dataset = TimeSeriesDataset(data)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, output_size, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(
            src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )
        output = self.fc_out(output)
        return output

# Hyperparameters
input_size = 1
d_model = 64
nhead = 4
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 128
output_size = 1
dropout = 0.1

model = TimeSeriesTransformer(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, output_size, dropout)

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to create subsequent mask
def create_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask

# Training loop
num_epochs = 10
losses = []
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for src, tgt, src_mask, tgt_mask in dataloader:
        src, tgt = src.unsqueeze(-1), tgt.unsqueeze(-1)  # Adding feature dimension
        tgt_input = tgt[:, :-1, :]
        tgt_output = tgt[:, 1:, :]

        tgt_subsequent_mask = create_subsequent_mask(tgt_input.size(1)).to(tgt_input.device)

        optimizer.zero_grad()
        output = model(
            src, tgt_input, tgt_mask=tgt_subsequent_mask, 
            src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask[:, :-1]
        )

        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1, tgt_output.size(-1))
        loss = criterion(output, tgt_output)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')


# Plotting the training loss over epochs
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()


