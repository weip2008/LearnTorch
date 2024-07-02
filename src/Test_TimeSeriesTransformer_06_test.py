import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import time

# Step 1: Load and process the CSV data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path, header=None, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"Error parsing the file: {e}")
        return [], []

    low_data = []
    high_data = []
    
    for row in data.values:
        prefix = row[:2]
        sequence = row[2:]
        
        if prefix.tolist() == [1, 0]:
            low_data.append(sequence[~pd.isna(sequence)].tolist())
        elif prefix.tolist() == [0, 1]:
            high_data.append(sequence[~pd.isna(sequence)].tolist())
    
    return low_data, high_data

# Step 2: Create a custom dataset for variable-length sequences
class VariableLengthTimeSeriesDataset(Dataset):
    def __init__(self, data, batch_size):
        self.data = self.normalize_data(data)
        self.batch_size = batch_size
        self.batches = self._create_batches()

    def normalize_data(self, data):
        # Normalize data to the range [0, 1]
        normalized_data = []
        for seq in data:
            seq_array = np.array(seq)
            if np.max(seq_array) - np.min(seq_array) > 0:
                normalized_seq = (seq_array - np.min(seq_array)) / (np.max(seq_array) - np.min(seq_array))
            else:
                normalized_seq = seq_array
            normalized_data.append(normalized_seq.tolist())
        return normalized_data

    def _create_batches(self):
        lengths = [len(seq) for seq in self.data]
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        
        batches = []
        for i in range(0, len(sorted_indices), self.batch_size):
            batch_indices = sorted_indices[i:i+self.batch_size]
            batches.append([self.data[idx] for idx in batch_indices])
        
        return batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        batch = self.batches[index]
        return collate_fn(batch)

# Step 3: Collate function to pad sequences and create masks
def collate_fn(batch):
    sequences = [torch.tensor(item, dtype=torch.float32) for item in batch]
    padded_sequences = rnn_utils.pad_sequence(sequences, batch_first=True)
    mask = torch.zeros(padded_sequences.size(0), padded_sequences.size(1), dtype=torch.bool)

    for i, seq_len in enumerate([len(seq) for seq in sequences]):
        mask[i, seq_len:] = True

    return padded_sequences, mask

# Transformer model definition
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

# Function to create subsequent mask
def create_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask

# Function to load testing data
def load_testing_data(file_path):
    return load_data(file_path)

# Function to evaluate the model
def evaluate_model(model, dataloaders, criterion):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for (low_batch, low_mask), (high_batch, high_mask) in zip(dataloaders[0], dataloaders[1]):
            for batch, mask in [(low_batch, low_mask), (high_batch, high_mask)]:
                batch = batch.unsqueeze(-1)  # Adding feature dimension
                tgt_input = batch[:, :-1, :]
                tgt_output = batch[:, 1:, :]

                tgt_subsequent_mask = create_subsequent_mask(tgt_input.size(1)).to(tgt_input.device)

                output = model(
                    batch, tgt_input, tgt_mask=tgt_subsequent_mask, 
                    src_key_padding_mask=mask, tgt_key_padding_mask=mask[:, :-1]
                )
                output = output.reshape(-1, output.size(-1))
                tgt_output = tgt_output.reshape(-1, tgt_output.size(-1))
                loss = criterion(output, tgt_output)
                total_loss += loss.item()
                num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss

# Load training data from CSV
print("1. Load training data")
low_data_train, high_data_train = load_data('data/SPX_TrainingData_200.csv')
print(f"    Loaded {len(low_data_train)} low_data and {len(high_data_train)} high_data.")

# Create datasets and dataloaders for training
print("2. Create training datasets and dataloaders")
batch_size = 32
low_train_dataset = VariableLengthTimeSeriesDataset(low_data_train, batch_size)
high_train_dataset = VariableLengthTimeSeriesDataset(high_data_train, batch_size)

low_train_dataloader = DataLoader(low_train_dataset, batch_size=None, shuffle=False)
high_train_dataloader = DataLoader(high_train_dataset, batch_size=None, shuffle=False)

# Initialize model
print("3. Initializing model...")
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

# Training loop
print("4. Training Loop")
num_epochs = 10
model.train()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    epoch_loss = 0.0
    print(f"Starting epoch {epoch+1}/{num_epochs}...")
    for (low_batch, low_mask), (high_batch, high_mask) in zip(low_train_dataloader, high_train_dataloader):
        for batch, mask in [(low_batch, low_mask), (high_batch, high_mask)]:
            batch = batch.unsqueeze(-1)  # Adding feature dimension
            tgt_input = batch[:, :-1, :]
            tgt_output = batch[:, 1:, :]

            tgt_subsequent_mask = create_subsequent_mask(tgt_input.size(1)).to(tgt_input.device)

            optimizer.zero_grad()
            output = model(
                batch, tgt_input, tgt_mask=tgt_subsequent_mask, 
                src_key_padding_mask=mask, tgt_key_padding_mask=mask[:, :-1]
            )
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1, tgt_output.size(-1))
            loss = criterion(output, tgt_output)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
    avg_loss = epoch_loss / (len(low_train_dataloader) + len(high_train_dataloader))  # Adjusting for two dataloaders
    epoch_end_time = time.time()
    epoch_duration = (epoch_end_time - epoch_start_time) / 60 # convert to minutes
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f} minutes')

# Save the model
torch.save(model.state_dict(), 'timeseries_transformer.pth')
print("Model saved successfully.")

# Load the testing data
print("Loading testing data...")
low_data_test, high_data_test = load_testing_data('data/SPX_TestingData_201.csv')
print(f"    Loaded {len(low_data_test)} low_data and {len(high_data_test)} high_data.")

# Create datasets and dataloaders for testing
low_test_dataset = VariableLengthTimeSeriesDataset(low_data_test, batch_size)
high_test_dataset = VariableLengthTimeSeriesDataset(high_data_test, batch_size)

low_test_dataloader = DataLoader(low_test_dataset, batch_size=None, shuffle=False)
high_test_dataloader = DataLoader(high_test_dataset, batch_size=None, shuffle=False)

# Evaluate the model
print("Evaluating model...")
avg_loss = evaluate_model(model, (low_test_dataloader, high_test_dataloader), criterion)
print(f"Test Loss: {avg_loss:.4f}")
